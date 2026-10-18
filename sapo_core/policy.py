"""Factorized head-gating and graph-based placement policy.

Device indices are zero-based; inactive operators have mapping -1.
"""

from dataclasses import dataclass
from functools import lru_cache
import heapq

import torch
from torch import nn
from torch.nn import functional as F

from .distributions import FeasibleBernoulli, NonemptyBernoulli


@dataclass(frozen=True)
class Observation:
    state: torch.Tensor  # [state_dim]
    node_features: torch.Tensor  # [J, node_dim]
    device_features: torch.Tensor  # [M, device_dim]
    head_id: torch.Tensor  # [J], -1 means always active; otherwise 0..H-1
    edges: torch.Tensor  # [E, 2], directed source -> destination
    residency: torch.Tensor  # [J, M] Boolean, resident weight blocks
    memory_cost: torch.Tensor  # [J], nonnegative integer units (e.g. bytes)
    capacity: torch.Tensor  # [M], nonnegative integer units, same unit as cost


@dataclass(frozen=True)
class Action:
    gate: torch.Tensor  # [H] Boolean, at least one True
    mapping: torch.Tensor  # [J] integer, inactive operators use -1
    log_prob: torch.Tensor  # scalar joint gate + mapping log probability


def _topological_order(active: list[int], edges: torch.Tensor) -> tuple[list[int], list[list[int]]]:
    j_count = max(active, default=-1) + 1
    predecessors = [[] for _ in range(j_count)]
    successors = [[] for _ in range(j_count)]
    active_set = set(active)
    indegree = {j: 0 for j in active}
    for src, dst in edges.tolist():
        if src in active_set and dst in active_set:
            predecessors[dst].append(src)
            successors[src].append(dst)
            indegree[dst] += 1
    ready = [j for j in active if indegree[j] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        src = heapq.heappop(ready)
        order.append(src)
        for dst in successors[src]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(ready, dst)
    if len(order) != len(active):
        raise ValueError("edges must define a DAG")
    return order, predecessors


class GraphAttention(nn.Module):
    """One directed GAT layer: each node attends to itself and predecessors."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale = hidden_dim**-0.5

    def forward(self, x: torch.Tensor, order: list[int], predecessors: list[list[int]]) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        contextual = []
        positions = []
        for node in order:
            neighbors = [node, *predecessors[node]]
            score = (k[neighbors] * q[node]).sum(-1) * self.scale
            weights = F.softmax(score, dim=0)
            contextual.append((weights[:, None] * v[neighbors]).sum(0))
            positions.append(node)
        result = x.clone()
        if positions:
            result[positions] = F.elu(self.output(torch.cat((x[positions], torch.stack(contextual)), dim=-1)))
        return result


class Actor(nn.Module):
    """Nonempty gate plus GAT and autoregressive feasible placement decoder.

    Memory values must be exact nonnegative integers in a common unit, such as
    bytes. Gate probability is conditioned on nonempty gates whose active
    operators admit a complete feasible mapping. A lookahead feasibility test
    in every decoder mask keeps all sampled prefixes extendable. These exact
    checks can be exponential and target small reference scenarios.
    """

    def __init__(self, state_dim: int, node_dim: int, device_dim: int, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        if min(state_dim, node_dim, device_dim, num_heads, hidden_dim) <= 0:
            raise ValueError("all actor dimensions must be positive")
        self.state_dim, self.node_dim, self.device_dim = state_dim, node_dim, device_dim
        self.num_heads, self.hidden_dim = num_heads, hidden_dim
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh())
        self.gate_head = nn.Linear(hidden_dim, num_heads)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.device_encoder = nn.Linear(device_dim, hidden_dim)
        self.gat = GraphAttention(hidden_dim)
        self.decoder = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.query = nn.Linear(hidden_dim * 3, hidden_dim)
        self.memory_score = nn.Linear(1, 1, bias=False)

    def _validate(self, obs: Observation) -> None:
        if obs.node_features.ndim != 2 or obs.device_features.ndim != 2:
            raise ValueError("node_features and device_features must be 2-D")
        j, m = obs.node_features.shape[0], obs.device_features.shape[0]
        if j == 0:
            raise ValueError("operator graph must contain at least one node")
        if obs.state.shape != (self.state_dim,) or obs.node_features.shape != (j, self.node_dim):
            raise ValueError("state or node feature shape does not match actor dimensions")
        if obs.device_features.shape != (m, self.device_dim) or m == 0:
            raise ValueError("device feature shape does not match actor dimensions")
        if obs.head_id.shape != (j,) or obs.head_id.dtype not in (torch.int32, torch.int64):
            raise ValueError("head_id must be a length-J integer tensor")
        if bool(((obs.head_id < -1) | (obs.head_id >= self.num_heads)).any()):
            raise ValueError("head_id must be -1 or a valid gate index")
        for head in range(self.num_heads):
            if not bool((obs.head_id == head).any()):
                raise ValueError(f"gate head {head} has no operator node")
        if obs.edges.ndim != 2 or obs.edges.shape[1] != 2 or obs.edges.dtype not in (torch.int32, torch.int64):
            raise ValueError("edges must have shape [E, 2] and integer dtype")
        if obs.edges.numel() and bool(((obs.edges < 0) | (obs.edges >= j)).any()):
            raise ValueError("edge endpoint is outside the node range")
        if obs.edges.shape[0] != torch.unique(obs.edges, dim=0).shape[0]:
            raise ValueError("duplicate DAG edges are not allowed")
        if obs.residency.shape != (j, m) or obs.residency.dtype != torch.bool:
            raise ValueError("residency must be a Boolean [J, M] tensor")
        if obs.memory_cost.shape != (j,) or obs.capacity.shape != (m,):
            raise ValueError("memory_cost or capacity shape is invalid")
        tensors = (obs.state, obs.node_features, obs.device_features, obs.memory_cost, obs.capacity)
        if any(not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError("observation contains nonfinite values")
        parameter = next(self.parameters())
        for name, tensor in (("state", obs.state), ("node_features", obs.node_features),
                             ("device_features", obs.device_features)):
            if not tensor.is_floating_point() or tensor.dtype != parameter.dtype:
                raise ValueError(f"{name} must have actor floating-point dtype {parameter.dtype}")
        if obs.state.device != parameter.device:
            raise ValueError("observation and actor must be on the same device")
        for name, tensor in (("memory_cost", obs.memory_cost), ("capacity", obs.capacity)):
            if tensor.dtype == torch.bool or torch.is_complex(tensor):
                raise ValueError(f"{name} must use a real numeric dtype")
            if tensor.is_floating_point() and not bool((tensor == tensor.round()).all()):
                raise ValueError(f"{name} must use exact integer memory units")
        if bool((obs.memory_cost < 0).any()) or bool((obs.capacity < 0).any()):
            raise ValueError("memory costs and capacities must be nonnegative")
        devices = {t.device for t in (obs.state, obs.node_features, obs.device_features, obs.head_id,
                                      obs.edges, obs.residency, obs.memory_cost, obs.capacity)}
        if len(devices) != 1:
            raise ValueError("all observation tensors must be on one device")
        _topological_order(list(range(j)), obs.edges)

    @staticmethod
    def _completion_checker(obs: Observation):
        costs = tuple(int(x) for x in obs.memory_cost.tolist())
        resident = tuple(tuple(bool(x) for x in row) for row in obs.residency.tolist())

        @lru_cache(None)
        def possible(nodes: tuple[int, ...], capacities: tuple[int, ...]) -> bool:
            if not nodes:
                return True
            # Most constrained first cuts the reference search space.
            node = min(nodes, key=lambda j: (sum(resident[j][d] and costs[j] <= capacities[d]
                                                for d in range(len(capacities))), -costs[j], j))
            remaining = tuple(j for j in nodes if j != node)
            for d in range(len(capacities)):
                if resident[node][d] and costs[node] <= capacities[d]:
                    next_caps = list(capacities)
                    next_caps[d] -= costs[node]
                    if possible(remaining, tuple(next_caps)):
                        return True
            return False

        return possible

    def _feasible_gate_support(self, obs: Observation, possible, initial_cap: tuple[int, ...]) -> torch.Tensor:
        always = tuple(torch.nonzero(obs.head_id == -1, as_tuple=False).flatten().tolist())
        if not possible(always, initial_cap):
            raise ValueError("no nonempty gate has a feasible mapping")
        by_head = [tuple(torch.nonzero(obs.head_id == h, as_tuple=False).flatten().tolist())
                   for h in range(self.num_heads)]
        gates: list[list[bool]] = []

        def visit(head: int, nodes: tuple[int, ...], bits: list[bool]) -> None:
            if head == self.num_heads:
                if any(bits):
                    gates.append(bits.copy())
                return
            bits.append(False)
            visit(head + 1, nodes, bits)
            bits.pop()
            active_with_head = tuple(sorted((*nodes, *by_head[head])))
            if possible(active_with_head, initial_cap):
                bits.append(True)
                visit(head + 1, active_with_head, bits)
                bits.pop()

        visit(0, always, [])
        if not gates:
            raise ValueError("no nonempty gate has a feasible mapping")
        return torch.tensor(gates, dtype=torch.bool, device=obs.state.device)

    def _run(self, obs: Observation, gate: torch.Tensor | None, mapping: torch.Tensor | None,
             deterministic: bool = False) -> Action:
        self._validate(obs)
        j, m = obs.node_features.shape[0], obs.device_features.shape[0]
        possible = self._completion_checker(obs)
        initial_cap = tuple(int(x) for x in obs.capacity.tolist())
        state_embedding = self.state_encoder(obs.state)
        gate_logits = self.gate_head(state_embedding)
        if possible(tuple(range(j)), initial_cap):
            gate_dist = NonemptyBernoulli(gate_logits)
        else:
            support = self._feasible_gate_support(obs, possible, initial_cap)
            gate_dist = FeasibleBernoulli(gate_logits, support)
        if gate is None:
            gate = gate_dist.mode() if deterministic else gate_dist.sample()
        else:
            if gate.device != obs.state.device:
                raise ValueError("gate must be on the observation device")
            gate_dist.log_prob(gate)  # Validate before selecting active nodes.
        log_prob = gate_dist.log_prob(gate)
        active_mask = obs.head_id.eq(-1) | gate[obs.head_id.clamp(min=0)]
        active = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
        order, predecessors = _topological_order(active, obs.edges)
        sampling = mapping is None
        if not sampling:
            if mapping.shape != (j,) or mapping.dtype not in (torch.int32, torch.int64) or mapping.device != obs.state.device:
                raise ValueError("mapping must be a length-J integer tensor on the observation device")
            if bool((mapping[~active_mask] != -1).any()):
                raise ValueError("inactive operators must have mapping -1")
            if bool((mapping[active_mask] < 0).any()) or bool((mapping[active_mask] >= m).any()):
                raise ValueError("active operators need valid device indices")
        else:
            mapping = torch.full((j,), -1, dtype=torch.long, device=obs.state.device)

        if not order:
            return Action(gate, mapping, log_prob)
        node_embedding = self.gat(self.node_encoder(obs.node_features), order, predecessors)
        device_embedding = self.device_encoder(obs.device_features)
        hidden = state_embedding
        remaining_cap = initial_cap
        capacity_scale = torch.tensor(initial_cap, device=obs.state.device, dtype=obs.state.dtype).clamp_min(1)
        for index, node in enumerate(order):
            remaining_nodes = tuple(order[index + 1 :])
            feasible = []
            for d in range(m):
                if bool(obs.residency[node, d]) and int(obs.memory_cost[node]) <= remaining_cap[d]:
                    next_cap = list(remaining_cap)
                    next_cap[d] -= int(obs.memory_cost[node])
                    feasible.append(possible(remaining_nodes, tuple(next_cap)))
                else:
                    feasible.append(False)
            mask = torch.tensor(feasible, device=obs.state.device, dtype=torch.bool)
            if not bool(mask.any()):
                raise ValueError(f"operator {node} has no feasible device")
            if predecessors[node]:
                predecessor_context = device_embedding[mapping[predecessors[node]]].mean(0)
            else:
                predecessor_context = torch.zeros_like(hidden)
            query = self.query(torch.cat((hidden, node_embedding[node], predecessor_context)))
            score = (device_embedding * query).sum(-1) * (self.hidden_dim**-0.5)
            remaining_tensor = torch.tensor(remaining_cap, device=obs.state.device, dtype=obs.state.dtype)
            remaining_tensor = remaining_tensor / capacity_scale
            score = score + self.memory_score(remaining_tensor[:, None]).squeeze(-1)
            score = score.masked_fill(~mask, -torch.inf)
            distribution = torch.distributions.Categorical(logits=score)
            if sampling:
                chosen = score.argmax() if deterministic else distribution.sample()
                mapping[node] = chosen
            else:
                chosen = mapping[node]
                if int(chosen) < 0 or int(chosen) >= m or not bool(mask[int(chosen)]):
                    raise ValueError(f"operator {node} is assigned an infeasible device")
            log_prob = log_prob + distribution.log_prob(chosen)
            next_cap = list(remaining_cap)
            next_cap[int(chosen)] -= int(obs.memory_cost[node])
            remaining_cap = tuple(next_cap)
            hidden = self.decoder(torch.cat((node_embedding[node], device_embedding[int(chosen)])), hidden)
        return Action(gate, mapping, log_prob)

    def sample_action(self, obs: Observation, deterministic: bool = False) -> Action:
        """Sample a feasible action, or choose gate MAP and local decoder argmax."""
        return self._run(obs, None, None, deterministic=deterministic)

    def evaluate_actions(self, obs: Observation, gate: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
        return self._run(obs, gate, mapping).log_prob
