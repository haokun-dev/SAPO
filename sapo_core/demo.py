"""Self-contained SAPO learning example using synthetic activations and costs."""
import argparse
import copy
import json

import torch
from torch import nn

from .attention import SparseHeadAttention
from .policy import Actor, Observation
from .ppo import PPOConfig, PPOTrainer, ValueNetwork, collect_episodes
from .reward import terminal_reward
from .state import build_state, device_state_features
from .scheduling import build_schedule
from .execution import execute_graph


class SyntheticLayerEnv:
    """Two attention layers with generated telemetry, profiles and link costs.

    Kernels are pre-positioned on two logical CPU workers. Every active DAG
    node, including aggregation, dispatches through its selected worker entry.
    Cost units are synthetic; the schedule models device compute serialization
    and independent cross-device edge delays alpha + beta * payload.
    """
    num_heads = 3
    depth = 2
    width = 12
    num_devices = 2
    state_dim = width + depth + num_devices * (num_devices + 1)

    def __init__(self):
        self.layers = nn.ModuleList([SparseHeadAttention(self.width, self.num_heads, qkv_bias=True)
                                     for _ in range(self.depth)]).eval()
        self.classifier = nn.Linear(self.width, 4).eval()
        self.worker_devices = {i: torch.device('cpu') for i in range(self.num_devices)}
        self.worker_layers = {i: copy.deepcopy(self.layers).eval() for i in self.worker_devices}
        self.head_id = torch.tensor([0, 1, 2, -1])
        self.edges = torch.tensor([[0, 3], [1, 3], [2, 3]])
        self.residency = torch.ones((4, 2), dtype=torch.bool)
        self.memory_cost = torch.tensor([2, 2, 2, 1], dtype=torch.int64)
        self.base_cost = torch.tensor([2., 2., 2., 1.])
        self.speeds = torch.tensor([1., 1.5])
        self.episode_metrics = []
        self.kernels = [self._kernels_for_layer(index) for index in range(self.depth)]

    def _kernels_for_layer(self, index):
        kernels = {}
        for worker in self.worker_devices:
            model = self.worker_layers[worker][index]
            for head in range(self.num_heads):
                mask = torch.zeros(self.num_heads, dtype=torch.bool)
                mask[head] = True
                assignment = torch.full((self.num_heads,), -1, dtype=torch.long)
                assignment[head] = worker
                def head_kernel(values, model=model, mask=mask, assignment=assignment, worker=worker):
                    return model.forward_partition(values[0], mask, assignment, worker)
                kernels[(head, worker)] = head_kernel
            def aggregate_kernel(values, bias=model.proj.bias):
                return torch.stack(values).sum(0) + bias
            kernels[(self.num_heads, worker)] = aggregate_kernel
        return kernels

    def _refresh_telemetry(self):
        # Generated snapshots change across layers. Real callers supply their
        # measured utilization, capacity, link bandwidth and offline profiles.
        self.utilization = torch.tensor([.2, .3]) + .05 * self.index
        self.capacity = torch.tensor([4 + self.index % 2, 4], dtype=torch.int64)
        self.bandwidth = torch.tensor([[0., 1200.], [800., 0.]]) * (1 + .1 * self.index)
        self.alpha = torch.tensor([[0., .1], [.12, 0.]])
        self.beta = torch.zeros_like(self.bandwidth)
        off_diagonal = ~torch.eye(self.num_devices, dtype=torch.bool)
        self.beta[off_diagonal] = 1 / self.bandwidth[off_diagonal]
        self.payloads = torch.full((len(self.edges),), self.features.numel() * self.features.element_size(), dtype=torch.float32)
        self.compute_times = self.base_cost[:, None] / (self.speeds * (1 - self.utilization))[None, :]
        self.node_features = torch.cat((torch.tensor([[1.], [1.], [1.], [0.]]), self.compute_times), dim=1)
        # Explicit feature scales: memory / 8 and bandwidth / 1000.
        self.device_features = device_state_features(self.utilization, self.capacity.float() / 8,
                                                      self.bandwidth / 1000)

    def _schedule(self, mapping):
        return build_schedule(self.compute_times, self.edges, mapping, self.alpha, self.beta, self.payloads)

    def _cost(self, mapping):
        return self._schedule(mapping).makespan

    def _observation(self):
        state = build_state(self.utilization, self.capacity.float() / 8, self.bandwidth / 1000,
                            self.features[0], self.index, self.depth)
        return Observation(state, self.node_features, self.device_features, self.head_id, self.edges,
                           self.residency, self.memory_cost, self.capacity)

    @torch.no_grad()
    def reset(self):
        self.index, self.total_cost, self.baseline_cost = 0, 0., 0.
        self.execution_records = []
        self.features = torch.randn(1, 5, self.width)
        full = self.features.clone()
        for layer in self.layers:
            full = full + layer(full)
        self.full_logits = self.classifier(full.mean(1))
        self._refresh_telemetry()
        return self._observation()

    @torch.no_grad()
    def step(self, action):
        if self.index >= self.depth:
            raise RuntimeError('Call reset before another episode')
        active = self.head_id.eq(-1) | action.gate[self.head_id.clamp_min(0)]
        if action.mapping.shape != self.head_id.shape or not torch.equal(action.mapping >= 0, active):
            raise ValueError('Action gate and operator mapping must agree')
        schedule = self._schedule(action.mapping)
        outputs, records = execute_graph(self.edges, action.mapping, self.kernels[self.index],
            {head: (self.features,) for head in range(self.num_heads)}, self.worker_devices)
        assert [record.node for record in records] == list(schedule.order)
        self.execution_records.extend((self.index, record.node, record.device_index) for record in records)
        self.features = self.features + outputs[self.num_heads]
        self.total_cost += schedule.makespan
        self.baseline_cost += self._cost(torch.tensor([0, 1, 0, 1]))
        self.index += 1
        terminated = self.index == self.depth
        reward, info = 0., {}
        if terminated:
            logits = self.classifier(self.features.mean(1))
            terms = terminal_reward(self.full_logits, logits, self.total_cost, self.baseline_cost)
            reward = terms.reward.item()
            info = {'output_kl_full_to_dynamic': terms.output_kl.item(),
                    'synthetic_cost_units': self.total_cost, 'terminal_reward': reward,
                    'executed_nodes': len(self.execution_records)}
            self.episode_metrics.append(info)
        self._refresh_telemetry()
        return self._observation(), reward, terminated, False, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 1 or args.seed < 0:
        parser.error('iterations must be positive and seed must be nonnegative')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = SyntheticLayerEnv()
    actor = Actor(env.state_dim, 3, 3, env.num_heads, hidden_dim=32)
    critic = ValueNetwork(env.state_dim, hidden_dim=32)
    trainer = PPOTrainer(actor, critic, PPOConfig(epochs=2, minibatch_size=4, seed=args.seed))
    history = []
    for iteration in range(args.iterations):
        transitions = collect_episodes(actor, critic, env, episodes=4, max_steps=env.depth)
        before = [param.detach().clone() for param in actor.parameters()]
        metrics = trainer.update(transitions)
        assert any(not torch.equal(old, new) for old, new in zip(before, actor.parameters()))
        history.append({'iteration': iteration + 1, 'transitions': len(transitions), **metrics})
    print(json.dumps({'example': 'synthetic activations and cost units', 'seed': args.seed,
                      'updates': history, 'last_collected_episode': env.episode_metrics[-1]}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
