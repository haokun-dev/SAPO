"""Deterministic critical-path cost for a mapped SAPO operator DAG.

This is an analytical scheduler over supplied costs, not a hardware profiler.
Operators sharing a device run without overlap in the active subgraph's stable
topological order.
Transfers on different edges contribute independent dependency delays; link
contention is outside this model.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class ExecutionSchedule:
    """A complete schedule for active nodes of the supplied graph.

    ``order`` contains active node indices in the active subgraph's stable
    topological order (smallest ready index first).
    ``starts`` and ``finishes`` have one entry per input node, and
    ``communication_times`` has one entry per input edge. ``None`` marks an
    inactive node or an edge incident to one; zero denotes an active edge whose
    endpoints share a device. All numerical times use the caller's cost units.
    """

    order: tuple[int, ...]
    starts: tuple[float | None, ...]
    finishes: tuple[float | None, ...]
    makespan: float
    communication_times: tuple[float | None, ...]


def _tensor(value: Tensor | object, name: str) -> Tensor:
    try:
        return torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc


def _indices(value: Tensor | object, name: str, shape: tuple[int, ...]) -> Tensor:
    array = _tensor(value, name)
    if name == "edges" and array.numel() == 0:
        array = array.to(torch.int64)
        if array.ndim == 1:
            array = array.reshape(0, 2)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must contain integer indices")
    return array.to(torch.int64)


def _costs(value: Tensor | object, name: str, shape: tuple[int, ...]) -> Tensor:
    array = _tensor(value, name)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype == torch.bool or array.is_complex():
        raise ValueError(f"{name} must contain real numeric costs")
    array = array.to(torch.float64)
    if not bool(torch.isfinite(array).all()) or bool((array < 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative costs")
    return array


def build_schedule(
    compute_times: Tensor | object,
    edges: Tensor | object,
    mapping: Tensor | object,
    alpha: Tensor | object,
    beta: Tensor | object,
    payloads: Tensor | object,
) -> ExecutionSchedule:
    """Schedule a mapped DAG and return its makespan and per-node timing.

    Args:
        compute_times: Nonnegative offline profile table ``(J, M)``; entry
            ``[j, i]`` is operator ``j``'s duration on device ``i``.
        edges: Directed dependency indices ``(E, 2)``. The full graph must be
        acyclic and contain no duplicate edges. Edges incident to inactive
        nodes are excluded at runtime.
        mapping: Integer vector ``(J,)``: device ``0..M-1`` for active nodes,
            or ``-1`` for inactive nodes.
        alpha: Nonnegative pairwise link startup costs ``(M, M)``.
        beta: Nonnegative pairwise cost per payload unit ``(M, M)``.
        payloads: Nonnegative edge payloads ``(E,)``, in units consistent with
            ``beta``. Same-device edges have zero communication cost.

    Returns:
        ``ExecutionSchedule``. Every active operator starts after its active
        predecessors' finishes plus any transfer costs, and after the previous
        operator on its own device has finished. A graph with no active nodes
        has makespan zero. Costs are supplied estimates, not measured timings.
    """
    compute = _tensor(compute_times, "compute_times")
    if compute.ndim != 2 or compute.shape[1] < 1:
        raise ValueError("compute_times must have shape (J, M) with M >= 1")
    jobs, devices = compute.shape
    compute = _costs(compute, "compute_times", (jobs, devices))

    arcs = _tensor(edges, "edges")
    if arcs.numel() == 0 and arcs.ndim == 1:
        arcs = arcs.reshape(0, 2)
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    arcs = _indices(arcs, "edges", (arcs.shape[0], 2))
    edge_count = arcs.shape[0]
    assignment = _indices(mapping, "mapping", (jobs,))
    if bool((assignment < -1).any()) or bool((assignment >= devices).any()):
        raise ValueError("mapping entries must be -1 or valid device indices")
    startup = _costs(alpha, "alpha", (devices, devices))
    inverse_bandwidth = _costs(beta, "beta", (devices, devices))
    volumes = _costs(payloads, "payloads", (edge_count,))

    successors: list[list[int]] = [[] for _ in range(jobs)]
    incoming: list[list[int]] = [[] for _ in range(jobs)]
    indegree = [0] * jobs
    edge_pairs = arcs.tolist()
    seen_edges: set[tuple[int, int]] = set()
    for edge_index, (source, target) in enumerate(edge_pairs):
        if source < 0 or source >= jobs or target < 0 or target >= jobs:
            raise ValueError("edges contain an out-of-range node index")
        if (source, target) in seen_edges:
            raise ValueError("edges must not contain duplicate edges")
        seen_edges.add((source, target))
        successors[source].append(target)
        incoming[target].append(edge_index)
        indegree[target] += 1

    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    checked_nodes = 0
    while ready:
        node = heapq.heappop(ready)
        checked_nodes += 1
        for target in successors[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if checked_nodes != jobs:
        raise ValueError("edges must form a directed acyclic graph")

    active = assignment.tolist()
    active_successors: list[list[int]] = [[] for _ in range(jobs)]
    active_indegree = [0] * jobs
    for source, target in edge_pairs:
        if active[source] >= 0 and active[target] >= 0:
            active_successors[source].append(target)
            active_indegree[target] += 1
    active_ready = [node for node in range(jobs) if active[node] >= 0 and active_indegree[node] == 0]
    heapq.heapify(active_ready)
    topological: list[int] = []
    while active_ready:
        node = heapq.heappop(active_ready)
        topological.append(node)
        for target in active_successors[node]:
            active_indegree[target] -= 1
            if active_indegree[target] == 0:
                heapq.heappush(active_ready, target)

    edge_times: list[float | None] = [None] * edge_count
    for index, (source, target) in enumerate(edge_pairs):
        source_device, target_device = active[source], active[target]
        if source_device < 0 or target_device < 0:
            continue
        if source_device == target_device:
            edge_times[index] = 0.0
        else:
            duration = float(startup[source_device, target_device]) + (
                float(inverse_bandwidth[source_device, target_device]) * float(volumes[index])
            )
            if not math.isfinite(duration):
                raise ValueError("communication cost overflowed to a nonfinite value")
            edge_times[index] = duration

    starts: list[float | None] = [None] * jobs
    finishes: list[float | None] = [None] * jobs
    device_free = [0.0] * devices
    order: list[int] = []
    for node in topological:
        device = active[node]
        earliest = device_free[device]
        for edge_index in incoming[node]:
            source = int(arcs[edge_index, 0])
            transfer = edge_times[edge_index]
            if transfer is not None:
                assert finishes[source] is not None
                earliest = max(earliest, finishes[source] + transfer)
        finish = earliest + float(compute[node, device])
        if not math.isfinite(finish):
            raise ValueError("schedule duration overflowed to a nonfinite value")
        starts[node], finishes[node] = earliest, finish
        device_free[device] = finish
        order.append(node)

    return ExecutionSchedule(
        order=tuple(order),
        starts=tuple(starts),
        finishes=tuple(finishes),
        makespan=max(device_free, default=0.0),
        communication_times=tuple(edge_times),
    )
