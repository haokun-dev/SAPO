"""Local DAG execution using the policy's operator-to-worker mapping.

Workers are logical indices backed by local ``torch.device`` values. Kernels
must already be placed on their intended device; this module transfers only
activation tensors and does not provide remote execution or collectives.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ExecutionRecord:
    node: int
    device_index: int
    device: torch.device


def execute_graph(
    edges: Tensor | Sequence[tuple[int, int]],
    mapping: Tensor | Sequence[int],
    kernels: Mapping[tuple[int, int], Callable[[tuple[Tensor, ...]], Tensor]],
    inputs: Mapping[int, tuple[Tensor, ...]],
    devices: Mapping[int, torch.device],
) -> tuple[dict[int, Tensor], list[ExecutionRecord]]:
    """Run active DAG nodes once, on precisely their assigned local workers.

    ``mapping[node] == -1`` skips that node. Each active node must have a
    preloaded ``kernels[(node, mapping[node])]``. A source kernel receives
    ``inputs[node]``; all other kernels receive active predecessor outputs in
    the original ``edges`` order. An active node whose predecessors were all
    skipped receives an empty tuple. Every argument is moved to that worker's
    device before invoking its kernel. Output device mismatches are errors.
    """
    assignments = torch.as_tensor(mapping)
    if assignments.ndim != 1 or assignments.numel() == 0 or assignments.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise ValueError("mapping must be a nonempty integer vector")
    assigned = assignments.tolist()
    if any(worker < -1 for worker in assigned):
        raise ValueError("mapping entries must be -1 or nonnegative worker indices")
    if not isinstance(devices, Mapping) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        or not isinstance(device, torch.device)
        for index, device in devices.items()
    ):
        raise ValueError("devices must map nonnegative integer worker indices to torch.device")
    if any(device.type == "cuda" and device.index is None for device in devices.values()):
        raise ValueError("CUDA worker devices require an explicit device index")
    for worker in assigned:
        if worker >= 0 and worker not in devices:
            raise ValueError(f"assigned worker {worker} is missing from devices")

    edge_tensor = torch.as_tensor(edges)
    if edge_tensor.numel() == 0:
        edge_pairs: list[tuple[int, int]] = []
    elif edge_tensor.ndim == 2 and edge_tensor.shape[1] == 2 and edge_tensor.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        edge_pairs = [tuple(pair) for pair in edge_tensor.tolist()]
    else:
        raise ValueError("edges must be integer (source, destination) pairs")
    count = len(assigned)
    if any(src < 0 or src >= count or dst < 0 or dst >= count for src, dst in edge_pairs):
        raise ValueError("edge endpoint is outside mapping")
    if len(set(edge_pairs)) != len(edge_pairs):
        raise ValueError("duplicate edges are unsupported")

    # Check the complete graph for cycles, then execute only active nodes.
    successors = [[] for _ in range(count)]
    predecessors = [[] for _ in range(count)]
    indegree = [0] * count
    for src, dst in edge_pairs:
        successors[src].append(dst)
        predecessors[dst].append(src)
        indegree[dst] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for dst in successors[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                heapq.heappush(ready, dst)
    if len(order) != count:
        raise ValueError("edges must define a DAG")

    # A skipped predecessor imposes no execution dependency. Scheduling uses
    # the active induced graph, so resolve ties there rather than filtering a
    # full-graph topological order afterward.
    active = {node for node, worker in enumerate(assigned) if worker >= 0}
    active_indegree = {node: sum(pred in active for pred in predecessors[node]) for node in active}
    active_ready = [node for node, degree in active_indegree.items() if degree == 0]
    heapq.heapify(active_ready)
    active_order = []
    while active_ready:
        node = heapq.heappop(active_ready)
        active_order.append(node)
        for dst in successors[node]:
            if dst in active:
                active_indegree[dst] -= 1
                if active_indegree[dst] == 0:
                    heapq.heappush(active_ready, dst)

    outputs: dict[int, Tensor] = {}
    records: list[ExecutionRecord] = []
    for node in active_order:
        worker = assigned[node]
        target = devices[worker]
        key = (node, worker)
        if key not in kernels or not callable(kernels[key]):
            raise ValueError(f"missing callable kernel for node {node} on worker {worker}")
        if predecessors[node]:
            arguments = tuple(outputs[pred].to(target) for pred in predecessors[node] if pred in outputs)
        else:
            if node not in inputs:
                raise ValueError(f"source node {node} has no external input")
            external = inputs[node]
            if not isinstance(external, tuple) or any(not isinstance(t, Tensor) for t in external):
                raise ValueError(f"source node {node} input must be a tuple of tensors")
            arguments = tuple(t.to(target) for t in external)
        result = kernels[key](arguments)
        if not isinstance(result, Tensor) or result.device != target:
            raise ValueError(f"kernel for node {node} must return a tensor on {target}")
        outputs[node] = result
        records.append(ExecutionRecord(node, worker, target))
    return outputs, records
