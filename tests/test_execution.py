import pytest
import torch

from sapo_core.execution import ExecutionRecord, execute_graph


DEVICES = {0: torch.device("cpu"), 1: torch.device("cpu")}


def test_mapping_dispatches_head_and_aggregate_kernels_on_assigned_workers():
    called = []
    weight = torch.nn.Parameter(torch.tensor(2.0))
    original_weight_id = id(weight)

    def head(node, worker):
        def kernel(args):
            called.append((node, worker, len(args)))
            return args[0] * weight if node == 0 else args[0] + 3
        return kernel

    def aggregate(args):
        called.append((2, 1, len(args)))
        return args[0] + args[1]

    kernels = {(0, 0): head(0, 0), (0, 1): head(0, 1),
               (1, 0): head(1, 0), (1, 1): head(1, 1), (2, 1): aggregate}
    outputs, records = execute_graph(
        [(0, 2), (1, 2)], [0, 1, 1], kernels,
        {0: (torch.tensor(2.),), 1: (torch.tensor(4.),)}, DEVICES)
    assert called == [(0, 0, 1), (1, 1, 1), (2, 1, 2)]
    assert records == [ExecutionRecord(0, 0, DEVICES[0]),
                       ExecutionRecord(1, 1, DEVICES[1]),
                       ExecutionRecord(2, 1, DEVICES[1])]
    assert outputs[2].item() == 11
    assert id(weight) == original_weight_id and weight.device == torch.device("cpu")
    outputs[2].backward()
    assert weight.grad.item() == 2


def test_inactive_predecessor_is_skipped_but_aggregate_still_executes():
    called = []
    kernels = {
        (1, 0): lambda args: args[0] + 1,
        (2, 1): lambda args: (called.append(len(args)) or args[0] * 2),
    }
    outputs, records = execute_graph(
        [(0, 2), (1, 2)], [-1, 0, 1], kernels,
        {1: (torch.tensor(3.),)}, DEVICES)
    assert list(outputs) == [1, 2]
    assert called == [1] and outputs[2].item() == 8
    assert [(r.node, r.device_index) for r in records] == [(1, 0), (2, 1)]


def test_stable_topology_is_computed_on_active_induced_graph():
    outputs, records = execute_graph(
        [(2, 0)], [0, 0, -1],
        {(0, 0): lambda args: torch.tensor(0.) if not args else args[0],
         (1, 0): lambda args: args[0]},
        {1: (torch.tensor(1.),)}, DEVICES)
    # Node 0 has a skipped predecessor and therefore runs with no arguments.
    # The inactive edge must not force unrelated active node 1 ahead of node 0.
    assert [(r.node, r.device_index) for r in records] == [(0, 0), (1, 0)]
    assert list(outputs) == [0, 1]


@pytest.mark.parametrize("mapping", [[0, 2], [-2, 0], [0.0, 1.0], [True, False]])
def test_rejects_invalid_mapping(mapping):
    with pytest.raises(ValueError):
        execute_graph([], mapping, {}, {}, DEVICES)


def test_missing_kernel_and_source_input_are_errors():
    with pytest.raises(ValueError, match="missing callable kernel"):
        execute_graph([], [0], {}, {0: (torch.tensor(1.),)}, DEVICES)
    with pytest.raises(ValueError, match="external input"):
        execute_graph([], [0], {(0, 0): lambda args: args[0]}, {}, DEVICES)


def test_cycle_and_invalid_edge_are_errors_even_if_node_is_inactive():
    with pytest.raises(ValueError, match="DAG"):
        execute_graph([(0, 1), (1, 0)], [-1, 0], {}, {}, DEVICES)
    with pytest.raises(ValueError, match="outside mapping"):
        execute_graph([(0, 2)], [0, 0], {}, {}, DEVICES)


def test_kernel_must_return_tensor_on_assigned_device():
    with pytest.raises(ValueError, match="must return a tensor"):
        execute_graph([], [0], {(0, 0): lambda args: 5}, {0: (torch.tensor(1.),)}, DEVICES)
    with pytest.raises(ValueError, match="must return a tensor"):
        execute_graph([], [0], {(0, 0): lambda args: torch.empty(1, device="meta")},
                      {0: (torch.tensor(1.),)}, DEVICES)


def test_cuda_worker_requires_explicit_index_before_execution():
    with pytest.raises(ValueError, match="explicit device index"):
        execute_graph([], [0], {}, {}, {0: torch.device("cuda")})
