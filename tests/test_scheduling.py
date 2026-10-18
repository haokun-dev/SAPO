"""Analytical DAG scheduling tests; all durations are supplied cost units."""

import pytest
import torch

from sapo_core.scheduling import build_schedule


def test_fork_join_uses_link_latency_and_critical_path():
    schedule = build_schedule(
        compute_times=[[2, 2], [3, 3], [1, 1]],
        edges=[[0, 2], [1, 2]],
        mapping=[0, 1, 0],
        alpha=[[0, 0.5], [0.5, 0]],
        beta=[[0, 0.25], [0.25, 0]],
        payloads=[4, 4],
    )
    assert schedule.order == (0, 1, 2)
    assert schedule.starts == pytest.approx((0, 0, 4.5))
    assert schedule.finishes == pytest.approx((2, 3, 5.5))
    assert schedule.communication_times == pytest.approx((0, 1.5))
    assert schedule.makespan == pytest.approx(5.5)


def test_shared_device_serializes_while_other_device_runs_in_parallel():
    schedule = build_schedule(
        compute_times=[[2, 2], [3, 3], [4, 4]],
        edges=[],
        mapping=[0, 0, 1],
        alpha=[[0, 0], [0, 0]],
        beta=[[0, 0], [0, 0]],
        payloads=[],
    )
    assert schedule.order == (0, 1, 2)
    assert schedule.starts == pytest.approx((0, 2, 0))
    assert schedule.finishes == pytest.approx((2, 5, 4))
    assert schedule.makespan == pytest.approx(5)
    assert schedule.communication_times == ()


def test_inactive_nodes_and_edges_are_explicitly_excluded():
    schedule = build_schedule(
        compute_times=[[2, 2], [3, 3], [1, 1]],
        edges=[[0, 2], [1, 2]],
        mapping=[-1, 1, 0],
        alpha=[[0, 0.5], [0.5, 0]],
        beta=[[0, 0.25], [0.25, 0]],
        payloads=[4, 4],
    )
    assert schedule.order == (1, 2)
    assert schedule.starts == (None, 0.0, 4.5)
    assert schedule.finishes == (None, 3.0, 5.5)
    assert schedule.communication_times == (None, 1.5)
    assert schedule.makespan == pytest.approx(5.5)

    empty = build_schedule([[1], [2]], [[0, 1]], [-1, -1], [[0]], [[0]], [1])
    assert empty.order == ()
    assert empty.starts == (None, None)
    assert empty.finishes == (None, None)
    assert empty.communication_times == (None,)
    assert empty.makespan == 0


def test_inactive_predecessor_does_not_change_active_device_order():
    schedule = build_schedule(
        compute_times=[[2], [3], [10]],
        edges=[[2, 0]],
        mapping=[0, 0, -1],
        alpha=[[0]],
        beta=[[0]],
        payloads=[1],
    )
    assert schedule.order == (0, 1)
    assert schedule.starts == (0.0, 2.0, None)
    assert schedule.finishes == (2.0, 5.0, None)
    assert schedule.communication_times == (None,)
    assert schedule.makespan == 5.0


def test_synthetic_demo_mapping_has_longer_path_than_max_load_plus_crossings():
    schedule = build_schedule(
        compute_times=[[2, 2 / 1.5]] * 3 + [[1, 1 / 1.5]],
        edges=[[0, 3], [1, 3], [2, 3]],
        mapping=[1, 1, 0, 0],
        alpha=[[0, 0.1], [0.1, 0]],
        beta=[[0, 0], [0, 0]],
        payloads=[1, 1, 1],
    )
    assert schedule.starts[3] == pytest.approx(2 / 1.5 + 2 / 1.5 + 0.1)
    assert schedule.makespan == pytest.approx(3.7666666666666666)
    assert schedule.makespan != pytest.approx(3.2)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"compute_times": [[float("nan")], [1]]}, "compute_times"),
        ({"compute_times": [[-1], [1]]}, "compute_times"),
        ({"edges": [[0, 2]]}, "out-of-range"),
        ({"edges": [[0, 1], [1, 0]], "payloads": [1, 1]}, "acyclic"),
        ({"edges": [[0, 1], [1, 0]], "mapping": [-1, -1], "payloads": [1, 1]}, "acyclic"),
        ({"edges": [[0, 1], [0, 1]], "payloads": [1, 1]}, "duplicate"),
        ({"edges": [[0.0, 1.0]]}, "integer"),
        ({"mapping": [0.0, 0.0]}, "integer"),
        ({"mapping": [0, 1]}, "mapping"),
        ({"mapping": [-2, 0]}, "mapping"),
        ({"alpha": [[float("inf")]]}, "alpha"),
        ({"beta": [[-1]]}, "beta"),
        ({"payloads": [-1]}, "payloads"),
    ],
)
def test_invalid_inputs_raise(override, match):
    arguments = dict(
        compute_times=[[1], [1]], edges=[[0, 1]], mapping=[0, 0],
        alpha=[[0]], beta=[[0]], payloads=[1],
    )
    arguments.update(override)
    with pytest.raises(ValueError, match=match):
        build_schedule(**arguments)


def test_stable_topological_order_with_torch_inputs():
    schedule = build_schedule(
        compute_times=torch.tensor([[1.0], [1.0], [1.0], [1.0]]),
        edges=torch.tensor([[0, 3], [1, 3]]),
        mapping=torch.tensor([0, 0, -1, 0]),
        alpha=torch.zeros(1, 1),
        beta=torch.zeros(1, 1),
        payloads=torch.zeros(2),
    )
    assert schedule.order == (0, 1, 3)
    assert schedule.starts == (0.0, 1.0, None, 2.0)
    assert schedule.makespan == 3.0
