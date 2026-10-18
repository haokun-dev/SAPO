import pytest
import torch
from sapo_core.state import build_state, device_state_features


def test_state_has_each_outgoing_link_and_sequence_mean():
    utilization = torch.tensor([.2, .3, .4])
    memory = torch.tensor([10, 20, 30])
    bandwidth = torch.tensor([[0., 12., 13.], [21., 0., 23.], [31., 32., 0.]])
    content = torch.tensor([[1., 2.], [3., 4.]])
    expected_devices = torch.tensor([[.2, 10., 12., 13.], [.3, 20., 21., 23.], [.4, 30., 31., 32.]])
    torch.testing.assert_close(device_state_features(utilization, memory, bandwidth), expected_devices)
    state = build_state(utilization, memory, bandwidth, content, 1, 3)
    torch.testing.assert_close(state, torch.cat((expected_devices.flatten(), torch.tensor([2., 3., 0., 1., 0.]))))
    changed = bandwidth.clone()
    changed[0, 2] = 99
    assert not torch.equal(state, build_state(utilization, memory, changed, content, 1, 3))


def test_terminal_layer_and_bad_telemetry():
    util, memory, bandwidth, content = torch.tensor([.5]), torch.tensor([1]), torch.zeros(1, 1), torch.ones(2, 3)
    state = build_state(util, memory, bandwidth, content, 2, 2)
    torch.testing.assert_close(state[-2:], torch.zeros(2))
    with pytest.raises(ValueError):
        build_state(torch.tensor([1.5]), memory, bandwidth, content, 0, 2)
    with pytest.raises(ValueError):
        build_state(util, memory, torch.ones(2, 2), content, 0, 2)
