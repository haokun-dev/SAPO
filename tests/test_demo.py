import torch

from sapo_core.demo import SyntheticLayerEnv
from sapo_core.policy import Action


def test_demo_executes_head_and_aggregate_mappings_and_updates_state():
    torch.manual_seed(11)
    env = SyntheticLayerEnv()
    before_obs = env.reset()
    before = env.features.clone()
    gate = torch.tensor([True, False, True])
    mapping = torch.tensor([1, -1, 0, 1])
    expected = before + env.layers[0](before, gate)
    schedule = env._schedule(mapping)
    observation, reward, terminal, truncated, _ = env.step(Action(gate, mapping, torch.tensor(0.)))
    torch.testing.assert_close(env.features, expected)
    assert reward == 0 and not terminal and not truncated
    assert env.execution_records == [(0, 0, 1), (0, 2, 0), (0, 3, 1)]
    assert env.total_cost == schedule.makespan
    assert observation.state.shape == (env.state_dim,)
    assert not torch.equal(observation.capacity, before_obs.capacity)
    assert not torch.equal(observation.device_features, before_obs.device_features)
    _, reward, terminal, truncated, info = env.step(Action(gate, mapping, torch.tensor(0.)))
    assert terminal and not truncated
    assert info['executed_nodes'] == 6
    assert reward == info['terminal_reward']
