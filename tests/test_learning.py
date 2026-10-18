from dataclasses import replace
import math

import pytest
import torch

from sapo_core.policy import Actor, Observation
from sapo_core.ppo import (PPOConfig, PPOTrainer, ValueNetwork, collect_episodes,
                           clipped_policy_loss, generalized_advantage_estimate)
from sapo_core.reward import terminal_reward


def scene():
    return Observation(torch.tensor([0.4, 0.1, -0.2]),
        torch.tensor([[1., 0.], [0., 1.]]), torch.eye(2),
        torch.tensor([0, 1]), torch.empty((0, 2), dtype=torch.long),
        torch.ones((2, 2), dtype=torch.bool), torch.ones(2, dtype=torch.long),
        torch.tensor([2, 2]))


def test_reward_formula_direction_and_teacher_detach():
    full = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64)).requires_grad_()
    candidate = torch.log(torch.tensor([[0.4, 0.6]], dtype=torch.float64)).requires_grad_()
    result = terminal_reward(full, candidate, 0.5, 1.)
    expected_kl = 0.8 * math.log(0.8 / 0.4) + 0.2 * math.log(0.2 / 0.6)
    expected_weight = 0.2 / (1 + math.exp(-8 * (0.08 - expected_kl)))
    assert result.output_kl.item() == pytest.approx(expected_kl)
    assert result.time_cost.item() == pytest.approx(math.expm1(0.5))
    assert result.reward.item() == pytest.approx(-expected_kl - expected_weight * math.expm1(0.5))
    assert result.output_kl.item() != pytest.approx(0.4 * math.log(0.4 / 0.8) + 0.6 * math.log(0.6 / 0.2))
    result.reward.sum().backward()
    assert full.grad is None
    assert torch.isfinite(candidate.grad).all()


def test_reward_weight_prefers_fidelity_and_rejects_bad_times():
    full = torch.tensor([[4., 0.]])
    identical = terminal_reward(full, full, 1., 1.)
    poor = terminal_reward(full, -full, 1., 1.)
    assert identical.output_kl == 0
    assert poor.latency_weight < identical.latency_weight
    for total, base in [(1, 0), (-1, 1), (float('nan'), 1), (1e10, 1)]:
        with pytest.raises(ValueError):
            terminal_reward(full, full, total, base)


def test_gae_matches_hand_calculation_with_terminal_and_truncation():
    # First episode terminates at t1. The next one truncates at t2 and
    # bootstraps V(next)=10; its signal must not enter the first episode.
    adv, target = generalized_advantage_estimate(torch.tensor([0., 2., 3.]),
        torch.tensor([1., 1., 2.]), torch.tensor([1., 0., 10.]),
        torch.tensor([False, True, False]), torch.tensor([False, True, True]),
        gamma=0.9, gae_lambda=0.8)
    torch.testing.assert_close(adv, torch.tensor([0.62, 1., 10.]))
    torch.testing.assert_close(target, torch.tensor([1.62, 2., 12.]))


def test_ppo_clipping_positive_and_negative_advantages():
    ratios = torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([2., 2., -2., -2.], dtype=torch.float64)
    loss = clipped_policy_loss(ratios.log(), torch.zeros(4), adv, .2)
    # Independent explicit objective min(r*A, clipped(r)*A).
    assert loss.item() == pytest.approx(-(2.4 + 1. - 3. - 1.6) / 4)
    loss.backward()
    torch.testing.assert_close(ratios.grad, torch.tensor([0., -.5, .5, 0.], dtype=torch.float64))


class SmallEnv:
    def __init__(self):
        self.count = 0

    def reset(self):
        self.count = 0
        return scene()

    def step(self, action):
        self.count += 1
        reward = -float(action.gate.sum()) if self.count == 2 else 0.
        return replace(scene(), state=scene().state + self.count * 0.1), reward, self.count == 2, False, {}


def test_collection_freezes_behavior_and_update_changes_actor_and_critic():
    torch.manual_seed(5)
    actor = Actor(3, 2, 2, 2, 16)
    critic = ValueNetwork(3, 16)
    before_a = [p.detach().clone() for p in actor.parameters()]
    before_v = [p.detach().clone() for p in critic.parameters()]
    transitions = collect_episodes(actor, critic, SmallEnv(), episodes=3, max_steps=2)
    assert len(transitions) == 6
    for previous, current in zip(before_a, actor.parameters()):
        torch.testing.assert_close(previous, current)
    for step in transitions:
        assert not step.old_log_prob.requires_grad and not step.value.requires_grad
        replay = actor.evaluate_actions(step.inputs, step.gate, step.mapping)
        torch.testing.assert_close(replay, step.old_log_prob)
        if step.terminated:
            assert step.next_value == 0
    trainer = PPOTrainer(actor, critic, PPOConfig(epochs=2, minibatch_size=3, entropy_coefficient=.01))
    metrics = trainer.update(transitions)
    assert all(math.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(previous, current) for previous, current in zip(before_a, actor.parameters()))
    assert any(not torch.equal(previous, current) for previous, current in zip(before_v, critic.parameters()))


def test_horizon_is_truncation_and_observation_is_snapshotted():
    actor = Actor(3, 2, 2, 2, 16)
    critic = ValueNetwork(3, 16)
    transitions = collect_episodes(actor, critic, SmallEnv(), episodes=1, max_steps=1)
    assert transitions[0].episode_end and not transitions[0].terminated
    expected = critic(scene().state + .1)
    torch.testing.assert_close(transitions[0].next_value, expected)


def test_entropy_score_function_matches_enumeration_gradient():
    # Exact expectation of the estimator used by PPOTrainer on a small policy.
    logits = torch.tensor([-.4, .8, .2], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(0)
    probabilities = logp.exp()
    negative_entropy = (probabilities * logp).sum()
    expected_grad, = torch.autograd.grad(negative_entropy, logits, retain_graph=True)
    surrogate = (probabilities.detach() * (logp.detach() + 1) * logp).sum()
    surrogate_grad, = torch.autograd.grad(surrogate, logits)
    torch.testing.assert_close(expected_grad, surrogate_grad)
