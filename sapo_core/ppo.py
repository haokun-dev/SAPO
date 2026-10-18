"""On-policy rollout collection, boundary-aware GAE and factorized PPO updates."""
from dataclasses import dataclass, fields
import math
from typing import Any

import torch
from torch import Tensor, nn

from .policy import Observation


@dataclass(frozen=True)
class Transition:
    inputs: Observation
    gate: Tensor
    mapping: Tensor
    old_log_prob: Tensor
    value: Tensor
    next_value: Tensor
    reward: float
    terminated: bool
    episode_end: bool


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 32
    max_grad_norm: float = 0.5
    seed: int = 42

    def __post_init__(self):
        numbers = (self.gamma, self.gae_lambda, self.clip_ratio, self.value_coefficient,
                   self.entropy_coefficient, self.learning_rate, self.max_grad_norm)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('PPO parameters must be finite')
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1 and 0 < self.clip_ratio < 1):
            raise ValueError('Invalid discount, GAE lambda or clip ratio')
        if min(self.value_coefficient, self.entropy_coefficient) < 0 or min(self.learning_rate, self.max_grad_norm) <= 0:
            raise ValueError('Invalid loss coefficient, learning rate or gradient bound')
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in (self.epochs, self.minibatch_size)):
            raise ValueError('Epochs and minibatch size must be positive integers')


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state).squeeze(-1)


@torch.no_grad()
def generalized_advantage_estimate(rewards: Tensor, values: Tensor, next_values: Tensor,
                                   terminated: Tensor, episode_ends: Tensor,
                                   gamma=0.99, gae_lambda=0.95):
    """A true terminal has no bootstrap; a truncated episode does bootstrap.

    ``episode_ends`` stops GAE recursion across either kind of boundary.
    ``next_values`` is captured by the frozen behavior critic during collection.
    """
    arrays = (rewards, values, next_values, terminated, episode_ends)
    if rewards.ndim != 1 or rewards.numel() == 0 or any(a.shape != rewards.shape for a in arrays):
        raise ValueError('GAE inputs must be equally sized nonempty vectors')
    if any(a.device != rewards.device for a in arrays):
        raise ValueError('GAE inputs must share a device')
    if terminated.dtype != torch.bool or episode_ends.dtype != torch.bool:
        raise ValueError('Boundary flags must be boolean')
    if (terminated & ~episode_ends).any():
        raise ValueError('Every terminal transition must end its episode')
    if not all(torch.isfinite(a).all() for a in arrays[:3]):
        raise ValueError('GAE inputs must be finite')
    if not (0 <= gamma <= 1 and 0 <= gae_lambda <= 1):
        raise ValueError('Invalid discount or GAE lambda')
    deltas = rewards + gamma * (~terminated).to(values.dtype) * next_values - values
    advantages = torch.empty_like(values)
    carry = values.new_zeros(())
    for index in reversed(range(len(values))):
        carry = deltas[index] + gamma * gae_lambda * (~episode_ends[index]).to(values.dtype) * carry
        advantages[index] = carry
    return advantages, advantages + values


def clipped_policy_loss(new_log_prob: Tensor, old_log_prob: Tensor, advantages: Tensor, clip_ratio=0.2):
    if new_log_prob.shape != old_log_prob.shape or new_log_prob.shape != advantages.shape or new_log_prob.numel() == 0:
        raise ValueError('PPO likelihoods and advantages must have identical nonempty shapes')
    if not 0 < clip_ratio < 1:
        raise ValueError('clip_ratio must be in (0, 1)')
    old_log_prob, advantages = old_log_prob.detach(), advantages.detach()
    ratio = (new_log_prob - old_log_prob).exp()
    if not torch.isfinite(ratio).all() or not torch.isfinite(advantages).all():
        raise FloatingPointError('Non-finite PPO likelihood ratio or advantage')
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()


@torch.no_grad()
def collect_episodes(actor, critic, env: Any, episodes: int, max_steps: int):
    """Collect before calling ``update``; parameters remain fixed in this call.

    env.reset() -> Observation. env.step(action) ->
    (next_inputs, reward, terminated, truncated, info). A collection horizon
    acts as truncation, not a terminal. The environment owns execution and cost.
    """
    if episodes < 1 or max_steps < 1:
        raise ValueError('episodes and max_steps must be positive')
    actor_mode, critic_mode = actor.training, critic.training
    actor.eval()
    critic.eval()
    transitions = []
    try:
        for _ in range(episodes):
            inputs = env.reset()
            for index in range(max_steps):
                saved = Observation(**{field.name: getattr(inputs, field.name).detach().clone()
                                       for field in fields(Observation)})
                action = actor.sample_action(saved)
                value = critic(saved.state).detach().clone()
                next_inputs, reward, terminal, truncated, _ = env.step(action)
                boundary = bool(terminal or truncated or index == max_steps - 1)
                next_value = torch.zeros_like(value) if terminal else critic(next_inputs.state).detach().clone()
                if not math.isfinite(float(reward)):
                    raise ValueError('Environment returned non-finite reward')
                transitions.append(Transition(saved, action.gate.detach().clone(), action.mapping.detach().clone(),
                    action.log_prob.detach().clone(), value, next_value, float(reward), bool(terminal), boundary))
                if boundary:
                    break
                inputs = next_inputs
    finally:
        actor.train(actor_mode)
        critic.train(critic_mode)
    return transitions


class PPOTrainer:
    """Separate actor/critic optimizers; joint action log-probability is replayed.

    Entropy gradients use fresh on-policy samples, not behavior-policy actions:
    grad H = -E[(log p + 1) grad log p]. This score-function estimate accounts
    for gate-conditioned autoregressive mappings without enumerating actions.
    """
    def __init__(self, actor, critic, config: PPOConfig | None = None):
        self.actor, self.critic = actor, critic
        self.config = config or PPOConfig()
        if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
            raise ValueError('Actor and critic must have separate parameters')
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate)
        self.updates = 0

    def update(self, transitions: list[Transition]):
        if len(transitions) < 2:
            raise ValueError('At least two on-policy transitions are required')
        cfg = self.config
        device = transitions[0].value.device
        values = torch.stack([t.value.detach().reshape(()) for t in transitions])
        next_values = torch.stack([t.next_value.detach().reshape(()) for t in transitions])
        rewards = torch.tensor([t.reward for t in transitions], dtype=values.dtype, device=device)
        terminated = torch.tensor([t.terminated for t in transitions], dtype=torch.bool, device=device)
        ends = torch.tensor([t.episode_end for t in transitions], dtype=torch.bool, device=device)
        advantages, targets = generalized_advantage_estimate(rewards, values, next_values, terminated, ends,
                                                            cfg.gamma, cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_lp = torch.stack([t.old_log_prob.detach().reshape(()) for t in transitions])
        generator = torch.Generator().manual_seed(cfg.seed + self.updates)
        metrics = []
        actor_mode, critic_mode = self.actor.training, self.critic.training
        # Keep stochastic layers disabled so replay has the behavior likelihood semantics.
        self.actor.eval()
        self.critic.eval()
        try:
            for _ in range(cfg.epochs):
                order = torch.randperm(len(transitions), generator=generator)
                for start in range(0, len(order), cfg.minibatch_size):
                    indices = order[start:start + cfg.minibatch_size].tolist()
                    batch = [transitions[i] for i in indices]
                    new_lp = torch.stack([self.actor.evaluate_actions(t.inputs, t.gate, t.mapping) for t in batch])
                    policy_loss = clipped_policy_loss(new_lp, old_lp[indices], advantages[indices], cfg.clip_ratio)
                    entropy_estimate = new_lp.new_zeros(())
                    entropy_loss = new_lp.new_zeros(())
                    if cfg.entropy_coefficient:
                        fresh_lp = torch.stack([self.actor.sample_action(t.inputs).log_prob for t in batch])
                        entropy_estimate = -fresh_lp.detach().mean()
                        entropy_loss = ((fresh_lp.detach() + 1) * fresh_lp).mean()
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    (policy_loss + cfg.entropy_coefficient * entropy_loss).backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    predicted_values = torch.stack([self.critic(t.inputs.state).reshape(()) for t in batch])
                    value_loss = torch.nn.functional.mse_loss(predicted_values, targets[indices])
                    if not torch.isfinite(value_loss):
                        raise FloatingPointError('Non-finite critic loss')
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (cfg.value_coefficient * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                    self.critic_optimizer.step()
                    metrics.append((policy_loss.detach().item(), value_loss.detach().item(), entropy_estimate.item()))
            self.updates += 1
        finally:
            self.actor.train(actor_mode)
            self.critic.train(critic_mode)
        return dict(zip(('policy_loss', 'value_loss', 'entropy_estimate'),
                        [sum(row[i] for row in metrics) / len(metrics) for i in range(3)]))
