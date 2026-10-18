"""SAPO core policy, sparse attention, reward and optimization operators."""
from .attention import SparseHeadAttention
from .distributions import NonemptyBernoulli
from .policy import Actor, Action, Observation
from .ppo import PPOConfig, PPOTrainer, ValueNetwork, collect_episodes, generalized_advantage_estimate
from .reward import RewardTerms, terminal_reward
from .scheduling import ExecutionSchedule, build_schedule
from .execution import ExecutionRecord, execute_graph
from .state import build_state, device_state_features

__all__ = ['SparseHeadAttention', 'NonemptyBernoulli', 'Actor', 'Action', 'Observation',
           'PPOConfig', 'PPOTrainer', 'ValueNetwork', 'collect_episodes',
           'generalized_advantage_estimate', 'RewardTerms', 'terminal_reward',
           'ExecutionSchedule', 'build_schedule', 'ExecutionRecord', 'execute_graph',
           'build_state', 'device_state_features']
