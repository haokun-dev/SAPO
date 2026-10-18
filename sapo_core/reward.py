"""Terminal output-fidelity and execution-cost reward."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardTerms:
    reward: Tensor
    output_kl: Tensor
    time_cost: Tensor
    latency_weight: Tensor


def terminal_reward(full_logits: Tensor, dynamic_logits: Tensor,
                    total_time, baseline_time, *, kappa_time=1.0,
                    lambda_max=0.2, kl_target=0.08, kappa_kl=8.0) -> RewardTerms:
    """Compute the terminal objective; both times must use the same units.

    Logits have shape [..., classes]; time tensors must broadcast to [...].
    The full-model distribution is a detached teacher. No reward is assigned
    here to intermediate transitions; the caller emits zero before termination.
    """
    if full_logits.shape != dynamic_logits.shape or full_logits.ndim < 1:
        raise ValueError('Full and dynamic logits must have matching [..., classes] shapes')
    if full_logits.shape[-1] < 2 or not full_logits.is_floating_point() or not dynamic_logits.is_floating_point():
        raise ValueError('Expected floating logits for at least two classes')
    if full_logits.device != dynamic_logits.device:
        raise ValueError('Logits must be on the same device')
    if not torch.isfinite(full_logits).all() or not torch.isfinite(dynamic_logits).all():
        raise ValueError('Logits must be finite')
    params = (kappa_time, lambda_max, kl_target, kappa_kl)
    if not all(math.isfinite(x) for x in params) or kappa_time <= 0 or kappa_kl <= 0 or min(lambda_max, kl_target) < 0:
        raise ValueError('Invalid reward parameters')
    log_full = full_logits.detach().log_softmax(-1)
    log_dynamic = dynamic_logits.log_softmax(-1)
    kl = (log_full.exp() * (log_full - log_dynamic)).sum(-1).clamp_min(0)
    time = torch.as_tensor(total_time, dtype=kl.dtype, device=kl.device)
    baseline = torch.as_tensor(baseline_time, dtype=kl.dtype, device=kl.device)
    if not torch.isfinite(time).all() or not torch.isfinite(baseline).all() or (time < 0).any() or (baseline <= 0).any():
        raise ValueError('Times must be finite; total >= 0 and baseline > 0')
    try:
        time, baseline = torch.broadcast_to(time, kl.shape), torch.broadcast_to(baseline, kl.shape)
    except RuntimeError as exc:
        raise ValueError('Time shape must broadcast to the logit batch shape') from exc
    cost = torch.expm1(kappa_time * time / baseline)
    if not torch.isfinite(cost).all():
        raise ValueError('Exponential time cost overflow; check units and kappa_time')
    weight = lambda_max * torch.sigmoid(kappa_kl * (kl_target - kl))
    return RewardTerms(-kl - weight * cost, kl, cost, weight)
