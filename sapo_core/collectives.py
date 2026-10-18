"""Inference-only distributed collectives for SAPO's hybrid parallel layer.

The caller initializes the process group and invokes each function in the same
order on every participating rank. These collectives intentionally do not
provide an autograd path; policy training must not treat them as differentiable
communication operators.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn


def _group_size_and_rank(group: dist.ProcessGroup | None) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed process group must be initialized by the caller")
    return dist.get_world_size(group=group), dist.get_rank(group=group)


@torch.no_grad()
def sum_head_partials(
    local_output: Tensor,
    bias: Tensor | None = None,
    group: dist.ProcessGroup | None = None,
) -> Tensor:
    """Sum bias-free output-projection contributions from head partitions.

    Args:
        local_output: This rank's dense ``(batch, tokens, features)`` partial
            result, such as ``SparseHeadAttention.forward_partition`` output.
            Every rank must pass the same shape, dtype and device type.
        bias: Optional output-projection bias of shape ``(features,)``, equal on
            all ranks. Added once, *after* the sum, including on empty-head
            ranks. Do not include it in ``local_output``.
        group: An already initialized process group, or ``None`` for the
            default group. The caller owns its lifecycle.

    Returns:
        The full output on every rank. ``local_output`` is not modified.
        This function is inference-only and has no autograd support.
    """
    if local_output.ndim != 3:
        raise ValueError("local_output must have shape (batch, tokens, features)")
    if bias is not None and (
        bias.shape != (local_output.shape[-1],)
        or bias.dtype != local_output.dtype
        or bias.device != local_output.device
    ):
        raise ValueError("bias must match the output feature dimension, dtype and device")
    _group_size_and_rank(group)
    output = local_output.clone()
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
    if bias is not None:
        output.add_(bias)
    return output


@torch.no_grad()
def sequence_parallel_ffn(
    x: Tensor,
    ffn: nn.Module,
    group: dist.ProcessGroup | None = None,
) -> Tensor:
    """Apply a token-independent FFN to contiguous sequence shards.

    Args:
        x: Identical full input ``(batch, tokens, features)`` on every rank.
        ffn: Identical, deterministic, token-independent, shape-preserving FFN
            on every rank, in inference mode. The caller owns parameter
            replication and mode. An empty local shard is skipped.
        group: An already initialized process group, or ``None`` for the
            default group. The caller owns its lifecycle.

    Returns:
        The full FFN result on every rank, with original token order restored.
        Remainder tokens go to lower group ranks. Uneven and empty shards are
        supported by padding only for ``all_gather`` and trimming afterward.
        This function is inference-only and has no autograd support.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, tokens, features)")
    world_size, rank = _group_size_and_rank(group)
    tokens = x.shape[1]
    base, remainder = divmod(tokens, world_size)
    lengths = [base + (r < remainder) for r in range(world_size)]
    start = rank * base + min(rank, remainder)
    length = lengths[rank]

    if tokens == 0:
        return x.clone()

    local_input = x.narrow(1, start, length)
    local_output = ffn(local_input) if length else local_input
    if not isinstance(local_output, Tensor) or local_output.shape != local_input.shape:
        raise ValueError("ffn must return a tensor with the local input shape")
    if local_output.dtype != x.dtype or local_output.device != x.device:
        raise ValueError("ffn output must match the input dtype and device")

    padded_length = base + (remainder > 0)
    padded = x.new_zeros((x.shape[0], padded_length, x.shape[2]))
    if length:
        padded[:, :length].copy_(local_output)
    shards = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(shards, padded.contiguous(), group=group)
    return torch.cat([shard[:, :count] for shard, count in zip(shards, lengths)], dim=1)
