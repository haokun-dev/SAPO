"""Structurally sparse, head-gated self-attention for a shared input batch.

The parameter names and shapes match ``sapo.vision_transformer.Attention``.
Each call gathers only active heads' Q/K/V rows and output-projection columns
before either linear operation. Masks are discrete execution decisions; gradients
flow through the selected weights and input, not through the mask or assignment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SparseHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim and num_heads must be positive, and dim divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _active_heads(self, head_mask: Tensor | list[int] | tuple[int, ...] | None, device: torch.device) -> Tensor:
        if head_mask is None:
            return torch.arange(self.num_heads, device=device)
        mask = torch.as_tensor(head_mask, device=device)
        if mask.shape != (self.num_heads,):
            raise ValueError(f"head_mask must have shape ({self.num_heads},)")
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError("head_mask must contain only 0 or 1")
        active = torch.nonzero(mask, as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError("head_mask must retain at least one head")
        return active

    def _project_heads(self, x: Tensor, heads: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        offsets = torch.arange(self.head_dim, device=x.device)
        features = (heads[:, None] * self.head_dim + offsets).reshape(-1)
        qkv_rows = torch.cat((features, features + dim, features + 2 * dim))
        bias = self.qkv.bias.index_select(0, qkv_rows) if self.qkv.bias is not None else None
        qkv = F.linear(x, self.qkv.weight.index_select(0, qkv_rows), bias)
        qkv = qkv.reshape(batch, tokens, 3, heads.numel(), self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attention = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        values = (attention @ v).transpose(1, 2).reshape(batch, tokens, -1)
        return F.linear(values, self.proj.weight.index_select(1, features), bias=None)

    def forward(self, x: Tensor, head_mask: Tensor | list[int] | tuple[int, ...] | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.qkv.in_features:
            raise ValueError(f"x must have shape (batch, tokens, {self.qkv.in_features})")
        heads = self._active_heads(head_mask, x.device)
        output = self._project_heads(x, heads)
        if self.proj.bias is not None:
            output = output + self.proj.bias
        return self.proj_drop(output)

    def forward_partition(
        self,
        x: Tensor,
        head_mask: Tensor | list[int] | tuple[int, ...],
        assignment: Tensor | list[int] | tuple[int, ...],
        device_index: int,
    ) -> Tensor:
        """Return one logical device's bias-free contribution, without communication.

        ``assignment[h]`` is a zero-based device index for retained heads and
        ``-1`` for dropped heads. Sum all partitions and add ``proj.bias`` once
        to recover ``forward`` when projection dropout is disabled. The caller
        supplies actual distributed communication, if needed.
        """
        if x.ndim != 3 or x.shape[-1] != self.qkv.in_features:
            raise ValueError(f"x must have shape (batch, tokens, {self.qkv.in_features})")
        active = self._active_heads(head_mask, x.device)
        devices = torch.as_tensor(assignment, device=x.device)
        if devices.shape != (self.num_heads,) or devices.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError(f"assignment must be an integer vector of shape ({self.num_heads},)")
        if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        retained = torch.zeros(self.num_heads, device=x.device, dtype=torch.bool)
        retained[active] = True
        if bool(torch.any(devices[retained] < 0)) or bool(torch.any(devices[~retained] != -1)):
            raise ValueError("retained heads need nonnegative assignments; dropped heads need -1")
        local = active[devices[active] == device_index]
        if local.numel() == 0:
            return x.new_zeros((*x.shape[:2], x.shape[-1]))
        return self._project_heads(x, local)
