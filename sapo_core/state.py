"""Construct SAPO state features from a caller-supplied telemetry snapshot."""
import torch
from torch import Tensor


def device_state_features(utilization: Tensor, memory_available: Tensor, bandwidth: Tensor) -> Tensor:
    """Rows are [utilization_i, memory_i, bandwidth_i_to_each_other_device].

    Device order is stable; self-links are omitted. Memory and bandwidth units
    are chosen consistently by the caller. These functions do not collect OS
    telemetry or silently substitute generated values.
    """
    if utilization.ndim != 1 or utilization.numel() == 0:
        raise ValueError('utilization must be a nonempty device vector')
    devices = len(utilization)
    if memory_available.shape != (devices,) or bandwidth.shape != (devices, devices):
        raise ValueError('Expected memory [M] and bandwidth [M, M]')
    if not utilization.is_floating_point():
        raise ValueError('utilization must have floating dtype')
    if memory_available.device != utilization.device or bandwidth.device != utilization.device:
        raise ValueError('Telemetry tensors must share a device')
    if any(not torch.isfinite(value).all() for value in (utilization, memory_available, bandwidth)):
        raise ValueError('Telemetry must be finite')
    if ((utilization < 0) | (utilization > 1)).any() or (memory_available < 0).any() or (bandwidth < 0).any():
        raise ValueError('Invalid utilization, memory, or bandwidth')
    peers = ~torch.eye(devices, dtype=torch.bool, device=utilization.device)
    outgoing = bandwidth.to(utilization.dtype)[peers].reshape(devices, devices - 1)
    return torch.cat((utilization[:, None], memory_available.to(utilization.dtype)[:, None], outgoing), dim=1)


def build_state(utilization: Tensor, memory_available: Tensor, bandwidth: Tensor,
                token_features: Tensor, layer_index: int, num_layers: int) -> Tensor:
    """Concatenate telemetry, sequence-mean content and a one-hot layer index.

    token_features is [tokens, embedding_dim] for one observation. At a terminal
    boundary layer_index == num_layers, the layer encoding is all zeros.
    """
    if not isinstance(num_layers, int) or num_layers < 1 or not isinstance(layer_index, int) or not 0 <= layer_index <= num_layers:
        raise ValueError('Invalid layer index or layer count')
    if token_features.ndim != 2 or min(token_features.shape) < 1 or not token_features.is_floating_point():
        raise ValueError('token_features must be nonempty floating [tokens, embedding_dim]')
    if token_features.device != utilization.device or token_features.dtype != utilization.dtype or not torch.isfinite(token_features).all():
        raise ValueError('Content and telemetry must have matching dtype/device and finite values')
    device_features = device_state_features(utilization, memory_available, bandwidth)
    layer = token_features.new_zeros(num_layers)
    if layer_index < num_layers:
        layer[layer_index] = 1
    return torch.cat((device_features.flatten(), token_features.mean(0), layer))
