import pytest
import torch
from torch.nn import functional as F

from sapo_core.attention import SparseHeadAttention


def dense_reference(module, x, mask):
    batch, tokens, dim = x.shape
    qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    scores = ((q @ k.transpose(-2, -1)) * module.scale).softmax(dim=-1)
    values = (scores @ v) * mask.view(1, -1, 1, 1)
    return module.proj(values.transpose(1, 2).reshape(batch, tokens, dim))


@pytest.mark.parametrize("mask", [[1, 1, 1, 1], [1, 0, 1, 0]])
def test_forward_and_gradients_match_dense_reference(mask):
    torch.manual_seed(23)
    sparse = SparseHeadAttention(12, 4, qkv_bias=True).double()
    mask = torch.tensor(mask)
    x = torch.randn(2, 5, 12, dtype=torch.double, requires_grad=True)
    expected = dense_reference(sparse, x, mask)
    actual = sparse(x, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    params = (sparse.qkv.weight, sparse.qkv.bias, sparse.proj.weight, sparse.proj.bias)
    expected_grads = torch.autograd.grad(expected.square().sum(), (x, *params), retain_graph=True)
    actual_grads = torch.autograd.grad(actual.square().sum(), (x, *params))
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-11)

    if mask.tolist() != [1, 1, 1, 1]:
        inactive = torch.tensor([3, 4, 5, 9, 10, 11])
        for offset in (0, 12, 24):
            assert torch.count_nonzero(actual_grads[1].index_select(0, inactive + offset)) == 0
        assert torch.count_nonzero(actual_grads[3].index_select(1, inactive)) == 0


def test_qkv_and_projection_use_only_selected_dimensions(monkeypatch):
    module = SparseHeadAttention(12, 4)
    x = torch.randn(2, 3, 12)
    calls = []
    original = F.linear

    def recording_linear(input, weight, bias=None):
        calls.append((input.shape, weight.shape))
        return original(input, weight, bias)

    monkeypatch.setattr(F, "linear", recording_linear)
    module(x, [0, 1, 0, 1])
    assert calls == [(torch.Size([2, 3, 12]), torch.Size([18, 12])),
                     (torch.Size([2, 3, 6]), torch.Size([12, 6]))]


def test_partition_sum_and_empty_local_result():
    torch.manual_seed(42)
    module = SparseHeadAttention(12, 4, qkv_bias=True).double()
    x = torch.randn(2, 5, 12, dtype=torch.double)
    mask = [1, 0, 1, 1]
    assignment = [0, -1, 2, 0]
    parts = [module.forward_partition(x, mask, assignment, p) for p in range(4)]
    assert torch.count_nonzero(parts[1]) == 0
    assert torch.count_nonzero(parts[3]) == 0
    torch.testing.assert_close(sum(parts) + module.proj.bias, module(x, mask), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mask", [[0, 0, 0, 0], [1, 0], [1, 2, 0, 0]])
def test_rejects_invalid_masks(mask):
    module = SparseHeadAttention(12, 4)
    with pytest.raises(ValueError):
        module(torch.randn(1, 2, 12), mask)


@pytest.mark.parametrize("assignment", [[0, -1, -1, -1], [-1, -1, 0, -1],
                                         [0, 0, 0], [0.0, -1.0, 1.0, -1.0]])
def test_rejects_invalid_assignments(assignment):
    module = SparseHeadAttention(12, 4)
    with pytest.raises(ValueError):
        module.forward_partition(torch.randn(1, 2, 12), [1, 0, 1, 0], assignment, 0)


def test_rejects_invalid_device_index_and_global_empty_mask():
    module = SparseHeadAttention(12, 4)
    x = torch.randn(1, 2, 12)
    with pytest.raises(ValueError):
        module.forward_partition(x, [1, 0, 1, 0], [0, -1, 1, -1], -1)
    with pytest.raises(ValueError):
        module.forward_partition(x, [0, 0, 0, 0], [-1] * 4, 0)
