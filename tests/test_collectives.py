"""Real Gloo collective checks against single-process dense computations."""

from __future__ import annotations

from datetime import timedelta
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from sapo_core.attention import SparseHeadAttention
from sapo_core.collectives import sequence_parallel_ffn, sum_head_partials


def _worker(rank: int, world_size: int, init_file: str, results: mp.Queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=Path(init_file).as_uri(),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=25),
        )
        torch.manual_seed(1701)

        # The third rank has no assigned input features in the three-rank case.
        x = torch.randn(2, 5, 2)
        weight = torch.randn(2, 2)
        bias = torch.randn(2)
        features = torch.tensor_split(torch.arange(2), world_size)[rank]
        local = F.linear(x[..., features], weight[:, features], bias=None)
        original = local.clone()
        actual = sum_head_partials(local, bias)
        expected = F.linear(x, weight, bias)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(local, original)
        assert not actual.requires_grad

        ffn = nn.Sequential(nn.Linear(2, 7), nn.GELU(), nn.Linear(7, 2)).eval()
        for tokens in (5, 2, 1, 0):
            full_input = torch.randn(2, tokens, 2)
            full_actual = sequence_parallel_ffn(full_input, ffn)
            full_expected = ffn(full_input)
            torch.testing.assert_close(full_actual, full_expected, atol=1e-6, rtol=1e-6)
            assert full_actual.shape == full_input.shape
            assert not full_actual.requires_grad

        # Exercise the actual head partition API and both collectives as one layer.
        attention = SparseHeadAttention(dim=12, num_heads=3, qkv_bias=True).eval()
        with torch.no_grad():
            attention.proj.bias.fill_(0.375)
        layer_input = torch.randn(2, 5, 12)
        mask = [1, 0, 1]
        assignment = [0, -1, 1]
        partial = attention.forward_partition(layer_input, mask, assignment, rank)
        if rank == 2:
            assert torch.count_nonzero(partial) == 0
        attention_actual = sum_head_partials(partial, attention.proj.bias)
        attention_expected = attention(layer_input, mask)
        torch.testing.assert_close(attention_actual, attention_expected, atol=1e-6, rtol=1e-6)

        layer_ffn = nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 12)).eval()
        layer_actual = sequence_parallel_ffn(attention_actual, layer_ffn)
        layer_expected = layer_ffn(attention_expected)
        torch.testing.assert_close(layer_actual, layer_expected, atol=1e-6, rtol=1e-6)
        assert not layer_actual.requires_grad
        results.put((rank, None))
    except BaseException:
        results.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 3))
def test_gloo_collectives_match_dense(world_size: int) -> None:
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as temporary_dir:
        init_file = os.path.join(temporary_dir, "gloo-init")
        results = context.Queue()
        processes = [
            context.Process(target=_worker, args=(rank, world_size, init_file, results))
            for rank in range(world_size)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=40)
            hanging = [process.pid for process in processes if process.is_alive()]
            assert not hanging, f"Gloo workers timed out: {hanging}"
            failures = []
            for _ in processes:
                try:
                    rank, error = results.get(timeout=2)
                except queue.Empty:
                    failures.append("worker exited without reporting a result")
                    continue
                if error is not None:
                    failures.append(f"rank {rank}:\n{error}")
            failures.extend(
                f"worker pid {process.pid} exited with {process.exitcode}"
                for process in processes
                if process.exitcode != 0
            )
            assert not failures, "\n".join(failures)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
            results.close()
            results.join_thread()
