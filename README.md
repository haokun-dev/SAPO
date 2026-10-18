# SAPO Core

Core algorithms and operators for **Structure-Aware Parallel Orchestration**: nonempty attention-head gating, graph-based device assignment, sparse attention execution, critical-path scheduling and policy optimization.

## Install and run

Python 3.10+ and PyTorch 2.3+.

```bash
git clone https://github.com/haokun-dev/SAPO.git
cd SAPO
python -m pip install -e '.[test]'
python -m pytest -q
python -m sapo_core.demo --iterations 2 --seed 42
```

The example uses generated activations and synthetic cost units. It runs gating, feasible placement, sparse attention, critical-path costs, terminal reward evaluation and PPO updates without downloading data or weights.

## Modules

| Module | Function |
| --- | --- |
| `distributions.py` | Bernoulli head selection conditioned on a nonempty mask |
| `policy.py` | Graph attention encoder and autoregressive device assignment |
| `attention.py` | Selected-head QKV computation and partitioned output projection |
| `scheduling.py` | Critical-path timing with device serialization and pairwise link costs |
| `execution.py` | DAG execution through assigned pre-positioned worker kernels |
| `state.py` | Utilization, memory, outgoing-link and content state construction |
| `collectives.py` | Head-partial all-reduce and sequence-parallel FFN all-gather |
| `reward.py` | Output-fidelity and normalized latency terminal objective |
| `ppo.py` | Rollouts, generalized advantage estimation and clipped PPO updates |

```python
import torch
from sapo_core import SparseHeadAttention

attention = SparseHeadAttention(dim=48, num_heads=4).eval()
x = torch.randn(2, 16, 48)
y = attention(x, head_mask=[1, 0, 1, 0])
assert y.shape == x.shape
```

[Algorithm and API notes](docs/ALGORITHMS.md) · [Test record](docs/VALIDATION.md) · [Source-use terms](LICENSE)

## Source availability

The source is publicly available. No open-source license is granted; see [LICENSE](LICENSE).
