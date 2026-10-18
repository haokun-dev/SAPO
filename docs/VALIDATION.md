# Validation

```bash
python -m pytest -q
python -m sapo_core.demo --iterations 2 --seed 42
```

The test suite checks:

- Gate probabilities, feasible support, entropy and joint action likelihoods.
- Residency and memory constraints, deterministic decoding and action replay.
- Sparse attention outputs and gradients against a dense computation.
- DAG dependency timing, device serialization and pairwise communication costs.
- Operator dispatch, active-graph ordering and telemetry-state construction.
- Reward formulas, GAE boundaries, PPO clipping and parameter updates.
- Gloo execution across two and three CPU processes, including empty and uneven partitions.

The example exercises policy learning with generated activations and synthetic cost units. GitHub Actions checks package installation, the test suite and the example on Linux.
