# Algorithms and API

SAPO combines head gating, graph-based placement, sparse attention execution and policy optimization. This document describes the algorithms and module interfaces.

## Gate and joint policy

For Bernoulli probabilities `p_h`, the gate probability is

```text
p(gate | nonempty) = product_h p_h^gate_h (1-p_h)^(1-gate_h)
                    / (1 - product_h (1-p_h))
```

`NonemptyBernoulli` samples the first retained index using mutually exclusive first-success events, then samples later heads independently. This implements the conditioned distribution directly. The same log normalization appears in sampling and likelihood evaluation.

`Actor(state_dim, node_dim, device_dim, num_heads, hidden_dim=64)` consumes one `Observation`:

| Field | Shape and meaning |
| --- | --- |
| `state` | `[state_dim]`, system/content/layer features |
| `node_features` | `[J, node_dim]`, operator descriptors and profiled costs |
| `device_features` | `[M, device_dim]`, device descriptors |
| `head_id` | `[J]`, head index or `-1` for an always-active operator |
| `edges` | `[E, 2]`, source/destination indices in a DAG |
| `residency` | Boolean `[J, M]`, weight availability |
| `memory_cost` | `[J]`, nonnegative integer memory units |
| `capacity` | `[M]`, nonnegative available memory in the same units |

Every head is represented by at least one node. Floating features match the actor's dtype and device. Memory allocations reserve their stated capacity throughout this decision; they are released when the next independent scene is constructed. Use integer bytes or another shared integer unit for exact capacity accounting.

Feasibility is checked on the **active subgraph**. If every nonempty gate is feasible, the direct nonempty-Bernoulli sampler is used. Otherwise, the joint feasible-action mask is also applied to the gate support before normalization:

```text
p(gate | S, feasible) = BernoulliWeight(gate) * is_nonempty_and_schedulable(gate)
                       / sum_over_feasible_nonempty_gates BernoulliWeight(gate)
```

The resource-constrained branch enumerates the feasible support with monotone pruning. The exact normalization is reused in sampling, action replay and PPO likelihoods. Thus a full model that does not fit can still yield valid pruned actions. A state with no nonempty feasible gate raises an error. During mapping, a candidate device is enabled only if the remaining active nodes can still be placed. These exact feasibility searches are exponential in the worst case; they do not alter an action after it is sampled.

A directed graph-attention layer contextualizes each active node using itself and its predecessors. A GRU decoder visits a stable topological order, using node embeddings, predecessor placements, device embeddings and remaining capacity.

`sample_action(observation, deterministic=False)` returns `Action(gate, mapping, log_prob)`. Device indices are zero-based; inactive nodes have mapping `-1`. `evaluate_actions(observation, gate, mapping)` replays those exact decisions and returns the joint gate-plus-mapping log probability. Invalid actions raise an error instead of being repaired after sampling. Setting `deterministic=True` chooses a maximum-probability feasible gate and a masked argmax at each decoder step; this is a deterministic decoder, not a claim of globally optimal placement.

## Sparse attention and collectives

`SparseHeadAttention` retains compatible `qkv` and `proj` parameter shapes. Before matrix multiplication, it gathers only the retained heads' QKV rows and output-projection columns. Inputs and outputs have shape `[batch, tokens, dim]`; a common nonempty mask of shape `[heads]` applies to the batch.

`forward_partition(x, head_mask, assignment, device_index)` returns the bias-free local contribution. Empty local partitions return zeros. Summing all local contributions and adding the projection bias once equals the selected-head forward pass in deterministic inference mode.

`sum_head_partials(local_output, bias)` performs actual process-group `all_reduce(SUM)` before adding bias. `sequence_parallel_ffn(x, ffn)` assigns contiguous token ranges, evaluates the local FFN and gathers the shards in sequence order. Unequal and empty token ranges are handled by padded gather and trimming.

The caller initializes the process group, provides matching weights/plans/input on all ranks, and invokes collectives in the same order. Collective wrappers are inference operators with `no_grad`; the FFN is deterministic, token-independent and shape-preserving. Projection/attention dropout is disabled for partition-equivalence checks.

## State, schedule and execution

`build_state(utilization, memory_available, bandwidth, token_features, layer_index, num_layers)` constructs device-wise telemetry, mean content embedding and layer encoding. Every outgoing link is represented separately; self-links are omitted. The caller supplies measured snapshots and consistent feature units/scales. The example uses generated snapshots and explicitly scales memory by 8 and bandwidth by 1000.

`build_schedule(compute_times, edges, mapping, alpha, beta, payloads)` constructs the active execution order and timing from a `[operators, devices]` profile table. In stable active-graph topological order:

```text
transfer(u,v) = 0                                  if assigned to the same device
                alpha[device_u,device_v]
                + beta[device_u,device_v]*payload  otherwise
start(v) = max(device_available[device_v],
               max_predecessors(finish(u) + transfer(u,v)))
finish(v) = start(v) + compute_times[v,device_v]
T_total = max_active_nodes finish(v)
```

Operators on the same device do not overlap. Edge transfers are modeled as independent dependency delays. The output includes node starts/finishes and edge communication times, making the execution-graph critical path inspectable. Removed nodes do not influence active-node ordering.

`execute_graph(edges, mapping, kernels, inputs, devices)` calls the pre-positioned kernel registered for **each** active `(operator, assigned_worker)` pair. Source kernels receive external input tensors; other kernels receive active predecessor outputs in the supplied edge order. Activations move to the assigned local device, while model weights remain pre-positioned. The executor returns node outputs and dispatch records. Worker device indices must be explicit for CUDA. It executes in topological order for functional checking; concurrent/distributed execution uses the separate collective operators and caller runtime.

The example dispatches attention-head and aggregation kernels through two logical CPU workers, computes costs from the same active graph, and then updates the policy. Its telemetry, profiles and costs are generated examples, not hardware timings.

## Reward and PPO

The terminal objective combines output divergence and normalized execution cost:

```text
D = sum P_full * (log P_full - log P_dynamic)
C = exp(kappa_time * total_time / baseline_time) - 1
weight = lambda_max * sigmoid(kappa_kl * (kl_target - D))
reward = -D - weight * C
```

The full-model distribution is a detached teacher. Both time arguments use the same units. An environment supplies execution costs and emits zero rewards before its terminal decision. The included demonstration supplies synthetic cost units explicitly.

`collect_episodes` keeps behavior-policy/value parameters fixed and snapshots observations, actions and likelihoods. The environment supplies `reset() -> Observation` and `step(action) -> (next_observation, reward, terminated, truncated, info)`.

GAE bootstraps at truncation, uses zero continuation value at true termination, and never propagates advantages across episode boundaries. PPO replays the joint likelihood of each stored action, normalizes rollout advantages and applies the clipped surrogate. Actor and critic use separate optimizers and gradient clipping.

The entropy term uses fresh samples from the current joint policy. Its gradient estimator follows `grad H = -E[(log p + 1) grad log p]`, including the gate-conditioned mapping distribution. The value target is formed once from the behavior critic before update epochs. PPO clipping is the optimizer mechanism; no hard trust-region constraint is imposed.
