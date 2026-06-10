# Learned KV Cache Eviction Simulation

This repository contains the simulation code for a course paper on learned KV Cache eviction for large language model inference.

The project evaluates a lightweight MLP-based block eviction policy, referred to as `Learned`, on synthetic sparse block-level KV Cache access traces. It compares the policy with LRU, an H2O-style block-level heuristic, an attention-only control baseline, and a Belady OPT upper bound under the same protected-anchor-block constraint.

## Scope

This is a trace-driven simulator, not an end-to-end LLM serving system.

- The access traces are synthetic.
- The simulation measures block-level cache hit rate.
- The results do not directly prove end-to-end latency, throughput, or generation quality improvements in real LLM serving.

## Files

- `kv_cache_sim.py`: trace generator, MLP training, cache-policy simulation, ablation, and plotting.
- `figures/kv_cache_sim_results.png`: main 40% cache-budget comparison.
- `figures/kv_cache_budget_results.png`: multi-budget comparison.
- `figures/kv_cache_cost_results.png`: simplified system-cost estimate.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulator:

```bash
python kv_cache_sim.py
```

The script prints the main metrics and regenerates:

- `kv_cache_sim_results.png`
- `kv_cache_budget_results.png`
- `kv_cache_cost_results.png`

Expected main result under the default configuration:

| Policy | Hit rate |
| --- | ---: |
| OPT (constrained upper bound) | 92.7% |
| Learned | 90.8% |
| Attn-only control | 90.8% |
| H2O-style | 90.2% |
| LRU | 84.2% |

The Attn-only control evicts the cached block with the lowest cumulative attention score without Recent Window protection. In the default stationary synthetic trace, it matches `Learned`, so the results should not be interpreted as evidence that the MLP policy is stronger than simple cumulative-attention sorting.

The OPT value above is a constrained Belady upper bound computed under the same protected-anchor-block rule used by all other policies. A true unconstrained Belady upper bound is not lower than this value.

## Ablation Summary

The simulator also runs 12 feature configurations: six leave-one-out settings and six single-feature settings. Under the default 40% cache budget, the main ablation results are:

| Feature configuration | Hit rate | Delta vs. full features |
| --- | ---: | ---: |
| Full 6 features | 90.8% | baseline |
| Without block position | 90.8% | ±0.0 pp |
| Without cumulative attention | 90.8% | ±0.0 pp |
| Without log access count | 90.7% | -0.1 pp |
| Without access frequency | 90.6% | -0.2 pp |
| Without recency | 90.7% | -0.1 pp |
| Without context occupancy | 90.8% | ±0.0 pp |
| Block position only | 43.4% | -47.4 pp |
| Cumulative attention only | 90.8% | ±0.0 pp |
| Log access count only | 90.7% | -0.1 pp |
| Access frequency only | 90.7% | -0.1 pp |
| Recency only | 84.1% | -6.7 pp |

These results show that cumulative attention, log access count, and access frequency are redundant strong proxies for the fixed synthetic block-importance distribution. Block position alone performs poorly. Although the training and test traces share the same importance distribution, the Zipf importance values are randomly assigned to block IDs, so the mapping from scalar position to importance is highly irregular. A small single-input MLP tends to learn a smooth function over position and cannot reliably represent this jagged mapping.

## Multi-Seed Summary

Across 5 independent seeds at a 40% cache budget:

| Policy | Mean hit rate | Std. dev. |
| --- | ---: | ---: |
| OPT (constrained upper bound) | 90.9% | ±2.1% |
| Learned | 88.1% | ±3.0% |
| Attn-only control | 88.0% | ±3.0% |
| H2O-style | 87.3% | ±3.2% |
| LRU | 80.7% | ±4.6% |

This supports the same conservative interpretation as the default run: `Learned` is competitive with attention-based heuristics on the stationary synthetic traces, but it does not establish a clear advantage over simple cumulative-attention sorting.

The script also includes a simplified system-cost model. This is an estimate, not a hardware measurement. It uses a conservative synchronous-transfer model for swap-in and swap-out cost, and does not model asynchronous prefetch, overlap, or runtime scheduling. The default model assumes:

- 2 MB per KV block.
- 16 GB/s effective transfer bandwidth.
- 10 us fixed latency per block transfer.
- 0.05 us hit cost.
- 0.10 us compulsory miss cost.
- 0.02 us MLP prediction cost per candidate block for `Learned`.

Under this model, the default run estimates:

| Policy | Estimated total cost | Average access cost | Cost reduction vs. LRU |
| --- | ---: | ---: | ---: |
| OPT (constrained upper bound) | 939.4 ms | 19.28 us | 54.6% |
| Attn-only control | 1196.7 ms | 24.56 us | 42.1% |
| Learned | 1200.4 ms | 24.63 us | 41.9% |
| H2O-style | 1270.4 ms | 26.07 us | 38.5% |
| LRU | 2067.3 ms | 42.43 us | 0.0% |

## Method Summary

The simulator creates synthetic sparse block-level traces with:

- 100 KV blocks.
- Zipf(1.5) block importance distribution.
- An explicit attention-sink block.
- Recency-biased access weights for recent active blocks.
- About 25% active blocks accessed at each step.

`Learned` extracts six features for each block:

- block position
- cumulative attention score
- log access count
- access frequency
- recency
- context occupancy

A small 6-32-1 MLP predicts reuse distance. The cache evicts blocks with larger predicted reuse distance.

The current experiments cover stationary synthetic traces. Non-stationary scenarios such as dormant-then-reactivated blocks, distribution drift, and layer/head-specific attention behavior are not evaluated in this repository.

## Citation

If this repository is referenced in the course paper, use the repository URL and commit hash for reproducibility.
