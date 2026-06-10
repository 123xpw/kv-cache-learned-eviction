# Learned KV Cache Eviction Simulation

This repository contains the simulation code for a course paper on learned KV Cache eviction for large language model inference.

The project evaluates a lightweight MLP-based block eviction policy, referred to as `Learned`, on synthetic sparse block-level KV Cache access traces. It compares the policy with LRU, an H2O-style block-level heuristic, and a Belady OPT upper bound under the same protected-anchor-block constraint.

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
| H2O-style | 90.2% |
| LRU | 84.2% |

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

## Citation

If this repository is referenced in the course paper, use the repository URL and commit hash for reproducibility.
