# Learned KV Cache Eviction Simulation

This repository contains the simulation code for a course paper on learned KV Cache eviction for large language model inference.

The project evaluates a lightweight MLP-based block eviction policy, referred to as `Learned`, on synthetic sparse block-level KV Cache access traces. It compares the policy with LRU, an H2O-style block-level heuristic, and Belady OPT.

## Scope

This is a trace-driven simulator, not an end-to-end LLM serving system.

- The access traces are synthetic.
- The simulation measures block-level cache hit rate.
- The results do not directly prove end-to-end latency, throughput, or generation quality improvements in real LLM serving.

## Files

- `kv_cache_sim.py`: trace generator, MLP training, cache-policy simulation, ablation, and plotting.
- `figures/kv_cache_sim_results.png`: main 40% cache-budget comparison.
- `figures/kv_cache_budget_results.png`: multi-budget comparison.
- `paper/论文_KV_Cache管理策略.md`: paper draft corresponding to the simulation.

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

Expected main result under the default configuration:

| Policy | Hit rate |
| --- | ---: |
| OPT | 92.7% |
| Learned | 90.8% |
| H2O-style | 90.2% |
| LRU | 84.2% |

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
