# Mini Inference Engine

[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![Model](https://img.shields.io/badge/Model-GPT--2-purple)]()
[![Status](https://img.shields.io/badge/status-in%20development-yellow)]()

A from-scratch LLM inference engine, built to understand how serving systems actually work: the transformer forward pass written as plain functions, a KV cache to avoid redundant computation, and continuous batching to keep throughput high under realistic request traffic. Every stage is verified token-for-token against Hugging Face.

## Overview

Most of the cost of running a language model is not the model itself but how requests are scheduled around it. This project starts at the bottom, with a hand-written GPT-2 forward pass (`token ids → logits`, no `nn.Module`), and builds upward toward the techniques used in production serving stacks.

The goals are:

- **Correctness first.** Outputs are checked against the Hugging Face reference implementation at every stage before anything is optimized.
- **Measured performance.** Each optimization is justified with numbers: tokens/sec, p95 time-to-first-token, and memory utilization.
- **Scheduling under load.** Static batching and continuous batching are compared under simulated random request arrivals.

## Current Status

| Phase | Status |
|---|---|
| GPT-2 forward pass (plain functions: `layer_norm`, `attention`, `mlp`, `block`, `forward`) | ✅ Complete |
| Verification harness against Hugging Face (`compare_hf.py`) | ✅ Complete |
| Autoregressive generation loop | 🔄 In progress |
| KV cache | ⏳ Planned |
| Static batching baseline | ⏳ Planned |
| Continuous batching | ⏳ Planned |
| Benchmarks and graphs (tokens/sec, p95 TTFT, memory) | ⏳ Planned |
| Paged KV cache (stretch goal) | ⏳ Planned |

## Correctness

The forward pass is validated against Hugging Face GPT-2 on a set of test prompts:

| Check | Result |
|---|---|
| Argmax of logits at every position | ✅ Matches HF on all 4 prompts |
| Max absolute logit difference | ~1.1e-4 to 1.3e-4 (target: under 1e-4) |

The argmax matches everywhere, but the logit difference currently sits slightly above the target tolerance. If it needs tightening, the plan is to bisect layer by layer (embeddings, block 0, block 1, ...) to find where the drift enters.

## Architecture

The engine is organized as small, composable functions rather than classes:

```
token ids
   │
   ▼
token + position embeddings
   │
   ▼
┌───────────────────────────────┐
│ transformer block  (× N)      │
│   layer_norm → attention      │   causal mask, multi-head
│   + residual                  │
│   layer_norm → mlp            │
│   + residual                  │
└───────────────────────────────┘
   │
   ▼
final layer_norm → logits
   │
   ▼
sampling loop (append token, repeat)
```

Planned layers on top of this core:

- **KV cache:** store each layer's keys and values so every new token only computes attention for itself instead of re-processing the whole prompt.
- **Batching scheduler:** a request queue and scheduler that compares static batching (wait for the whole batch to finish) with continuous batching (swap finished requests out and new ones in at every step).
- **Paged KV cache (stretch):** allocate cache memory in fixed-size blocks to reduce fragmentation and improve memory utilization.

## Tech Stack

**Implemented**
- Python
- GPT-2 (small open model, CPU is sufficient)
- Hugging Face Transformers as the reference for verification

**Planned**
- Request simulator with random arrival times
- Benchmark harness and plots for tokens/sec, p95 time-to-first-token, and memory utilization

## Roadmap

1. ✅ Implement the GPT-2 forward pass as plain functions
2. ✅ Verify logits and argmax against Hugging Face
3. 🔄 Build the autoregressive generation loop
4. ⏳ Add a KV cache and verify outputs are identical to the uncached version
5. ⏳ Build a static batching baseline
6. ⏳ Implement continuous batching and compare under simulated arrivals
7. ⏳ Benchmark and publish graphs (tokens/sec, p95 TTFT, memory utilization)
8. ⏳ Stretch: paged KV cache

> Full usage instructions, including generation and benchmark commands, coming soon as the KV cache and batching stages land.