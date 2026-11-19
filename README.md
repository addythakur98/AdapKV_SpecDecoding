# Adaptive KV-Cache Compression and Speculative Decoding for Energy-Efficient LLM Inference on Low-Resource Hardware

**Major Project Implementation — Delhi Technological University (DTU), Dept. of Computer Science & Engineering**  
- **Draft Model**: `Qwen/Qwen2.5-0.5B`  
- **Target Model**: `Qwen/Qwen2.5-7B`  
- **Team Members (Group A5)**: Rudransh (23/CS/351), Saksham Thakur (23/CS/362)  
- **Supervisor**: Mr. Ravin Ahuja  
- **SDG Alignment**: SDG 9 (*Industry, Innovation and Infrastructure*) & SDG 12 (*Responsible Consumption and Production*)

---

## 1. Overview & Architecture

Autoregressive inference on multi-billion parameter LLMs is bottlenecked by two primary factors on consumer-grade hardware:
1. **Memory-Bound Sequential Decoding**: Generating each token requires reading billions of target model weights from GPU VRAM.
2. **Linearly Growing KV-Cache Footprint**: Storing full-precision Key-Value (KV) states over long contexts exhausts consumer GPU memory and degrades attention memory bandwidth.

This repository implements a **two-pronged inference optimization framework** pairing **`Qwen/Qwen2.5-0.5B`** (draft model) with **`Qwen/Qwen2.5-7B`** (target model):

### (i) Adaptive KV-Cache Compression (`src/kv_cache/`)
A tiered KV-cache manager (`AdaptiveKVCache`, subclassing Hugging Face's `DynamicCache`) that dynamically scores, prunes, and quantizes cached attention states at every step:
- **Token Relevance Scoring**: Combines scaled dot-product query-key alignment with Value $L_2$-norm weighting ($O(N \cdot d)$ proxy scoring compatible with SDPA/FlashAttention) and an exponential moving average decay ($\beta = 0.95$) plus recency bias.
- **Three-Tier Memory Hierarchy**:
  1. **Full-Precision Tier (`FP16`/`BF16`)**: Preserves **Attention Sink tokens** (first $N_{\text{sink}}$ tokens, per StreamingLLM), **Top Heavy-Hitters** (highest cumulative relevance scores, per H2O), and the **Local Recency Window** (most recent $N_{\text{local}}$ tokens).
  2. **Quantized Mid-Tier (`INT4` / `INT8`)**: Medium-importance historical tokens are compressed using per-token asymmetric quantization (`quantizer.py`), packing two 4-bit integers per `uint8` byte ($\sim 4\times$ raw memory reduction over FP16).
  3. **Evicted Tier**: Low-importance tokens beyond `max_cache_size` are dynamically pruned.
- **Speculative Rollback (`crop`)**: Supports instantaneous $O(1)$ tail cropping to discard rejected draft token KV states before committing cache compression.

### (ii) Speculative Decoding Pipeline (`src/speculative/`)
- **Draft-then-Verify**: `Qwen/Qwen2.5-0.5B` autoregressively proposes $\gamma$ future tokens, which `Qwen/Qwen2.5-7B` verifies in a **single parallel forward pass**.
- **Lossless Greedy & Rejection Sampling**: Implements Leviathan et al. (2023) rejection sampling:
  $$\text{Accept } d_i \text{ with probability } \min\left(1, \frac{p_{\text{target}}(d_i)}{q_{\text{draft}}(d_i)}\right)$$
  with residual distribution resampling $\text{norm}(\max(0, p_i - q_i))$ upon rejection and a bonus token from position $\gamma + 1$ when all $\gamma$ tokens are accepted.
- **Dynamic Confidence-Aware Lookahead (`adaptive_gamma`)**: Halts draft speculation early when draft confidence falls below $\tau_{\text{conf}}$, avoiding wasted draft passes.
- **Vocabulary Alignment**: Automatically handles the padded embedding size difference between `Qwen2.5-0.5B` (`151936`) and `Qwen2.5-7B` (`152064`).

---

## 2. Project Structure

```text
AdaptiveKVCache/
├── config.py                        # Dataclass configurations (Models, AdaptiveKV, Speculative, Benchmark)
├── main.py                          # CLI entrypoint (benchmark, single generation, interactive chat)
├── requirements.txt                 # Project dependencies
├── src/
│   ├── models/
│   │   └── loader.py                # Loads Qwen/Qwen2.5-0.5B & Qwen/Qwen2.5-7B (FP16/BF16/4-bit/8-bit)
│   ├── kv_cache/
│   │   ├── quantizer.py             # Packed INT4 & INT8 per-token asymmetric KV quantizer
│   │   └── adaptive_cache.py        # AdaptiveKVCache (Sinks + Heavy Hitters + INT4 Tier + Pruning + Crop)
│   ├── speculative/
│   │   ├── baseline_decoder.py      # Standard autoregressive decoding with telemetry
│   │   └── speculative_decoder.py   # Speculative decoding with KV rollback & adaptive gamma
│   └── benchmark/
│       ├── energy_monitor.py        # NVML GPU power sampling thread (Joules/token & gCO2e/1k tokens)
│       ├── metrics.py               # Streaming Perplexity (PPL) & N-gram F1 quality metrics
│       ├── runner.py                # 4-way ablation benchmark runner (JSON + CSV export)
│       └── plots.py                 # 2x2 publication-ready Matplotlib comparison dashboard
└── tests/
    └── test_framework.py            # Unit & integration test suite
```

---

## 3. Installation

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 4. Usage

### A. Full 4-Way Ablation Benchmark (`Qwen/Qwen2.5-0.5B` + `Qwen/Qwen2.5-7B`)
Evaluates all four configurations across **Throughput (tok/s)**, **Latency (TTFT & TPOT)**, **KV-Cache Memory & Compression Ratio**, **Perplexity (PPL)**, and **Energy-per-Token (Joules/tok)**:
1. **Baseline**: `Qwen/Qwen2.5-7B` with standard uncompressed KV cache
2. **Adaptive KV-Cache Only**: `Qwen/Qwen2.5-7B` with dynamic pruning + INT4 quantization
3. **Speculative Decoding Only**: `Qwen/Qwen2.5-0.5B` $\rightarrow$ `Qwen/Qwen2.5-7B` with standard KV cache
4. **Proposed Combined Framework**: `Qwen/Qwen2.5-0.5B` $\rightarrow$ `Qwen/Qwen2.5-7B` + Adaptive KV-Cache Compression

```bash
# On GPUs with >= 16GB VRAM (FP16):
python main.py --mode benchmark --draft-model Qwen/Qwen2.5-0.5B --target-model Qwen/Qwen2.5-7B

# FP16 target by default (add --target-quant bnb4 only for GPUs with < 16 GB VRAM):
python main.py --mode benchmark --max-new-tokens 128 --gamma 5 --max-cache-size 256

# Context-Length Scaling & Crossover Sweep (1K -> 4K -> 8K -> 16K -> 32K, 25% proportional KV budget):
python main.py --context-sweep --sweep-lengths 1024,4096,8192,16384,32768 --cache-budget-ratio 0.25
```

### A2. Real-Task Benchmarks: LongBench + RULER (`--mode tasks`)
The context sweep uses synthetic, templated prompts, which are good for exact-length speed and memory curves but too predictable to judge quality or speculative acceptance. `--mode tasks` runs all 4 methods on standard long-context benchmarks. Every method gets byte-identical prompts, and every generation records **its task score and its own decode speed, acceptance rate and KV memory**.

| Suite | What it measures | Fidelity to the official benchmark |
|---|---|---|
| **LongBench** (THUDM, ACL'24) | Real single-doc QA, multi-doc QA, summarization, few-shot, synthetic retrieval and code completion | Official data (`THUDM/LongBench` data.zip), prompts, `max_new_tokens`, metrics (QA-F1, ROUGE-L, classification, retrieval, count, code edit-sim), middle truncation, and the no-chat-template rule for few-shot/code tasks |
| **LongBench-E** | Same tasks, length-balanced, scored in 0-4k / 4-8k / 8k+ buckets | Official `_e` splits and buckets, with the subset stratified by bucket |
| **RULER** (NVIDIA, COLM'24) | NIAH (single, multi-key, multi-value, multi-query), variable tracking, common/frequent word extraction at exact 4K-32K lengths | Official templates, task configs and `string_match_all` metric, re-implemented. Essay haystack = PG-19 (official: Paul Graham essays). QA tasks are omitted because LongBench covers real QA |

```bash
# Instruct pair (task prompts need instruction following; target and draft are never mixed)
python main.py --mode tasks --target-model Qwen/Qwen2.5-7B-Instruct --draft-model Qwen/Qwen2.5-0.5B-Instruct \
    --suite longbench,ruler --ruler-lengths 4096,8192,16384 --task-samples 20 --gamma 5

# Length-bucketed LongBench-E, every task
python main.py --mode tasks --suite longbench_e --longbench-tasks all --task-samples 30 ...

# Offline CPU check with tiny mock models (RULER is generated locally)
python main.py --mode tasks --mock --suite ruler --ruler-lengths 1024,2048 --task-samples 2
```

Outputs in `results/tasks/`:
- `task_predictions.jsonl`: one row per (sample, method), written immediately. Re-running **resumes** from it; rows from a different config are ignored.
- `task_summary.md`, `task_summary.csv` and `task_summary.json`: per-task scores (mean, std, 95% bootstrap CI), quality retained by AdaptKV, share of samples actually compressed, Spec-only vs Baseline token-exact losslessness, Proposed vs AdaptKV agreement, and per-suite decode speedup, acceptance, tokens/step and KV MB.
- `tasks_scores.png`, `tasks_ruler_length.png` and `tasks_efficiency.png`.

Notes for interpreting results:
- KV compression is **query-agnostic**: the prompt is compressed during chunked prefill, before the question at the end of the prompt is seen. This is a harder setting than SnapKV-style selection after prefill.
- Samples shorter than `--compress-min-len` (default 4096) are not compressed. The "Compressed" column shows how many were, and `score_on_compressed` isolates them.
- Use `--cache-budget-ratio` (default 0.25 for this mode) to trace the quality/memory curve, for example 0.1, 0.25 and 0.5.

### B. Google Colab Execution
Open [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb) on Google Colab (T4 GPU), upload `AdaptiveKVCache.zip`, and run the cells to execute both the 4-way ablation benchmark and the 1K–32K context-length sweep with real NVML hardware power monitoring.

### C. Single Prompt Generation or Interactive Mode
```bash
# Generate response with telemetry report:
python main.py --mode generate --prompt "Explain how speculative decoding accelerates LLM inference:"

# Interactive CLI session:
python main.py --mode interactive
```

### D. Fast Offline Verification & Unit Tests (No 15GB Model Download Required)
To test the entire pipeline and generate sample plots (`results/benchmark_comparison.png`, `results/context_sweep.png`, JSON/CSV exports) using miniature in-memory `Qwen2ForCausalLM` architectures:

```bash
python main.py --mode benchmark --mock
python main.py --context-sweep --mock --sweep-lengths 256,512,1024 --cache-budget-ratio 0.25
pytest tests/test_framework.py -v
```

## Draft cost vs context, shared keep-set, compress threshold

- **Draft decode attention (q_len <= 8)** uses split-KV batched matmuls instead of SDPA. On the draft's StaticCache
  HF always passes a full-capacity mask at q_len=1, which forces SDPA's memory-efficient kernel; for the 0.5B draft
  (2 KV heads) that kernel runs ~2 thread blocks per layer and streams the whole buffer serially, so draft time grew
  with context. Split-KV spreads the same reads over B x heads x splits blocks (exact, CUDA-graph friendly).
- Draft StaticCache capacity buckets: powers of two up to 4096, then multiples of 2048 (16K context -> 18432 slots,
  previously 32768).
- `--draft-cache {shared_keepset,full}` (default `shared_keepset`): in Proposed the draft keeps the target's
  consensus keep-set (sinks + recent window + top heavy hitters by mean target score), compacted in place inside its
  StaticCache, so CUDA graphs stay valid and the draft reads ~25% of the context. RoPE uses true logical positions.
- `--compress-min-len N` (default 4096): if prompt + max_new_tokens (PPL: prefix + continuation) < N, nothing is
  compressed: AdaptKV runs as Baseline, Proposed runs as Spec-only.
- Each spec row in a sweep prints a `[DraftStep]` line: one profiled draft step (time, logical/physical length,
  buffer capacity, attention path, mask shape, whether KV buffers are static addresses, copy ops, top kernels).

