# HANDOFF: Speculative Decoding + Adaptive KV Cache — Make the Benchmarks Right

You are taking over a research benchmark codebase. This brief was written by another agent who
diagnosed the project over many benchmark runs with the user. Read it fully before touching code.
It tells you what is broken, what was already proven, what to build, and how to work with the user.

---

## 0. Your situation (read first)

- **You have no GPU.** Your machine is CPU-only. The user runs everything on **Google Colab**
  (L4 24 GB, A100, sometimes G4 = RTX PRO 6000 96 GB). The user uploads `AdaptiveKVCache.zip`,
  runs `notebooks/colab_run.ipynb`, and pastes the output back.
- **Never fabricate GPU numbers.** You can only verify on CPU (`--mock`, small configs, parity
  tests). Everything else must be measured by the user on Colab.
- **The user installs flash-attn themselves** in a notebook cell
  (`!pip install -q flash-attn --no-build-isolation`). Do not add install logic. Detect it: if
  `import flash_attn` works, use it; otherwise fall back to SDPA and print a loud warning.
- **Keep the notebook's existing cells working unchanged.** The user's workflow is:
  upload zip → unzip → run. Do not add manual steps.
- Colab GPU time is expensive. One notebook run must produce everything (all configs, all
  context lengths), save results after each config, and never crash on OOM (skip + log).

---

## 1. Goal

Show that the **Proposed method (Speculative Decoding + Adaptive KV)** is better than Baseline,
Adaptive-KV-only and Spec-only, measured honestly, on **A100 and L4** (B=1 decode throughput),
while Adaptive KV gives a real **memory** win that enables larger batches.

### Success criteria
1. **Proposed decode throughput ≥ Baseline at every context length** (1k → 16k, 32k where it
   fits), B=1, on A100 and L4. Target ≥1.5× at short context.
2. **Proposed ≥ Spec-only at every context length.**
3. **Adaptive KV quality:** PPL within ~5–10% of baseline at budget 0.25 for contexts ≥ 8k,
   averaged over multiple documents. Acceptance rate of Proposed within a few points of Spec-only.
4. **Memory:** KV memory ≈ 4× smaller at budget 0.25 (draft cache included), and the batch-size
   sweep shows more sequences fit / higher tok/s per GPU than Baseline.

### Honesty rules (non-negotiable)
- Do not special-case benchmark inputs, tune on the eval documents, or change what a metric means
  to make a number look better.
- PPL must be computed **through the method's own cache** (see §6). If compression does not change
  PPL at all, that is a bug, not a result.
- If a criterion cannot be met, say so and explain why with evidence. A correct "it doesn't win at
  32k on L4 because X" is worth more than a fake win.

---

## 2. Setup

- Target: `Qwen/Qwen2.5-7B` (base). Draft: `Qwen/Qwen2.5-0.5B` (base). Same tokenizer.
  Never mix base and instruct between target and draft.
- Qwen2.5-7B: 28 layers, 28 Q heads, 4 KV heads (GQA ×7), head_dim 128 → ~56 KB KV/token (FP16).
  Qwen2.5-0.5B: 24 layers, 14 Q heads, 2 KV heads, head_dim 64 → ~12 KB KV/token.
  Draft `max_position_embeddings` = 32,768.
- Relevant files (from the previous audit): `model_loader/loader.py`, `config.py`, `main.py`,
  `src/speculative/speculative_decoder.py`, `src/kv_cache/adaptive_cache.py`,
  `notebooks/colab_run.ipynb`.

---

## 3. History: what went wrong and what was learned

Each item was verified from the user's logs. Do not re-learn these the hard way.

### Run 1 (T4) — metrics and AdaptKV were broken
- "tok/s" was dominated by prefill (generation too short). Fixed: report TTFT and TPOT separately.
- AdaptKV PPL was 352 → 15,079 (near random). Cause: position/RoPE handling after eviction.
  Fixed: PPL now within tens of percent.

### Run 2 (T4) — draft too expensive, verify too expensive
- Draft step cost > verify cost (c ≈ 0.7). Fixed with static cache etc.
- Verify (q_len = γ+1) costs 2–4× a normal decode step and grows steeply with context, while
  single-token decode stays nearly flat.

### Root causes of the verify slowdown (audit + profiling, confirmed)
1. **q_len > 1 forces an explicit attention mask.** PyTorch's `is_causal=True` aligns top-left,
   but verify queries sit at the *end* of the cache (bottom-right causal). So HF builds a
   (1,1,q_len,kv_len) mask.
2. **With a mask present, HF skips `enable_gqa=True` and calls `repeat_kv`**, copying K and V from
   4 → 28 heads on every layer (~3.3 GB traffic at 8k, ~6.6 GB at 16k). At q_len = 1 there is no
   mask, `enable_gqa=True` is used, and there are zero copies. Micro-benchmark signature: attention
   time jumps q=1→q=2, then is flat for q=2,4,6.
3. **Mask last-dim stride not a multiple of 8** (kv_len = 2054/4102/8198/16390, all ≡ 6 mod 8), so
   the efficient SDPA kernel is often rejected → math fallback (bmm + fp32 softmax + bmm).
4. **bitsandbytes NF4:** for q_len > 1 it dequantizes weights to FP16 instead of the fused
   single-token kernel → constant ~2× verify penalty.
5. **T4 (sm75) has no FlashAttention**, which would have handled bottom-right causal natively.

### Timing bug (confirmed)
- No `torch.cuda.synchronize()` before reading the draft timer / starting the verify timer. Draft
  kernels are async, so their GPU time is billed to **verify**. Result: `d` looks tiny (launch time
  only) and `v` looks huge. TPOT/throughput totals are still correct; only the d/v split is wrong.
- Evidence: at 32k (different draft cache path, which syncs) the split flips (d large, v small).

### Draft-cache inconsistencies (confirmed)
- In Proposed, the **draft KV cache is not compressed** (only the target's is). That's why memory
  is 2.6× not 4× smaller. Also draft and target see different contexts, which likely hurts
  acceptance (acceptance dipped to 46–55% at 8k).
- At **32k**, prompt+gen (33,045) > draft `max_position_embeddings` (32,768), so the code silently
  switches the draft to a 25% AdaptiveKVCache. This produced fake 32k "improvements" in memory and
  throughput. 32k results so far are not comparable.
- The draft StaticCache pre-sized to full length forces a dense unaligned mask even at q_len = 1
  (`create_causal_mask` disables the is_causal skip for compileable caches), so the draft also hits
  repeat_kv + math path at long context.

### L4 run (bnb 4-bit, SDPA)
- **Baseline TPOT is flat ~47 ms from 1k to 32k, and B=4 has the same step time as B=1.** Decode is
  **CPU/launch-overhead-bound**, not memory-bound. The GPU idles waiting for Python to launch
  kernels. Consequence: Adaptive KV cannot speed up B=1 decode in this regime (it removes GPU work
  that is already hidden).
- Spec-only: 2.32× at 1k, falling to 0.72× at 16k. Proposed beats Spec at every length
  (e.g. 0.98× vs 0.72× at 16k), but still loses to Baseline beyond ~16k.
- OOM warning at 32k (likely full-vocab logits over the whole sequence:
  32,768 × 151,936 ≈ 10 GB FP16).
- First config of each sweep (1k, g=3 spec) is an outlier (7 s TTFT): first-call/hub-download
  effect. Needs per-config warmup.

### A100 run (latest, regressed / unclear)
- Baseline TPOT **45.8 ms on an A100**, same as the L4. An A100 should do a FP16 7B step in
  ~10–15 ms. So the overhead fixes (CUDA graphs/compile) were not active or silently failed, or
  bnb 4-bit was still on.
- Spec-only 1.04× (L4 had 2.32× at 1k). Cause unknown; acceptance/γ/d-v split not reported.
- PPL identical (13.08) for all 4 methods → PPL path does not go through the compressed cache.
- AdaptKV "Pruning + INT" warm-tier quantization was enabled (new; adds per-layer `torch.cat`),
  giving 1.95× KV ratio instead of 4×.
- Short prompt (TTFT 74 ms): this wasn't the context sweep at all.
- **Lesson: every run must print an environment/config block, and every spec row must include
  acceptance, tokens/step, γ and the d/v split. Otherwise the run can't be interpreted.**

---

## 4. What to build (the "best proposed" system)

Implement each item behind a config flag (for ablation), defaults ON for the final system.

### 4.1 Model loading
- Target and draft in **FP16 (or BF16 on A100/L4)**, no bitsandbytes by default.
  `--target_quant {fp16,bf16,bnb4,awq}`. On L4, FP16 7B (~15 GB) + draft + KV should fit ≤16k;
  verify 32k and skip gracefully if it OOMs.
- `attn_implementation="flash_attention_2"` for **both** models when `flash_attn` is importable.
  FA2 handles q_len>1 with a KV cache using bottom-right causal alignment natively. No 4D mask,
  no repeat_kv. Fallback: SDPA + the packed-GQA verify patch (4.3).
- Print `model.config._attn_implementation` for both models in the environment block.

### 4.2 Kill launch overhead (critical; the L4/A100 baselines are overhead-bound)
- Static KV caches + **CUDA graphs** (or `torch.compile(mode="reduce-overhead")`) for:
  target decode (q_len=1), target verify (fixed buckets q_len = 2..γ+1), and draft decode.
- Must be applied **equally to Baseline**. The baseline must be as fast as possible or the
  comparison is unfair.
- Detect and report capture failures / recompilations (log `torch._dynamo` recompiles). If capture
  fails, fall back and say so in the results; never silently.
- Expected: A100 FP16 baseline TPOT ~10–15 ms; L4 FP16 ~50 ms (L4 is memory-bandwidth-bound at
  ~300 GB/s, which is fine; that's where speculation shines).
- Adaptive-KV eviction must work with static buffers: compact kept K/V into the front of a
  preallocated buffer and track the physical length, rather than reallocating tensors.

### 4.3 Verify path (if FA2 unavailable)
- Packed-GQA SDPA for q_len>1: reshape query (b,28,q,d) → (b,4,7q,d) so each KV head serves its 7
  query heads directly, with mask expanded accordingly. No repeat_kv. Parity with HF output was
  verified (max_abs_diff = 0.0).
- **Do not pad K/V** (that copies the cache). Allocate the mask with last dim padded to a multiple
  of 8 and pass a sliced view `[..., :kv_len]`, which gives an aligned stride with no K/V copy.

### 4.4 Timing correctness
- `--timing_mode`: `torch.cuda.synchronize()` before every timer read (draft, verify, rollback).
  Use it to report the d/v split. Report throughput/TPOT from a run with timing_mode OFF.
- Use CUDA events where possible.

### 4.5 Adaptive KV (target)
- Exact budget: keep exactly ratio × length tokens (log actual kept count per layer).
- Policy: attention sinks (first 4–8 tokens) + recent window + heavy hitters by accumulated
  attention score. Per-layer budgets optional.
- **RoPE/positions:** keys are stored already rotated, so kept keys stay valid. New tokens must use
  their **true absolute position** for RoPE (`position_ids` = logical position), while the cache
  write index/attention length uses the **physical** kept length. Keep these two concepts separate
  everywhere (`cache_position`, `position_ids`, mask length, FA2 seqlens).
- **Compress only when beneficial:** add `--compress_min_len` (default e.g. 4096). Below it,
  Proposed behaves exactly like Spec-only (no eviction overhead, no PPL loss at 1k). Report this
  threshold in the paper.
- Warm-tier INT4/INT8 quantization **OFF by default** (flag to ablate). It adds per-step `torch.cat`
  and changes the KV ratio.
- Scoring under FA2: FA2 doesn't return attention weights. Get importance scores cheaply (e.g.
  compute q·k scores for the last token(s) on a subset of heads/layers, or score during prefill
  chunks), and measure that overhead (`c_ovh`).

### 4.6 Draft cache in Proposed
- Compress the draft cache with the **same keep-set of token positions** as the target (token
  indices are shared even though layer/head counts differ). This (a) makes total KV ≈ 4× smaller
  and (b) should raise acceptance, since draft and target see the same context.
  Flag `--draft_cache {full,shared_keepset}`; ablate both.
- Same draft cache policy at **every** context length. Remove the silent switch at
  max_position_embeddings. If a length exceeds the draft's limit, skip it with a clear message.
- Round any StaticCache `max_cache_len` up to a multiple of 8.

### 4.7 Speculation parameters
- Default γ = 4 or 5; optionally adaptive γ by measured acceptance. Report γ in every row.
- Greedy verification for the main results (deterministic); keep sampling as an option.

### 4.8 Memory / OOM
- `logits_to_keep=1` (or slice before lm_head) during prefill and decode; never materialize
  full-sequence logits except in the chunked PPL routine.
- Chunked prefill. Log peak GPU memory per run (`torch.cuda.max_memory_allocated`).

---

## 5. Benchmark protocol

- Contexts: 1k, 4k, 8k, 16k, 32k (skip with message if OOM / over draft limit).
- Generation: `max_new_tokens = 256`. Batch size 1 for the main table.
- **N documents per context (default 5)** from a standard long-context corpus (e.g. PG-19).
  Report mean ± std. Single documents gave PPL/acceptance noise of ±40%.
- Warmup **per configuration** (not just once), 3 repeats.
- Methods: Baseline, AdaptKV-only, Spec-only (γ=3,5), Proposed (γ=3,5).
- Per row, always print: decode tok/s, speedup, TPOT mean±std, TTFT, total tok/s, KV MB
  (target + draft separately and total), KV token ratio, PPL, acceptance %, tokens/step, γ,
  d/v/rollback/overhead ms (timing_mode), theoretical speedup vs measured, peak memory.
- Drop energy/token from headline tables (it's just 1/throughput).
- **Environment block at the top of every output:** GPU name, torch/CUDA/flash-attn versions,
  dtype/quant per model, attn_implementation per model, CUDA graphs/compile status, all
  method flags.
- Batch-size sweep at 8k/16k/32k: B = 1, 2, 4, 8, … until OOM, Baseline vs AdaptKV (and Proposed
  if feasible). Report tok/s and peak memory. This is where AdaptKV's memory win becomes throughput.
- Replace the "crossover at" summary with the **range** of contexts where each method beats
  Baseline.

### Ablation (cumulative), one notebook run
- S0: current defaults + timing fix
- S1: + FlashAttention-2
- S2: + draft cache consistency + shared keep-set + logits_to_keep + compress_min_len
- S3: + FP16/BF16 instead of bnb4
- S4: + CUDA graphs / torch.compile (all methods incl. Baseline)
Run S0–S4 at 1k/8k/16k (γ=5), then the full protocol on S4. Save results after each config.
Print one comparison table at the end.

---

## 6. Correctness checks (CPU, before every hand-off)

1. **Spec is lossless:** Spec-only greedy output tokens == Baseline greedy output tokens.
   Proposed greedy output == AdaptKV-only greedy output.
2. **Verify parity:** packed-GQA / FA2-equivalent paths give the same logits as the reference
   (fp tolerance; identical argmax).
3. **PPL goes through the method's cache:** teacher-force the continuation using the method's
   cache (with eviction). Assert AdaptKV PPL ≠ Baseline PPL when compression is active, and ==
   when `compress_min_len` > context.
4. **Budget:** kept tokens == round(ratio × length) per layer; target and draft keep-sets identical
   when `shared_keepset`.
5. **Positions:** after eviction, the next token's RoPE position == its logical position.
6. `--mock` end-to-end sweep runs on CPU without errors.

---

## 7. How to work with the user

- Explain in plain, short language; the user prefers crisp point-wise answers.
- Commit after each change (one feature per commit) so regressions can be bisected.
- Ship an updated `AdaptiveKVCache.zip`; keep notebook cells unchanged. Tell the user in 2–3
  lines what changed and which output files to send back.
- Recommend testing on **L4 first** (cheap), then A100 for the final numbers.
- When results come back, first check the environment block (FA2 active? FP16? graphs captured?),
  then the d/v split and acceptance, before drawing conclusions.

---

## 8. Realistic expectations (tell the user, don't oversell)

- **A100 / L4 with FP16 + FA2 + CUDA graphs:** decode becomes memory-bound. Spec decoding should
  give ~1.5–2.5× and hold with context, since verify ≈ one decode step. Proposed should stay ≥ Spec,
  and pull ahead at long context, where the compressed cache cuts draft+verify KV reads.
- **AdaptKV-only at B=1:** small speed gains at most, since KV is a small share of traffic next to
  15 GB of FP16 weights until very long context. Its main win is memory → batch size. Frame the
  paper's claim accordingly.
- If Proposed still loses at long context after S4, check in order: acceptance (keep-set
  mismatch?), verify time vs single decode (FA2 really active?), eviction overhead (`c_ovh`),
  graph recompiles.

---

## 9. First actions

1. Read §3 and §4. Inspect the files in §2 and map each item to the code.
2. Answer, from the code, what produced the A100 table (flags, prompt length, PPL path, why
   Spec = 1.04×). Report this to the user before changing anything.
3. Implement S0→S4 + §4.5–4.8 behind flags, with the §6 checks passing on CPU.
4. Package the zip + notebook and hand off to the user with run instructions.