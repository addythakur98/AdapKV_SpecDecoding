"""
Real-task benchmark runner: LongBench / LongBench-E / RULER x {Baseline, AdaptKV, Spec-only, Proposed}.

For every sample, every method gets the byte-identical prompt. Per sample we record the official task
score and the efficiency telemetry of that exact generation (decode ms/token, TTFT, acceptance,
tokens/step, KV MB, whether compression was active), so quality and speed come from the same run.

Correctness checks reported per task:
  * lossless rate: Spec-only output tokens == Baseline output tokens (greedy speculative decoding is
    lossless; a small mismatch rate can come from FP16 non-associativity between the q_len=1 decode
    and q_len=gamma+1 verify kernels and is reported, not hidden).
  * Proposed vs AdaptKV agreement: NOT expected to be 100%. Speculation appends gamma+1 tokens per
    verify step, so the cache is compressed at different points than in token-by-token AdaptKV and the
    kept set can differ slightly. The paired score delta quantifies the effect on quality.
  * compression coverage: share of samples whose prompt + generation >= compress_min_len.

Robustness for Colab: predictions are appended to a JSONL file after every sample and the run resumes
from it (rows from a different config fingerprint are ignored). OOM / over-length samples are logged
and skipped, never crash the run.
"""

from dataclasses import replace
import csv
import gc
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from config import AdaptiveCacheConfig, SpeculativeConfig, cache_config_for_context
from src.benchmark.runner import _is_oom_or_overflow, _clear_cuda_state
from src.benchmark.tasks.common import TaskSample, resolve_chat_template
from src.benchmark.tasks.longbench import (
    LONGBENCH_CATEGORY,
    LONGBENCH_DEFAULT_MAX_LEN,
    build_longbench_samples,
)
from src.benchmark.tasks.ruler import build_ruler_samples, word_source, load_essay_words, RULER_TASKS
from src.benchmark.tasks.scoring import (
    score_longbench_sample,
    scorer_backends,
    string_match_all,
    string_match_part,
)
from src.models.loader import LoadedModels
from src.speculative import static_runner as _sr
from src.speculative.baseline_decoder import GenerationOutput, autoregressive_generate
from src.speculative.speculative_decoder import speculative_generate

METHODS: Dict[str, Tuple[str, bool, bool]] = {
    # id: (label, use_spec, use_adaptive_kv)
    "baseline": ("Baseline", False, False),
    "adaptive_kv": ("AdaptKV", False, True),
    "speculative_only": ("Spec-only", True, False),
    "combined_proposed": ("Proposed", True, True),
}
LOSSLESS_PAIRS = (("speculative_only", "baseline"), ("combined_proposed", "adaptive_kv"))

DEVIATIONS = [
    "LongBench: official prompts / max_new_tokens / metrics / middle truncation; deterministic random subset "
    "of N samples per task (seeded) instead of all 150-500.",
    "RULER: official templates & task configs re-implemented; essay haystack = PG-19 test books instead of "
    "Paul Graham essays; QA tasks omitted (LongBench covers real QA); N samples per task/length instead of 500.",
    "Spec-only vs Baseline should match token-for-token (greedy, lossless). Proposed vs AdaptKV is reported as "
    "agreement, not losslessness: the eviction schedule differs (per verify block vs per token).",
    "KV compression is query-agnostic: the prompt is compressed chunk by chunk during prefill, before the "
    "question at the end of the prompt is seen (harder than SnapKV-style post-prefill selection).",
]


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------
def _generate(
    models: LoadedModels,
    prompt: str,
    max_new_tokens: int,
    use_spec: bool,
    cache_cfg: AdaptiveCacheConfig,
    spec_cfg: SpeculativeConfig,
) -> GenerationOutput:
    if use_spec:
        return speculative_generate(
            draft_model=models.draft_model,
            target_model=models.target_model,
            tokenizer=models.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            spec_config=spec_cfg,
            target_cache_config=cache_cfg,
            stop_on_eos=True,
        )
    return autoregressive_generate(
        model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        cache_config=cache_cfg,
        temperature=spec_cfg.temperature,
        top_p=spec_cfg.top_p,
        stop_on_eos=True,
        use_static_cache=bool((not cache_cfg.enabled) and spec_cfg.target_use_static_cache),
        compile_model=spec_cfg.compile_target,
        compile_mode=spec_cfg.compile_mode,
        round_static_cache_to_8=spec_cfg.round_static_cache_to_8,
        logits_to_keep=cache_cfg.logits_to_keep,
        timing_mode=spec_cfg.timing_mode,
        error_on_overflow=spec_cfg.draft_error_on_overflow,
    )


def _timed_generate(models, sample: TaskSample, use_spec, cache_cfg, spec_cfg) -> Tuple[GenerationOutput, bool]:
    """
    Runs one generation. If it had to compile / capture a new CUDA graph (first sample in a new draft
    StaticCache capacity bucket), the timing includes compilation, so the sample is run again warm and
    the second (identical, greedy) run is the one reported.
    """
    before = set(_sr._COMPILED_RUNNER_CACHE.keys())
    out = _generate(models, sample.prompt, sample.max_new_tokens, use_spec, cache_cfg, spec_cfg)
    if set(_sr._COMPILED_RUNNER_CACHE.keys()) - before:
        out = _generate(models, sample.prompt, sample.max_new_tokens, use_spec, cache_cfg, spec_cfg)
        return out, True
    return out, False


def _score(sample: TaskSample, prediction: str) -> float:
    if sample.metric == "string_match_all":
        return string_match_all(prediction, sample.answers)
    if sample.metric == "string_match_part":
        return string_match_part(prediction, sample.answers)
    return score_longbench_sample(sample.task, sample.metric, prediction, sample.answers, sample.all_classes)


# --------------------------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------------------------
def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _bootstrap_ci(xs: Sequence[float], iters: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """95% percentile bootstrap CI of the mean (scores in [0,100])."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    import random as _r

    rng = _r.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def _ms_per_token(rows: List[Dict[str, Any]]) -> float:
    """Decode ms per generated token over all samples (total decode time / total decode tokens)."""
    t = sum(r["tpot_ms"] * max(0, r["n_gen"] - 1) for r in rows)
    n = sum(max(0, r["n_gen"] - 1) for r in rows)
    return t / n if n > 0 else float("nan")


def _fingerprint(meta: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _load_done(pred_path: str, fp: str) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    done: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    if not os.path.exists(pred_path):
        return done
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("fingerprint") == fp:
                done[(r["suite"], r["task"], r["sample_id"], r["method"])] = r
    return done


# --------------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------------
def build_task_samples(
    models: LoadedModels,
    suites: Sequence[str],
    longbench_tasks: Sequence[str],
    ruler_tasks: Sequence[str],
    ruler_lengths: Sequence[int],
    num_samples: int,
    max_len: int,
    use_chat: bool,
    longbench_dir: Optional[str],
    seed: int,
) -> List[List[TaskSample]]:
    """Returns sample groups (one per task, or per task x length for RULER)."""
    groups: List[List[TaskSample]] = []
    tok = models.tokenizer
    for suite in suites:
        if suite in ("longbench", "longbench_e"):
            for task in longbench_tasks:
                try:
                    s = build_longbench_samples(
                        tok, task, num_samples, max_length=max_len, use_chat=use_chat,
                        use_e=(suite == "longbench_e"), data_dir=longbench_dir, seed=seed,
                    )
                except Exception as exc:
                    print(f"[Tasks] Could not load {suite}/{task}: {type(exc).__name__}: {exc}. Skipping.")
                    continue
                # Sort by length: draft StaticCache buckets then grow monotonically (each compiled once).
                groups.append(sorted(s, key=lambda x: x.prompt_tokens))
        elif suite == "ruler":
            for L in ruler_lengths:
                for task in ruler_tasks:
                    try:
                        groups.append(build_ruler_samples(tok, task, int(L), num_samples, use_chat=use_chat, seed=seed))
                    except ValueError as exc:
                        print(f"[Tasks] Skipping ruler/{task}@{L}: {exc}")
        else:
            raise ValueError(f"Unknown suite '{suite}' (use longbench, longbench_e, ruler).")
    return groups


def run_task_benchmark(
    models: LoadedModels,
    suites: Sequence[str],
    longbench_tasks: Sequence[str],
    ruler_tasks: Sequence[str],
    ruler_lengths: Sequence[int],
    num_samples: int,
    cache_config: AdaptiveCacheConfig,
    spec_config: SpeculativeConfig,
    methods: Sequence[str] = tuple(METHODS.keys()),
    max_len: int = LONGBENCH_DEFAULT_MAX_LEN,
    chat_template: str = "auto",
    model_names: Tuple[str, str] = ("", ""),
    longbench_dir: Optional[str] = None,
    output_dir: str = "results/tasks",
    seed: int = 42,
    resume: bool = True,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, "task_predictions.jsonl")

    target_name, draft_name = model_names
    use_chat = resolve_chat_template(chat_template, target_name, models.tokenizer)
    comp_cfg = cache_config if cache_config.cache_budget_ratio is not None else replace(cache_config, cache_budget_ratio=0.25)
    comp_cfg = replace(comp_cfg, enabled=True)
    full_cfg = AdaptiveCacheConfig(
        enabled=False, prefill_chunk_size=comp_cfg.prefill_chunk_size, logits_to_keep=comp_cfg.logits_to_keep
    )
    methods = [m for m in methods if m in METHODS]
    if models.draft_model is None:
        methods = [m for m in methods if not METHODS[m][1]]

    if "ruler" in suites and any(RULER_TASKS[t].get("haystack") == "essay" for t in ruler_tasks):
        _, essay_src = load_essay_words()
    else:
        essay_src = "n/a"

    run_meta = {
        "target_model": target_name,
        "draft_model": draft_name,
        "chat_template": use_chat,
        "max_len": max_len,
        "seed": seed,
        "num_samples": num_samples,
        "gamma": spec_config.gamma,
        "temperature": spec_config.temperature,
        "draft_cache": spec_config.resolved_draft_cache(),
        "cache_budget_ratio": comp_cfg.cache_budget_ratio,
        "max_cache_size": comp_cfg.max_cache_size,
        "compress_min_len": comp_cfg.compress_min_len,
        "num_sink_tokens": comp_cfg.num_sink_tokens,
        "recent_window_size": comp_cfg.recent_window_size,
        "heavy_hitter_ratio": comp_cfg.heavy_hitter_ratio,
        "target_quant": getattr(models, "target_quant", "?"),
        "attn_impl": getattr(models.target_model.config, "_attn_implementation", "?"),
    }
    fp = _fingerprint(run_meta)
    done = _load_done(pred_path, fp) if resume else {}
    if done:
        print(f"[Tasks] Resuming: {len(done)} finished (sample, method) rows found in {pred_path} (fingerprint {fp}).")

    print("\n" + "=" * 100)
    print("REAL-TASK BENCHMARK")
    print("-" * 100)
    for k, v in run_meta.items():
        print(f"  {k:<20}: {v}")
    print(f"  {'suites':<20}: {', '.join(suites)}")
    print(f"  {'methods':<20}: {', '.join(METHODS[m][0] for m in methods)}")
    print(f"  {'scorers':<20}: {scorer_backends()}")
    print(f"  {'ruler words':<20}: {word_source() if 'ruler' in suites else 'n/a'} | essay haystack: {essay_src}")
    print("=" * 100)

    t_build = time.time()
    groups = build_task_samples(
        models, suites, longbench_tasks, ruler_tasks, ruler_lengths, num_samples, max_len, use_chat, longbench_dir, seed
    )
    n_total = sum(len(g) for g in groups)
    print(f"[Tasks] Built {n_total} samples in {len(groups)} groups ({time.time() - t_build:.1f}s).")

    all_rows: List[Dict[str, Any]] = list(done.values())
    with open(pred_path, "a", encoding="utf-8") as pred_f:
        for g_idx, group in enumerate(groups):
            if not group:
                continue
            g0 = group[0]
            g_label = f"{g0.suite}/{g0.task}" + (f"@{g0.context_length}" if g0.suite == "ruler" else "")
            toks = [s.prompt_tokens for s in group]
            print(
                f"\n[Tasks] ({g_idx + 1}/{len(groups)}) {g_label}: {len(group)} samples, prompt tokens "
                f"min/mean/max = {min(toks)}/{int(_mean(toks))}/{max(toks)}, max_new_tokens={g0.max_new_tokens}"
            )
            for m in methods:
                label, use_spec, use_adapt = METHODS[m]
                cfg = comp_cfg if use_adapt else full_cfg
                todo = [s for s in group if (s.suite, s.task, s.sample_id, m) not in done]
                if not todo:
                    continue
                # One short warmup per (group, method): kernels, allocator, hooks.
                try:
                    _generate(models, todo[0].prompt, 4, use_spec, cfg, spec_config)
                except Exception as exc:
                    if not _is_oom_or_overflow(exc)[0]:
                        raise
                    _clear_cuda_state(models)
                g_scores: List[float] = []
                for s in todo:
                    gc.collect()
                    row: Dict[str, Any] = {
                        "fingerprint": fp, "suite": s.suite, "task": s.task, "sample_id": s.sample_id,
                        "method": m, "gamma": spec_config.gamma if use_spec else None,
                        "prompt_tokens": s.prompt_tokens, "context_length": s.context_length,
                        "length_bucket": s.length_bucket, "max_new_tokens": s.max_new_tokens,
                        "metric": s.metric, "answers": s.answers, "meta": s.meta,
                        "compression_active": bool(
                            use_adapt and cache_config_for_context(cfg, s.prompt_tokens + s.max_new_tokens).enabled
                        ),
                    }
                    try:
                        out, rerun = _timed_generate(models, s, use_spec, cfg, spec_config)
                    except Exception as exc:
                        is_skip, reason = _is_oom_or_overflow(exc)
                        if not is_skip:
                            raise
                        _clear_cuda_state(models)
                        row.update(status="skipped", reason=reason[:300])
                        print(f"  {label:<10} {s.sample_id[:24]:<24} SKIPPED ({reason[:120]})")
                        pred_f.write(json.dumps(row) + "\n")
                        pred_f.flush()
                        all_rows.append(row)
                        continue
                    prompt_len = int(out.output_ids.shape[1]) - int(out.num_generated_tokens)
                    gen_ids = out.output_ids[0, prompt_len:].tolist()
                    cs = out.cache_stats or {}
                    dcs = out.draft_cache_stats or {}
                    score = _score(s, out.text)
                    g_scores.append(score)
                    row.update(
                        status="ok",
                        prediction=out.text,
                        gen_ids=gen_ids,
                        score=score,
                        n_gen=int(out.num_generated_tokens),
                        ttft_ms=float(out.ttft_ms),
                        tpot_ms=float(out.tpot_ms),
                        decode_tps=float(out.decode_tokens_per_sec),
                        acceptance=float(out.acceptance_rate) if use_spec else None,
                        tokens_per_step=float(out.mean_accepted_per_step) if use_spec else 1.0,
                        kv_target_mb=float(cs.get("actual_memory_mb", 0.0)),
                        kv_target_full_mb=float(cs.get("uncompressed_memory_mb", 0.0)),
                        kv_draft_mb=float(dcs.get("actual_memory_mb", 0.0)),
                        pruned_tokens=int(cs.get("pruned_tokens", 0) or 0),
                        peak_vram_mb=float(getattr(out, "peak_vram_mb", 0.0)),
                        rerun_after_compile=rerun,
                    )
                    pred_f.write(json.dumps(row) + "\n")
                    pred_f.flush()
                    all_rows.append(row)
                print(
                    f"  {label:<10} score={100 * _mean(g_scores):6.2f} over {len(g_scores)} new samples"
                    if g_scores else f"  {label:<10} (no new samples)"
                )

    summary = summarize(all_rows, methods, run_meta)
    summary["run_meta"] = {**run_meta, "fingerprint": fp, "suites": list(suites), "scorers": scorer_backends(),
                           "ruler_words": word_source() if "ruler" in suites else None,
                           "essay_haystack": essay_src, "deviations_from_official": DEVIATIONS}
    _write_outputs(summary, output_dir)
    print_task_summary(summary)
    return summary


# --------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------
def _method_stats(rows: List[Dict[str, Any]], base_rows: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    scores = [100.0 * r["score"] for r in ok]
    st: Dict[str, Any] = {
        "n": len(rows),
        "n_ok": len(ok),
        "score": _mean(scores),
        "score_std": _std(scores),
        "score_ci95": _bootstrap_ci(scores),
        "compressed_frac": _mean([1.0 if r["compression_active"] else 0.0 for r in ok]) if ok else float("nan"),
        "score_on_compressed": _mean([100.0 * r["score"] for r in ok if r["compression_active"]]),
        "ms_per_token": _ms_per_token(ok),
        "ttft_ms": _mean([r["ttft_ms"] for r in ok]),
        "kv_mb": _mean([r["kv_target_mb"] + r["kv_draft_mb"] for r in ok]),
        "kv_full_mb": _mean([r["kv_target_full_mb"] for r in ok]),
        "acceptance_pct": _mean([100.0 * r["acceptance"] for r in ok if r.get("acceptance") is not None]),
        "tokens_per_step": _mean([r["tokens_per_step"] for r in ok]),
        "mean_gen_tokens": _mean([r["n_gen"] for r in ok]),
    }
    if base_rows is not None:
        b = _ms_per_token([r for r in base_rows if r.get("status") == "ok"])
        st["decode_speedup"] = b / st["ms_per_token"] if st["ms_per_token"] and st["ms_per_token"] > 0 else float("nan")
    return st


def _lossless(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    ib = {r["sample_id"]: r for r in rows_b if r.get("status") == "ok"}
    pairs = [(r, ib[r["sample_id"]]) for r in rows_a if r.get("status") == "ok" and r["sample_id"] in ib]
    if not pairs:
        return {"n": 0}
    exact = sum(1 for a, b in pairs if a["gen_ids"] == b["gen_ids"])

    def prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n / max(1, max(len(a), len(b)))

    return {
        "n": len(pairs),
        "exact_match_pct": 100.0 * exact / len(pairs),
        "mean_common_prefix_pct": 100.0 * _mean([prefix(a["gen_ids"], b["gen_ids"]) for a, b in pairs]),
        "score_delta": 100.0 * _mean([a["score"] - b["score"] for a, b in pairs]),
    }


def summarize(rows: List[Dict[str, Any]], methods: Sequence[str], run_meta: Dict[str, Any]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str, Any], Dict[str, List[Dict[str, Any]]]] = {}
    for r in rows:
        if r.get("method") not in methods:
            continue
        ctx = r["context_length"] if r["suite"] == "ruler" else None
        groups.setdefault((r["suite"], r["task"], ctx), {}).setdefault(r["method"], []).append(r)

    task_rows: List[Dict[str, Any]] = []
    for (suite, task, ctx), by_m in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2] or 0, kv[0][1])):
        base = by_m.get("baseline")
        entry: Dict[str, Any] = {
            "suite": suite, "task": task, "context_length": ctx,
            "category": LONGBENCH_CATEGORY.get(task, "RULER" if suite == "ruler" else ""),
            "methods": {m: _method_stats(by_m[m], base) for m in methods if m in by_m},
            "lossless": {f"{a}_vs_{b}": _lossless(by_m.get(a, []), by_m.get(b, [])) for a, b in LOSSLESS_PAIRS
                         if a in by_m and b in by_m},
        }
        if suite == "longbench_e":
            buckets: Dict[str, Dict[str, float]] = {}
            for m in methods:
                for r in by_m.get(m, []):
                    if r.get("status") == "ok":
                        buckets.setdefault(r["length_bucket"], {}).setdefault(m, []).append(100.0 * r["score"])
            entry["buckets"] = {b: {m: _mean(v) for m, v in d.items()} for b, d in buckets.items()}
        task_rows.append(entry)

    # Suite-level averages (macro over tasks, like the official LongBench / RULER averages).
    suite_avgs: Dict[str, Dict[str, Any]] = {}
    for suite in sorted({t["suite"] for t in task_rows}):
        ts = [t for t in task_rows if t["suite"] == suite]
        keys = sorted({t["context_length"] for t in ts}, key=lambda x: x or 0) if suite == "ruler" else [None]
        for key in keys:
            sub = [t for t in ts if t["context_length"] == key]
            name = suite if key is None else f"{suite}@{key}"
            avg: Dict[str, Any] = {}
            for m in methods:
                vals = [t["methods"][m] for t in sub if m in t["methods"] and t["methods"][m]["n_ok"] > 0]
                if not vals:
                    continue
                all_ok = [r for t in sub for r in groups[(t["suite"], t["task"], t["context_length"])].get(m, [])
                          if r.get("status") == "ok"]
                all_base = [r for t in sub for r in groups[(t["suite"], t["task"], t["context_length"])].get("baseline", [])
                            if r.get("status") == "ok"]
                mpt = _ms_per_token(all_ok)
                bpt = _ms_per_token(all_base) if all_base else float("nan")
                avg[m] = {
                    "score": _mean(v["score"] for v in vals),
                    "acceptance_pct": _mean(v["acceptance_pct"] for v in vals if not math.isnan(v["acceptance_pct"])),
                    "tokens_per_step": _mean(v["tokens_per_step"] for v in vals),
                    "decode_speedup": (bpt / mpt) if mpt and mpt > 0 and not math.isnan(bpt) else float("nan"),
                    "kv_mb": _mean(v["kv_mb"] for v in vals),
                    "kv_full_mb": _mean(v["kv_full_mb"] for v in vals),
                    "compressed_frac": _mean(v["compressed_frac"] for v in vals),
                    "n_tasks": len(vals),
                }
            suite_avgs[name] = avg
    return {"tasks": task_rows, "suite_averages": suite_avgs, "methods": list(methods)}


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def summary_markdown(summary: Dict[str, Any]) -> str:
    methods = summary["methods"]
    labels = [METHODS[m][0] for m in methods]
    out: List[str] = []
    meta = summary.get("run_meta", {})
    out.append("# Real-task benchmark results\n")
    out.append(
        f"Target `{meta.get('target_model')}` | draft `{meta.get('draft_model')}` | chat template: "
        f"{meta.get('chat_template')} | gamma={meta.get('gamma')} | KV budget ratio={meta.get('cache_budget_ratio')} "
        f"(compress_min_len={meta.get('compress_min_len')}) | draft cache: {meta.get('draft_cache')} | "
        f"{meta.get('num_samples')} samples/task | seed {meta.get('seed')}\n"
    )
    for suite in ("longbench", "longbench_e", "ruler"):
        ts = [t for t in summary["tasks"] if t["suite"] == suite]
        if not ts:
            continue
        out.append(f"\n## {suite}: task scores (0-100)\n")
        hdr = "| Task | Ctx | " + " | ".join(labels) + " | AdaptKV keeps | Spec=Base (lossless) | Prop=AdaptKV (agree) | Compressed |"
        out.append(hdr)
        out.append("|" + "---|" * (hdr.count("|") - 1))
        for t in ts:
            ms = t["methods"]
            cells = [_fmt(ms[m]["score"]) if m in ms else "-" for m in methods]
            keep = "-"
            if "baseline" in ms and "adaptive_kv" in ms and ms["baseline"]["score"] > 0:
                keep = f"{100 * ms['adaptive_kv']['score'] / ms['baseline']['score']:.1f}%"
            ll = t.get("lossless", {})
            l1 = ll.get("speculative_only_vs_baseline", {}).get("exact_match_pct")
            l2 = ll.get("combined_proposed_vs_adaptive_kv", {}).get("exact_match_pct")
            comp = ms.get("adaptive_kv", {}).get("compressed_frac")
            out.append(
                f"| {t['task']} | {t['context_length'] or '-'} | " + " | ".join(cells)
                + f" | {keep} | {_fmt(l1, 0)}% | {_fmt(l2, 0)}% | {_fmt(100 * comp if comp is not None else None, 0)}% |"
            )
        if suite == "longbench_e":
            out.append("\n### LongBench-E by length bucket (macro avg over tasks)\n")
            out.append("| Bucket | " + " | ".join(labels) + " |")
            out.append("|---|" + "---|" * len(labels))
            for b in ("0-4k", "4-8k", "8k+"):
                cells = []
                for m in methods:
                    vals = [t["buckets"][b][m] for t in ts if b in t.get("buckets", {}) and m in t["buckets"][b]]
                    cells.append(_fmt(_mean(vals)))
                out.append(f"| {b} | " + " | ".join(cells) + " |")

    out.append("\n## Averages: quality + efficiency on the same generations\n")
    out.append("| Suite | Method | Avg score | Decode speedup vs Baseline | Acceptance % | Tokens/step | KV MB (target+draft) | KV MB uncompressed | Compressed samples |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for name, avg in summary["suite_averages"].items():
        for m in methods:
            if m not in avg:
                continue
            a = avg[m]
            out.append(
                f"| {name} | {METHODS[m][0]} | {_fmt(a['score'])} | {_fmt(a['decode_speedup'])}x | "
                f"{_fmt(a['acceptance_pct'], 1)} | {_fmt(a['tokens_per_step'])} | {_fmt(a['kv_mb'], 1)} | "
                f"{_fmt(a['kv_full_mb'], 1)} | {_fmt(100 * a['compressed_frac'], 0)}% |"
            )
    devs = meta.get("deviations_from_official") or []
    if devs:
        out.append("\n## Protocol notes\n")
        out += [f"- {d}" for d in devs]
        out.append(f"- Scorers: {meta.get('scorers')}")
        if meta.get("essay_haystack"):
            out.append(f"- RULER essay haystack source: {meta.get('essay_haystack')}; word lists: {meta.get('ruler_words')}")
    return "\n".join(out) + "\n"


def print_task_summary(summary: Dict[str, Any]) -> None:
    print("\n" + summary_markdown(summary))


def _write_outputs(summary: Dict[str, Any], output_dir: str) -> None:
    def _clean(o: Any) -> Any:
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o

    with open(os.path.join(output_dir, "task_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_clean(summary), f, indent=2, default=str)
    with open(os.path.join(output_dir, "task_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_markdown(summary))
    with open(os.path.join(output_dir, "task_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["suite", "task", "context_length", "method", "n_ok", "score", "score_std", "ci95_lo", "ci95_hi",
                    "decode_speedup", "ms_per_token", "ttft_ms", "acceptance_pct", "tokens_per_step", "kv_mb",
                    "kv_full_mb", "compressed_frac", "score_on_compressed"])
        for t in summary["tasks"]:
            for m, s in t["methods"].items():
                w.writerow([t["suite"], t["task"], t["context_length"], m, s["n_ok"], s["score"], s["score_std"],
                            s["score_ci95"][0], s["score_ci95"][1], s.get("decode_speedup"), s["ms_per_token"],
                            s["ttft_ms"], s["acceptance_pct"], s["tokens_per_step"], s["kv_mb"], s["kv_full_mb"],
                            s["compressed_frac"], s["score_on_compressed"]])
    try:
        from src.benchmark.plots import generate_task_plots

        generate_task_plots(summary, output_dir)
    except Exception as exc:  # plots must never lose a finished run
        print(f"[Tasks] Plot generation failed ({type(exc).__name__}: {exc}); JSON/CSV/MD written.")
