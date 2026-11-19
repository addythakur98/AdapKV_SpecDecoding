"""
4-Way Ablation Benchmark Runner.

Evaluates and compares:
  1. Baseline Autoregressive (Qwen2.5-7B with uncompressed Full-Precision KV Cache)
  2. Adaptive KV-Cache Only (Qwen2.5-7B with Dynamic Pruning + INT4/INT8 Quantization)
  3. Speculative Decoding Only (Qwen2.5-0.5B Draft + Qwen2.5-7B Target, Uncompressed KV)
  4. Proposed Combined Framework (Qwen2.5-0.5B + Qwen2.5-7B + Adaptive KV-Cache Compression)

Across metrics:
  - Throughput (tokens/sec) & Speedup
  - Latency: TTFT (ms) & TPOT (ms/token)
  - KV-Cache Memory Footprint (MB), Compression Ratio & Savings (%)
  - Target Forward Pass Reduction & Draft Acceptance Rate (%)
  - Energy per Token (Joules/token), Average Power (W) & Carbon Intensity (gCO2e/1k tokens)
  - Perplexity (PPL) & Token Quality Similarity (F1)
"""

import csv
from dataclasses import replace
import gc
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import torch

from config import AdaptiveCacheConfig, BenchmarkConfig, SpeculativeConfig
from src.benchmark.energy_monitor import EnergyMonitor
from src.benchmark.metrics import (
    build_long_context_prompt,
    check_passkey_hit,
    compute_ngram_f1,
    evaluate_perplexity,
    load_long_context_eval_text,
    load_long_context_eval_texts,
)
from src.benchmark.plots import (
    generate_benchmark_plots,
    generate_context_sweep_plots,
    generate_stage_comparison_plots,
)
from src.models.loader import LoadedModels, set_attn_implementation
from src.speculative.baseline_decoder import GenerationOutput, autoregressive_generate
from src.speculative.speculative_decoder import profile_speculative_step, speculative_generate
from src.speculative.static_runner import (
    ContextLengthOverflowError,
    dynamo_counters_summary,
    release_model_static_caches,
)


def _is_oom_or_overflow(exc: Exception) -> Tuple[bool, str]:
    """Detects CUDA OOM or strict sequence-length overflow so sweeps skip cleanly without crashing."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True, f"CUDA OutOfMemoryError: {exc}"
    if isinstance(exc, ContextLengthOverflowError):
        return True, f"ContextLengthOverflow: {exc}"
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True, f"Runtime OOM: {exc}"
    return False, ""


def _clear_cuda_state(models: Optional[LoadedModels] = None) -> None:
    if models is not None:
        release_model_static_caches(models.target_model)
        if models.draft_model is not None:
            release_model_static_caches(models.draft_model)
    else:
        release_model_static_caches(None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _vram_mb_now() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024.0 ** 2), 2)
    return 0.0


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    """Computes population-safe sample mean and std (0.0 when len <= 1)."""
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(max(0.0, var))


def _theoretical_spec_speedup(alpha: float, gamma_eff: float, cost_ratio_c: float) -> float:
    """
    Leviathan et al. theoretical speculative speedup:
      S = (1 - alpha^(gamma + 1)) / ((1 - alpha) * (gamma * c + 1))
    """
    if gamma_eff <= 0.0:
        return 1.0
    a = min(1.0, max(0.0, alpha))
    denom_c = max(1e-6, gamma_eff * max(0.0, cost_ratio_c) + 1.0)
    if a >= 0.9999:
        return (gamma_eff + 1.0) / denom_c
    num = 1.0 - (a ** (gamma_eff + 1.0))
    return num / (max(1e-6, 1.0 - a) * denom_c)


# Short separate pass (timing_mode=ON) used ONLY to report the draft/verify split.
DV_PASS_MAX_NEW_TOKENS = 64


def _timing_label(spec_cfg: SpeculativeConfig) -> str:
    return "on" if spec_cfg.timing_mode else "off"


def _dv_split_pass(
    models: LoadedModels,
    prompt: str,
    cache_cfg: AdaptiveCacheConfig,
    spec_cfg: SpeculativeConfig,
    max_new_tokens: int,
) -> Dict[str, float]:
    """
    Runs one short speculative pass with `timing_mode=True` (torch.cuda.synchronize before every
    draft/verify/rollback timer read) to get a trustworthy per-step d/v split. Throughput/TPOT are
    never taken from this pass; they come from the timed runs (timing_mode OFF by default).
    """
    if models.draft_model is None:
        return {}
    n_tok = max(2, min(DV_PASS_MAX_NEW_TOKENS, int(max_new_tokens)))
    try:
        out = speculative_generate(
            draft_model=models.draft_model,
            target_model=models.target_model,
            tokenizer=models.tokenizer,
            prompt=prompt,
            max_new_tokens=n_tok,
            spec_config=replace(spec_cfg, timing_mode=True, profile_draft_step=True),
            target_cache_config=cache_cfg,
            stop_on_eos=False,
        )
    except Exception as exc:
        is_oom, reason = _is_oom_or_overflow(exc)
        if not is_oom:
            raise
        _clear_cuda_state(models)
        print(f"  [d/v pass] Skipped ({reason})")
        return {}
    return {
        "draft_time_ms": float(out.draft_time_ms),
        "verify_time_ms": float(out.verify_time_ms),
        "rollback_time_ms": float(out.rollback_time_ms),
        "cache_overhead_ms": float(out.cache_overhead_ms),
        "draft_ms_per_token": float(out.draft_ms_per_token),
        "draft_path": str(out.draft_path),
        "dv_pass_tokens": n_tok,
        "draft_diag": dict(out.draft_diag or {}),
        "draft_cache_mode": str(out.draft_cache_mode),
    }


def _print_draft_diag(label: str, diag: Dict[str, Any]) -> None:
    """One compact line per spec config: what one draft decode step (q_len=1) actually does on this device."""
    if not diag:
        return
    if "error" in diag:
        print(f"  [DraftStep] {label}: diagnostics failed ({diag['error']}); timings unaffected.")
        return
    ea = diag.get("eager_attention", {}) or {}
    top = diag.get("compiled_top_kernels") or diag.get("eager_top_kernels") or []
    top_s = ", ".join(f"{k['name'][:40]}={k['ms']:.2f}ms" for k in top[:3])
    cp = diag.get("compiled_copies", {}) or {}
    print(
        f"  [DraftStep] {label}: step={diag.get('step_ms', 0):.2f}ms | ctx logical={diag.get('logical_len')} "
        f"phys={diag.get('physical_len')} cap={diag.get('static_capacity')} | attn={ea.get('path')} "
        f"q={ea.get('q_len')} kv_read={ea.get('kv_len')} splits={ea.get('n_split')} mask={ea.get('mask_shape')} | "
        f"kv_static_addr={diag.get('kv_static_address')} | copies={cp.get('copy_ops', 0)} ({cp.get('copy_ms', 0):.2f}ms) | "
        f"path={diag.get('path')} | top: {top_s}"
    )


def _ppl_through_cache(
    models: LoadedModels,
    eval_docs: List[str],
    cache_cfg: AdaptiveCacheConfig,
    continuation_tokens: int,
    tag: str,
) -> Tuple[float, float, float]:
    """
    Teacher-forces each document through the target model using `cache_cfg` (the method's own KV cache:
    full cache for Baseline/Spec-only, AdaptiveKVCache with eviction for AdaptKV/Proposed). The prefix is
    compressed chunk by chunk and the held-out continuation is scored against the compressed cache.
    Returns (ppl_mean, ppl_std, mean_pruned_tokens).
    """
    doc_ppls: List[float] = []
    pruned: List[float] = []
    for d_idx, doc_txt in enumerate(eval_docs):
        _clear_cuda_state(models)
        try:
            p_res = evaluate_perplexity(
                models.target_model,
                models.tokenizer,
                [doc_txt],
                cache_config=cache_cfg,
                chunk_size=128,
                continuation_tokens=continuation_tokens,
                logits_to_keep=cache_cfg.logits_to_keep,
            )
        except Exception as exc:
            is_oom, reason = _is_oom_or_overflow(exc)
            if not is_oom:
                raise
            _clear_cuda_state(models)
            print(f"  [PPL] OOM during {tag} doc {d_idx} ({reason}); skipping remaining docs.")
            break
        doc_ppls.append(float(p_res["perplexity"]))
        pruned.append(float(p_res.get("pruned_tokens", 0)))
    m, s = _mean_std(doc_ppls)
    p_mean, _ = _mean_std(pruned)
    return m, s, p_mean


def _check_ppl_compression(where: str, ppl_full: float, ppl_comp: float, pruned_tokens: float) -> str:
    """Flags the two failure modes behind 'identical PPL for every method'. Returns a status tag."""
    if pruned_tokens <= 0:
        print(
            f"  [PPL CHECK] {where}: AdaptiveKV pruned 0 tokens during PPL (context < compress_min_len or "
            "<= budget), so AdaptKV PPL == Baseline PPL by construction."
        )
        return "compression_inactive"
    if ppl_full > 0 and abs(ppl_full - ppl_comp) < 1e-6:
        print(
            f"  [PPL CHECK] WARNING {where}: {pruned_tokens:.0f} tokens pruned but AdaptKV PPL == Baseline PPL "
            f"({ppl_comp:.4f}). PPL is likely not going through the compressed cache."
        )
        return "SUSPICIOUS_identical"
    print(
        f"  [PPL CHECK] {where}: pruned {pruned_tokens:.0f} tokens/doc -> PPL full={ppl_full:.3f} "
        f"vs AdaptKV={ppl_comp:.3f} ({(ppl_comp / max(1e-9, ppl_full) - 1.0) * 100.0:+.2f}%)"
    )
    return "ok"


def _print_results_table(title: str, rows: List[Dict[str, Any]]) -> None:
    """
    One table format for every mode. Always includes: prompt length, quant, attn_implementation,
    timing_mode (of the throughput runs), gamma, acceptance %, tokens/step and the d/v split
    (per step, from the separate timing_mode=ON pass).
    """
    width = 250
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(
        f"{'Method':<36} | {'Ctx':>6} | {'Prompt':>6} | {'Quant':<5} | {'Attn':<17} | {'Timing':<6} | "
        f"{'Dec tok/s':>9} | {'Spd':>6} | {'TPOT ms':>13} | {'g':>2} | {'Acc %':>6} | {'Tok/step':>8} | "
        f"{'d/v ms (ON pass)':>17} | {'d ms/tok':>8} | {'Draft path':<38} | {'KV MB':>9} | {'PPL':>13} | Status"
    )
    print("-" * width)
    for r in rows:
        label = str(r.get("stage_label") or r.get("name", r.get("id", "?")))
        ctx = r.get("context_length", "-")
        prompt_len = r.get("actual_prompt_tokens", "-")
        quant = str(r.get("target_quant", "-"))
        attn = str(r.get("attn_implementation", "-"))
        timing = str(r.get("timing_mode", "-"))
        status = str(r.get("status", "OK"))
        if status != "OK":
            print(
                f"{label[:36]:<36} | {str(ctx):>6} | {str(prompt_len):>6} | {quant:<5} | {attn:<17} | {timing:<6} | "
                f"{'-':>9} | {'-':>6} | {'-':>13} | {'-':>2} | {'-':>6} | {'-':>8} | {'-':>17} | {'-':>8} | {'-':<38} | {'-':>9} | {'-':>13} | {status}"
            )
            continue
        is_spec = int(r.get("gamma", 0) or 0) > 0
        tpot = f"{r['tpot_ms']:.2f}" + (f"±{r['tpot_ms_std']:.2f}" if r.get("tpot_ms_std") else "")
        acc = f"{r.get('acceptance_rate_pct', 0.0):.1f}" if is_spec else "-"
        tps = f"{r.get('mean_accepted_per_step', 0.0):.2f}" if is_spec else "1.00"
        dv = (
            f"{r.get('draft_time_ms', 0.0):.1f} / {r.get('verify_time_ms', 0.0):.1f}"
            if is_spec and r.get("dv_source") != "unavailable"
            else "-"
        )
        ppl = f"{r.get('perplexity', 0.0):.3f}" + (f"±{r['perplexity_std']:.2f}" if r.get("perplexity_std") else "")
        d_tok = f"{r.get('draft_ms_per_token', 0.0):.2f}" if is_spec and r.get("dv_source") != "unavailable" else "-"
        d_path = str(r.get("draft_path", "-")) if is_spec else "-"
        print(
            f"{label[:36]:<36} | {str(ctx):>6} | {str(prompt_len):>6} | {quant:<5} | {attn:<17} | {timing:<6} | "
            f"{r['decode_tokens_per_sec']:>9.2f} | {r.get('decode_speedup', 1.0):>5.2f}x | {tpot:>13} | "
            f"{int(r.get('gamma', 0) or 0):>2} | {acc:>6} | {tps:>8} | {dv:>17} | {d_tok:>8} | {d_path[:38]:<38} | "
            f"{r.get('kv_cache_actual_mb', 0.0):>9.2f} | {ppl:>13} | OK"
        )
    print("-" * width)
    print(
        "  Timing = timing_mode of the throughput/TPOT runs. d/v = per-step draft/verify ms from a separate short "
        f"pass with timing_mode=ON ({DV_PASS_MAX_NEW_TOKENS} tokens); d ms/tok = draft time per drafted token."
    )
    dyn = dynamo_counters_summary()
    if dyn:
        print(
            f"  torch.compile: unique_graphs={dyn.get('unique_graphs')} | graph_breaks={dyn.get('graph_breaks_total')} | "
            f"recompiles={dyn.get('recompiles')} | cudagraph_skips={dyn.get('cudagraph_skips')}"
            + (f" | break reasons: {dyn.get('graph_break_reasons')}" if dyn.get("graph_breaks_total") else "")
        )
    print(
        "  PPL = target teacher-forced through each method's own KV cache: Baseline/Spec-only use the full cache "
        "(spec decoding is lossless, so equal by construction); AdaptKV/Proposed use the compressed AdaptiveKVCache."
    )
    print("=" * width)


def run_ablation_benchmark(
    models: LoadedModels,
    bench_config: Optional[BenchmarkConfig] = None,
    cache_config: Optional[AdaptiveCacheConfig] = None,
    spec_config: Optional[SpeculativeConfig] = None,
) -> Dict[str, Any]:
    """Runs the complete 4-configuration ablation study and exports reports + charts."""
    b_cfg = bench_config or BenchmarkConfig()
    c_cfg = cache_config or AdaptiveCacheConfig(enabled=True)
    s_cfg = spec_config or SpeculativeConfig(enabled=True)

    os.makedirs(b_cfg.output_dir, exist_ok=True)

    # Perplexity through each method's own KV cache. The old short PPL text (~150 tok) never exceeded the
    # cache budget, so AdaptKV never pruned and PPL was identical for every method. Use long held-out docs
    # (prefix >= 8x the base budget) so eviction actually happens before the scored continuation.
    uncompressed_cache_cfg = AdaptiveCacheConfig(
        enabled=False,
        prefill_chunk_size=c_cfg.prefill_chunk_size,
        logits_to_keep=c_cfg.logits_to_keep,
    )
    ppl_prefix = max(2048, 8 * int(c_cfg.max_cache_size))
    ppl_cont = min(b_cfg.eval_continuation_tokens, max(32, ppl_prefix // 2))
    ppl_docs, ppl_source = load_long_context_eval_texts(
        models.tokenizer,
        prefix_tokens=ppl_prefix,
        continuation_tokens=ppl_cont,
        dataset_name=b_cfg.ppl_dataset,
        num_docs=max(1, int(b_cfg.num_eval_docs)),
    )
    print(
        f"\n[Benchmark] Evaluating PPL through each cache ({len(ppl_docs)} docs, prefix={ppl_prefix}, "
        f"scored continuation={ppl_cont}, source={ppl_source})..."
    )
    ppl_u_m, ppl_u_s, _ = _ppl_through_cache(models, ppl_docs, uncompressed_cache_cfg, ppl_cont, "Baseline")
    ppl_c_m, ppl_c_s, ppl_c_pruned = _ppl_through_cache(models, ppl_docs, c_cfg, ppl_cont, "AdaptKV")
    ppl_check = _check_ppl_compression(f"PPL prefix={ppl_prefix}", ppl_u_m, ppl_c_m, ppl_c_pruned)
    ppl_uncompressed = {"perplexity": round(ppl_u_m, 4), "perplexity_std": round(ppl_u_s, 4)}
    ppl_compressed = {
        "perplexity": round(ppl_c_m, 4),
        "perplexity_std": round(ppl_c_s, 4),
        "pruned_tokens_per_doc": round(ppl_c_pruned, 1),
        "check": ppl_check,
    }

    modes = [
        {
            "id": "baseline",
            "name": "1. Baseline (Qwen2.5-7B Standard KV)",
            "short_name": "Baseline\n(Standard KV)",
            "use_speculative": False,
            "use_adaptive_cache": False,
            "ppl": ppl_uncompressed["perplexity"],
        },
        {
            "id": "adaptive_kv",
            "name": "2. Adaptive KV-Cache Only",
            "short_name": "Adaptive KV\nOnly",
            "use_speculative": False,
            "use_adaptive_cache": True,
            "ppl": ppl_compressed["perplexity"],
        },
        {
            "id": "speculative_only",
            "name": "3. Speculative Decoding Only (0.5B -> 7B)",
            "short_name": "Speculative\nOnly",
            "use_speculative": True,
            "use_adaptive_cache": False,
            "ppl": ppl_uncompressed["perplexity"],
        },
        {
            "id": "combined_proposed",
            "name": "4. Proposed Combined (Speculative + Adaptive KV)",
            "short_name": "Proposed\n(Spec + AdaptKV)",
            "use_speculative": True,
            "use_adaptive_cache": True,
            "ppl": ppl_compressed["perplexity"],
        },
    ]

    monitor = EnergyMonitor(
        sample_interval_ms=b_cfg.power_sample_interval_ms,
        fallback_gpu_tdp_watts=b_cfg.fallback_gpu_tdp_watts,
        fallback_cpu_tdp_watts=b_cfg.fallback_cpu_tdp_watts,
    )

    # Optional warmup run to eliminate one-time CUDA kernel compilation overhead
    if b_cfg.num_warmup_runs > 0:
        print("[Benchmark] Performing warmup pass...")
        _ = autoregressive_generate(
            models.target_model,
            models.tokenizer,
            b_cfg.prompts[0],
            max_new_tokens=min(8, b_cfg.max_new_tokens),
            cache_config=uncompressed_cache_cfg,
            stop_on_eos=False,
        )
        if models.draft_model is not None:
            _ = speculative_generate(
                models.draft_model,
                models.target_model,
                models.tokenizer,
                b_cfg.prompts[0],
                max_new_tokens=min(8, b_cfg.max_new_tokens),
                spec_config=s_cfg,
                target_cache_config=c_cfg,
                stop_on_eos=False,
            )

    summary_rows: List[Dict[str, Any]] = []
    baseline_tokens_per_Prompt: List[List[int]] = []
    baseline_throughput: Optional[float] = None
    baseline_energy_per_tok: Optional[float] = None

    for mode in modes:
        print(f"\n[Benchmark] Running {mode['name']}...")
        active_cache_cfg = (
            c_cfg if mode["use_adaptive_cache"] else AdaptiveCacheConfig(enabled=False)
        )

        prompt_outputs: List[GenerationOutput] = []
        energy_records = []
        f1_scores: List[float] = []

        for p_idx, prompt in enumerate(b_cfg.prompts):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            monitor.start()
            if not mode["use_speculative"]:
                out = autoregressive_generate(
                    model=models.target_model,
                    tokenizer=models.tokenizer,
                    prompt=prompt,
                    max_new_tokens=b_cfg.max_new_tokens,
                    cache_config=active_cache_cfg,
                    temperature=s_cfg.temperature,
                    top_p=s_cfg.top_p,
                    stop_on_eos=False,
                )
            else:
                out = speculative_generate(
                    draft_model=models.draft_model,
                    target_model=models.target_model,
                    tokenizer=models.tokenizer,
                    prompt=prompt,
                    max_new_tokens=b_cfg.max_new_tokens,
                    spec_config=s_cfg,
                    target_cache_config=active_cache_cfg,
                    stop_on_eos=False,
                )
            energy = monitor.stop(
                num_generated_tokens=out.num_generated_tokens,
                kv_compression_ratio=out.cache_stats.get("compression_ratio", 1.0),
                target_passes=out.target_forward_passes,
            )

            prompt_outputs.append(out)
            energy_records.append(energy)

            gen_ids = out.output_ids[0, -out.num_generated_tokens :].tolist()
            if mode["id"] == "baseline":
                baseline_tokens_per_Prompt.append(gen_ids)
                f1_scores.append(1.0)
            else:
                ref_ids = baseline_tokens_per_Prompt[p_idx]
                f1_scores.append(compute_ngram_f1(ref_ids, gen_ids, n=1))

        # Aggregate metrics across prompts
        n_runs = len(prompt_outputs)
        avg_tps = sum(o.tokens_per_sec for o in prompt_outputs) / n_runs
        avg_decode_tps = sum(o.decode_tokens_per_sec for o in prompt_outputs) / n_runs
        avg_ttft = sum(o.ttft_ms for o in prompt_outputs) / n_runs
        avg_tpot = sum(o.tpot_ms for o in prompt_outputs) / n_runs
        avg_target_passes = sum(o.target_forward_passes for o in prompt_outputs) / n_runs
        avg_accept_rate = sum(o.acceptance_rate for o in prompt_outputs) / n_runs
        avg_mean_acc = sum(o.mean_accepted_per_step for o in prompt_outputs) / n_runs
        avg_draft_ms = sum(o.draft_time_ms for o in prompt_outputs) / n_runs
        avg_verify_ms = sum(o.verify_time_ms for o in prompt_outputs) / n_runs
        avg_cache_ms = sum(o.cache_overhead_ms for o in prompt_outputs) / n_runs
        avg_prompt_tokens = sum(
            int(o.output_ids.shape[1]) - int(o.num_generated_tokens) for o in prompt_outputs
        ) / n_runs

        # d/v split from a separate short timing_mode=ON pass (throughput above is from timing-OFF runs)
        dv_source = "n/a"
        d_tok_ms, d_path = 0.0, "-"
        if mode["use_speculative"]:
            dv = _dv_split_pass(models, b_cfg.prompts[0], active_cache_cfg, s_cfg, b_cfg.max_new_tokens)
            if dv:
                avg_draft_ms = dv["draft_time_ms"]
                avg_verify_ms = dv["verify_time_ms"]
                d_tok_ms, d_path = dv["draft_ms_per_token"], dv["draft_path"]
                dv_source = f"timing_mode=ON pass ({dv['dv_pass_tokens']} tok)"
            else:
                dv_source = "unavailable"

        avg_target_kv_actual = sum(o.cache_stats["actual_memory_mb"] for o in prompt_outputs) / n_runs
        avg_draft_kv_actual = (
            sum(o.draft_cache_stats.get("actual_memory_mb", 0.0) for o in prompt_outputs) / n_runs
        )
        avg_kv_actual = avg_target_kv_actual + avg_draft_kv_actual

        avg_target_kv_uncomp = (
            sum(o.cache_stats["uncompressed_memory_mb"] for o in prompt_outputs) / n_runs
        )
        avg_draft_kv_uncomp = (
            sum(o.draft_cache_stats.get("uncompressed_memory_mb", 0.0) for o in prompt_outputs)
            / n_runs
        )
        avg_kv_uncomp = avg_target_kv_uncomp + avg_draft_kv_uncomp
        avg_comp_ratio = avg_kv_uncomp / max(1e-6, avg_kv_actual)
        avg_mem_saved_pct = (1.0 - (avg_kv_actual / max(1e-6, avg_kv_uncomp))) * 100.0
        avg_energy_j_tok = sum(e.energy_per_token_joules for e in energy_records) / n_runs
        avg_power_w = sum(e.avg_power_watts for e in energy_records) / n_runs
        avg_co2 = sum(e.co2_grams_per_1k_tokens for e in energy_records) / n_runs
        peak_vram_mb = max(
            max((e.peak_gpu_memory_mb for e in energy_records), default=0.0),
            max((getattr(o, "peak_vram_mb", 0.0) for o in prompt_outputs), default=0.0),
        )
        avg_f1 = sum(f1_scores) / len(f1_scores)

        if baseline_throughput is None:
            baseline_throughput = avg_tps
            baseline_decode_tps = avg_decode_tps
        if baseline_energy_per_tok is None:
            baseline_energy_per_tok = avg_energy_j_tok

        speedup = avg_tps / max(1e-6, baseline_throughput)
        decode_speedup = avg_decode_tps / max(1e-6, baseline_decode_tps)
        energy_reduction_pct = (
            (1.0 - (avg_energy_j_tok / baseline_energy_per_tok)) * 100.0
            if baseline_energy_per_tok > 0
            else 0.0
        )

        row = {
            "id": mode["id"],
            "name": mode["name"],
            "short_name": mode["short_name"],
            "context_length": "short",
            "actual_prompt_tokens": int(round(avg_prompt_tokens)),
            "target_quant": str(getattr(models, "target_quant", "fp16")),
            "attn_implementation": str(getattr(models, "attn_implementation", "sdpa")),
            "timing_mode": _timing_label(s_cfg),
            "gamma": int(s_cfg.gamma) if mode["use_speculative"] else 0,
            "dv_source": dv_source,
            "draft_ms_per_token": round(d_tok_ms, 3),
            "draft_path": d_path,
            "status": "OK",
            "tokens_per_sec": round(avg_tps, 2),
            "decode_tokens_per_sec": round(avg_decode_tps, 2),
            "speedup": round(speedup, 2),
            "decode_speedup": round(decode_speedup, 2),
            "ttft_ms": round(avg_ttft, 2),
            "tpot_ms": round(avg_tpot, 2),
            "target_forward_passes": round(avg_target_passes, 1),
            "acceptance_rate_pct": round(avg_accept_rate * 100.0, 1),
            "mean_accepted_per_step": round(avg_mean_acc, 2),
            "draft_time_ms": round(avg_draft_ms, 2),
            "verify_time_ms": round(avg_verify_ms, 2),
            "cache_overhead_ms": round(avg_cache_ms, 3),
            "target_kv_actual_mb": round(avg_target_kv_actual, 4),
            "draft_kv_actual_mb": round(avg_draft_kv_actual, 4),
            "kv_cache_actual_mb": round(avg_kv_actual, 4),
            "kv_cache_uncompressed_mb": round(avg_kv_uncomp, 4),
            "compression_ratio": round(avg_comp_ratio, 2),
            "kv_memory_saved_pct": round(avg_mem_saved_pct, 2),
            "peak_memory_mb": round(peak_vram_mb, 2),
            "avg_power_watts": round(avg_power_w, 2),
            "energy_per_token_j": round(avg_energy_j_tok, 4),
            "energy_saved_pct": round(energy_reduction_pct, 2),
            "co2_g_per_1k_tokens": round(avg_co2, 5),
            "perplexity": mode["ppl"],
            "perplexity_std": (
                ppl_compressed["perplexity_std"] if mode["use_adaptive_cache"] else ppl_uncompressed["perplexity_std"]
            ),
            "ppl_cache": "adaptive_kv" if mode["use_adaptive_cache"] else "full",
            "ppl_check": ppl_check,
            "token_similarity_f1": round(avg_f1, 4),
            "power_sensor_source": energy_records[0].sensor_source,
            "sample_generation": prompt_outputs[0].text[:240],
        }
        summary_rows.append(row)

    # Save JSON and CSV artifacts
    json_path = os.path.join(b_cfg.output_dir, "benchmark_results.json")
    csv_path = os.path.join(b_cfg.output_dir, "benchmark_results.csv")
    plot_path = generate_benchmark_plots(summary_rows, output_dir=b_cfg.output_dir)

    report = {
        "project_title": (
            "Adaptive KV-Cache Compression and Speculative Decoding for "
            "Energy-Efficient LLM Inference on Low-Resource Hardware"
        ),
        "draft_model": models.draft_model.config._name_or_path
        if models.draft_model is not None
        else "N/A",
        "target_model": models.target_model.config._name_or_path,
        "device": str(models.device),
        "dtype": str(models.dtype),
        "perplexity_evaluation": {
            "uncompressed": ppl_uncompressed,
            "adaptive_compressed": ppl_compressed,
        },
        "ablation_results": summary_rows,
        "plot_path": plot_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if summary_rows:
        fieldnames = [k for k in summary_rows[0].keys() if k != "sample_generation"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow({k: r[k] for k in fieldnames})

    _print_summary_table(summary_rows, json_path, csv_path, plot_path)
    return report


def _print_summary_table(
    rows: List[Dict[str, Any]],
    json_path: str,
    csv_path: str,
    plot_path: str,
) -> None:
    _print_results_table(
        "ABLATION BENCHMARK SUMMARY (short prompts) — Qwen2.5-0.5B -> Qwen2.5-7B",
        rows,
    )
    print(f"  [Saved JSON]  {json_path}")
    print(f"  [Saved CSV]   {csv_path}")
    print(f"  [Saved Plot]  {plot_path}")
    print("=" * 124 + "\n")


def run_context_sweep(
    models: LoadedModels,
    lengths: Optional[List[int]] = None,
    bench_config: Optional[BenchmarkConfig] = None,
    cache_config: Optional[AdaptiveCacheConfig] = None,
    spec_config: Optional[SpeculativeConfig] = None,
    evaluate_ppl: bool = True,
    gammas: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    profile_step: bool = False,
) -> Dict[str, Any]:
    """
    Sweeps prompt context lengths (e.g. 1K -> 4K -> 8K -> 16K -> 32K) across all 4 modes
    using proportional adaptive KV budgeting, reporting crossover points, held-out continuation
    PPL, repeat mean ± std, empirical cost ratio `c` vs theoretical speedup, optional `gamma` /
    `batch_size` sweeps, and `torch.profiler` operator breakdown.
    """
    b_cfg = bench_config or BenchmarkConfig()
    base_cache_cfg = cache_config or AdaptiveCacheConfig(enabled=True)
    # Default to 25% proportional KV retention during long-context sweeps unless specified
    sweep_cache_cfg = (
        base_cache_cfg
        if base_cache_cfg.cache_budget_ratio is not None
        else replace(base_cache_cfg, cache_budget_ratio=0.25)
    )
    uncompressed_cfg = AdaptiveCacheConfig(
        enabled=False,
        prefill_chunk_size=sweep_cache_cfg.prefill_chunk_size,
    )
    s_cfg = spec_config or SpeculativeConfig(enabled=True)
    target_lengths = lengths or b_cfg.sweep_lengths
    max_new_toks = b_cfg.sweep_max_new_tokens
    num_repeats = max(1, b_cfg.num_repeats)
    gamma_list = [int(g) for g in gammas] if gammas else [int(s_cfg.gamma)]

    os.makedirs(b_cfg.output_dir, exist_ok=True)
    monitor = EnergyMonitor(
        sample_interval_ms=b_cfg.power_sample_interval_ms,
        fallback_gpu_tdp_watts=b_cfg.fallback_gpu_tdp_watts,
        fallback_cpu_tdp_watts=b_cfg.fallback_cpu_tdp_watts,
    )

def _compute_beats_baseline_ranges(
    sweep_rows: List[Dict[str, Any]],
    lengths: List[int],
) -> Dict[str, Any]:
    """
    Step 8: Computes context-length ranges `[min..max]` where each method beats Baseline,
    replacing single-point `"crossover at"` reporting.
    """
    sorted_lengths = sorted({int(x) for x in lengths})
    method_ids = []
    for r in sweep_rows:
        mid = str(r.get("stage_mode_key", r.get("id", "")))
        if mid and mid != "baseline" and not mid.endswith("::baseline") and mid not in method_ids:
            method_ids.append(mid)

    def _fmt_range(vals: List[int]) -> Dict[str, Any]:
        if not vals:
            return {"winning_lengths": [], "range": "none"}
        lo, hi = min(vals), max(vals)
        return {
            "winning_lengths": vals,
            "range": f"{lo}" if lo == hi else f"{lo}..{hi}",
        }

    ranges: Dict[str, Dict[str, Any]] = {}
    for mid in method_ids:
        win_dec: List[int] = []
        win_tot: List[int] = []
        win_tpot: List[int] = []
        win_kv: List[int] = []
        win_eng: List[int] = []
        win_all: List[int] = []
        for length in sorted_lengths:
            base_r = next(
                (
                    r for r in sweep_rows
                    if int(r["context_length"]) == length
                    and r.get("status", "OK") == "OK"
                    and r.get("id") == "baseline"
                ),
                None,
            )
            cands = [
                r for r in sweep_rows
                if int(r["context_length"]) == length
                and r.get("status", "OK") == "OK"
                and str(r.get("stage_mode_key", r.get("id", ""))) == mid
            ]
            prop_r = max(cands, key=lambda r: float(r.get("decode_tokens_per_sec", 0.0))) if cands else None
            if not base_r or not prop_r:
                continue
            b_dec = float(prop_r["decode_tokens_per_sec"]) > float(base_r["decode_tokens_per_sec"])
            b_tot = float(prop_r["tokens_per_sec"]) > float(base_r["tokens_per_sec"])
            b_tpot = float(prop_r["tpot_ms"]) < float(base_r["tpot_ms"])
            b_kv = float(prop_r["kv_cache_actual_mb"]) < float(base_r["kv_cache_actual_mb"])
            b_eng = float(prop_r["energy_per_token_j"]) < float(base_r["energy_per_token_j"])
            if b_dec:
                win_dec.append(length)
            if b_tot:
                win_tot.append(length)
            if b_tpot:
                win_tpot.append(length)
            if b_kv:
                win_kv.append(length)
            if b_eng:
                win_eng.append(length)
            if b_dec and b_tpot and b_kv and b_eng:
                win_all.append(length)

        ranges[mid] = {
            "decode_throughput": _fmt_range(win_dec),
            "total_throughput": _fmt_range(win_tot),
            "tpot": _fmt_range(win_tpot),
            "kv_memory": _fmt_range(win_kv),
            "energy": _fmt_range(win_eng),
            "all_efficiency_metrics": _fmt_range(win_all),
        }

    # Include top-level compatibility summary for "combined_proposed" (or best stage)
    comb = ranges.get("combined_proposed", {})
    return {
        "beats_baseline_ranges": ranges,
        "decode_throughput_beats_baseline_range": comb.get("decode_throughput", {}).get("range", "none"),
        "tpot_beats_baseline_range": comb.get("tpot", {}).get("range", "none"),
        "kv_memory_beats_baseline_range": comb.get("kv_memory", {}).get("range", "none"),
        "all_efficiency_beats_baseline_range": comb.get("all_efficiency_metrics", {}).get("range", "none"),
    }


def _warmup_single_config(
    models: LoadedModels,
    prompt: str,
    use_spec: bool,
    cache_cfg: AdaptiveCacheConfig,
    spec_cfg: SpeculativeConfig,
    max_new_tokens: int,
) -> None:
    """
    Step 7: Warms up a single configuration (compile / CUDA graphs / JIT / hooks) separately
    right before timed measurements so first-config compilation never skews TTFT/TPOT.
    """
    warm_tokens = min(8, max(2, max_new_tokens))
    try:
        if not use_spec:
            _ = autoregressive_generate(
                model=models.target_model,
                tokenizer=models.tokenizer,
                prompt=prompt,
                max_new_tokens=warm_tokens,
                cache_config=cache_cfg,
                temperature=0.0,
                stop_on_eos=False,
                use_static_cache=bool((not cache_cfg.enabled) and spec_cfg.target_use_static_cache),
                compile_model=spec_cfg.compile_target,
                compile_mode=spec_cfg.compile_mode,
                round_static_cache_to_8=spec_cfg.round_static_cache_to_8,
                logits_to_keep=cache_cfg.logits_to_keep,
                timing_mode=spec_cfg.timing_mode,
                error_on_overflow=spec_cfg.draft_error_on_overflow,
            )
        elif models.draft_model is not None:
            _ = speculative_generate(
                draft_model=models.draft_model,
                target_model=models.target_model,
                tokenizer=models.tokenizer,
                prompt=prompt,
                max_new_tokens=warm_tokens,
                spec_config=spec_cfg,
                target_cache_config=cache_cfg,
                stop_on_eos=False,
            )
    except Exception as exc:
        is_oom, reason = _is_oom_or_overflow(exc)
        if is_oom:
            print(f"  [Warmup] Skipped ({reason})")
            _clear_cuda_state(models)
        else:
            raise


def _run_batch_sweep(
    models: LoadedModels,
    batch_lengths: List[int],
    batch_sizes: List[int],
    sweep_cache_cfg: AdaptiveCacheConfig,
    uncompressed_cfg: AdaptiveCacheConfig,
    s_cfg: SpeculativeConfig,
    max_new_toks: int,
    monitor: EnergyMonitor,
) -> List[Dict[str, Any]]:
    """Step 8: Sweeps batch sizes B=1,2,4,8,... until OOM across target context lengths."""
    rows: List[Dict[str, Any]] = []
    for rep_len in batch_lengths:
        rep_prompt, _ = build_long_context_prompt(
            models.tokenizer, target_tokens=rep_len, passkey="73914", needle_depth=0.30
        )
        print(f"\n[Sweep] === Batch-Size Scaling at Context={rep_len} (Baseline vs Adaptive KV) ===")
        for mode_id, mode_name, use_adapt in [
            ("baseline", "1. Baseline (Standard KV)", False),
            ("adaptive_kv", "2. Adaptive KV Only", True),
        ]:
            active_cfg = sweep_cache_cfg if use_adapt else uncompressed_cfg
            for bs in batch_sizes:
                bs_int = max(1, int(bs))
                batched_prompts = [rep_prompt] * bs_int
                _clear_cuda_state(models)
                try:
                    monitor.start()
                    out = autoregressive_generate(
                        model=models.target_model,
                        tokenizer=models.tokenizer,
                        prompt=batched_prompts,
                        max_new_tokens=max_new_toks,
                        cache_config=active_cfg,
                        temperature=s_cfg.temperature,
                        top_p=s_cfg.top_p,
                        stop_on_eos=False,
                        logits_to_keep=active_cfg.logits_to_keep,
                        timing_mode=s_cfg.timing_mode,
                    )
                    energy = monitor.stop(
                        num_generated_tokens=out.num_generated_tokens,
                        kv_compression_ratio=out.cache_stats.get("compression_ratio", 1.0),
                        target_passes=out.target_forward_passes,
                    )
                    peak_mem = max(out.peak_vram_mb, energy.peak_gpu_memory_mb)
                    brow = {
                        "batch_size": bs_int,
                        "context_length": rep_len,
                        "id": mode_id,
                        "name": mode_name,
                        "status": "OK",
                        "decode_tokens_per_sec": out.decode_tokens_per_sec,
                        "tokens_per_sec": out.tokens_per_sec,
                        "tpot_ms": out.tpot_ms,
                        "ttft_ms": out.ttft_ms,
                        "kv_cache_actual_mb": out.cache_stats["actual_memory_mb"],
                        "compression_ratio": out.cache_stats["compression_ratio"],
                        "peak_memory_mb": peak_mem,
                        "energy_per_token_j": round(energy.energy_per_token_joules, 4),
                    }
                    rows.append(brow)
                    print(
                        f"  B={bs_int:<2} | {mode_name:<26} | Dec {brow['decode_tokens_per_sec']:>7.2f} tok/s "
                        f"| Tot {brow['tokens_per_sec']:>7.2f} tok/s | Step {brow['tpot_ms']:>6.2f}ms "
                        f"| Peak {brow['peak_memory_mb']:>7.1f}MB | KV {brow['kv_cache_actual_mb']:>7.2f}MB"
                    )
                except Exception as exc:
                    is_oom, reason = _is_oom_or_overflow(exc)
                    if not is_oom:
                        raise
                    try:
                        monitor.stop(num_generated_tokens=0, kv_compression_ratio=1.0, target_passes=0)
                    except Exception:
                        pass
                    _clear_cuda_state(models)
                    print(f"  B={bs_int:<2} | {mode_name:<26} | OOM/SKIP ({reason}) — stopping larger B for {mode_id}")
                    rows.append(
                        {
                            "batch_size": bs_int,
                            "context_length": rep_len,
                            "id": mode_id,
                            "name": mode_name,
                            "status": f"OOM_SKIPPED ({reason[:60]})",
                            "decode_tokens_per_sec": 0.0,
                            "tokens_per_sec": 0.0,
                            "tpot_ms": 0.0,
                            "ttft_ms": 0.0,
                            "kv_cache_actual_mb": 0.0,
                            "compression_ratio": 1.0,
                            "peak_memory_mb": _vram_mb_now(),
                            "energy_per_token_j": 0.0,
                        }
                    )
                    break
    return rows


def run_context_sweep(
    models: LoadedModels,
    lengths: Optional[List[int]] = None,
    bench_config: Optional[BenchmarkConfig] = None,
    cache_config: Optional[AdaptiveCacheConfig] = None,
    spec_config: Optional[SpeculativeConfig] = None,
    evaluate_ppl: bool = True,
    gammas: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    batch_lengths: Optional[List[int]] = None,
    profile_step: bool = False,
) -> Dict[str, Any]:
    """
    Sweeps prompt context lengths across all 4 modes using proportional adaptive KV budgeting,
    N-doc mean ± std reporting, per-config warmup, graceful OOM skip, peak VRAM tracking,
    winning context-length ranges vs baseline, optional gamma/batch scaling, and torch.profiler breakdown.
    """
    b_cfg = bench_config or BenchmarkConfig()
    base_cache_cfg = cache_config or AdaptiveCacheConfig(enabled=True)
    sweep_cache_cfg = (
        base_cache_cfg
        if base_cache_cfg.cache_budget_ratio is not None
        else replace(base_cache_cfg, cache_budget_ratio=0.25)
    )
    uncompressed_cfg = AdaptiveCacheConfig(
        enabled=False,
        prefill_chunk_size=sweep_cache_cfg.prefill_chunk_size,
        logits_to_keep=sweep_cache_cfg.logits_to_keep,
    )
    s_cfg = spec_config or SpeculativeConfig(enabled=True)
    target_lengths = lengths or b_cfg.sweep_lengths
    max_new_toks = b_cfg.sweep_max_new_tokens
    num_repeats = max(1, b_cfg.num_repeats)
    num_docs = max(1, int(getattr(b_cfg, "num_eval_docs", 5)))
    gamma_list = [int(g) for g in gammas] if gammas else [int(s_cfg.gamma)]

    os.makedirs(b_cfg.output_dir, exist_ok=True)
    monitor = EnergyMonitor(
        sample_interval_ms=b_cfg.power_sample_interval_ms,
        fallback_gpu_tdp_watts=b_cfg.fallback_gpu_tdp_watts,
        fallback_cpu_tdp_watts=b_cfg.fallback_cpu_tdp_watts,
    )

    modes = [
        ("baseline", "1. Baseline (Standard KV)", False, False),
        ("adaptive_kv", "2. Adaptive KV Only", False, True),
        ("speculative_only", "3. Speculative Only", True, False),
        ("combined_proposed", "4. Proposed (Spec + AdaptKV)", True, True),
    ]

    sweep_rows: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    profiler_traces: List[Dict[str, Any]] = []

    for idx_len, target_len in enumerate(target_lengths):
        # Generate N disjoint prompts/documents per context length (Step 7)
        doc_prompts: List[str] = []
        doc_passkeys: List[Optional[str]] = []
        for d_idx in range(num_docs):
            dp, pk = build_long_context_prompt(
                models.tokenizer,
                target_tokens=target_len,
                passkey="73914",
                needle_depth=0.30,
                para_offset=d_idx * 25000,
            )
            doc_prompts.append(dp)
            doc_passkeys.append(pk)

        prompt = doc_prompts[0]
        actual_prompt_tokens = int(
            models.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        )
        print(
            f"\n[Sweep] === Context Length Target={target_len} (Actual={actual_prompt_tokens} tokens, "
            f"Budget Ratio={sweep_cache_cfg.cache_budget_ratio}, Docs={num_docs}, Repeats={num_repeats}) ==="
        )
        _compress_on = actual_prompt_tokens + max_new_toks >= int(sweep_cache_cfg.compress_min_len)
        print(
            f"[Sweep] KV compression at this length: {'ON' if _compress_on else 'OFF'} "
            f"(prompt {actual_prompt_tokens} + {max_new_toks} new tokens vs compress_min_len={sweep_cache_cfg.compress_min_len}); "
            + (
                f"Proposed draft cache = {s_cfg.resolved_draft_cache()}"
                if _compress_on
                else "AdaptKV runs as Baseline and Proposed runs as Spec-only."
            )
        )

        ppl_uncomp_mean, ppl_uncomp_std = 0.0, 0.0
        ppl_comp_mean, ppl_comp_std = 0.0, 0.0
        ppl_source = "disabled"
        ppl_check = "disabled"
        if evaluate_ppl:
            cont_len = min(b_cfg.eval_continuation_tokens, max(32, target_len // 2))
            eval_docs, ppl_source = load_long_context_eval_texts(
                models.tokenizer,
                prefix_tokens=target_len,
                continuation_tokens=cont_len,
                dataset_name=b_cfg.ppl_dataset,
                num_docs=num_docs,
                fallback_prompts=doc_prompts,
            )

            ppl_uncomp_mean, ppl_uncomp_std, _ = _ppl_through_cache(
                models, eval_docs, uncompressed_cfg, cont_len, f"Baseline@{target_len}"
            )
            ppl_comp_mean, ppl_comp_std, ppl_pruned = _ppl_through_cache(
                models, eval_docs, sweep_cache_cfg, cont_len, f"AdaptKV@{target_len}"
            )
            ppl_check = _check_ppl_compression(
                f"Ctx={target_len}", ppl_uncomp_mean, ppl_comp_mean, ppl_pruned
            )

        baseline_gen_ids_per_doc: Dict[int, List[int]] = {}
        baseline_tps: Optional[float] = None
        baseline_decode_tps: Optional[float] = None
        baseline_tpot_ms: Optional[float] = None
        baseline_j: Optional[float] = None

        for mode_id, mode_name, use_spec, use_adapt in modes:
            mode_gammas = gamma_list if use_spec else [int(s_cfg.gamma)]
            active_cfg = sweep_cache_cfg if use_adapt else uncompressed_cfg

            for cur_gamma in mode_gammas:
                cur_spec_cfg = replace(s_cfg, gamma=cur_gamma)

                # Step 7: Warm up THIS configuration separately at THIS length before timing
                if b_cfg.num_warmup_runs > 0:
                    _warmup_single_config(
                        models=models,
                        prompt=prompt,
                        use_spec=use_spec,
                        cache_cfg=active_cfg,
                        spec_cfg=cur_spec_cfg,
                        max_new_tokens=max_new_toks,
                    )

                run_outs: List[GenerationOutput] = []
                run_energies = []
                run_doc_indices: List[int] = []
                oom_reason: Optional[str] = None

                for d_idx, doc_prompt in enumerate(doc_prompts):
                    for _rep in range(num_repeats):
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        try:
                            monitor.start()
                            if not use_spec:
                                out = autoregressive_generate(
                                    model=models.target_model,
                                    tokenizer=models.tokenizer,
                                    prompt=doc_prompt,
                                    max_new_tokens=max_new_toks,
                                    cache_config=active_cfg,
                                    temperature=cur_spec_cfg.temperature,
                                    top_p=cur_spec_cfg.top_p,
                                    stop_on_eos=False,
                                    use_static_cache=bool(
                                        (not active_cfg.enabled) and cur_spec_cfg.target_use_static_cache
                                    ),
                                    compile_model=cur_spec_cfg.compile_target,
                                    compile_mode=cur_spec_cfg.compile_mode,
                                    round_static_cache_to_8=cur_spec_cfg.round_static_cache_to_8,
                                    logits_to_keep=active_cfg.logits_to_keep,
                                    timing_mode=cur_spec_cfg.timing_mode,
                                    error_on_overflow=cur_spec_cfg.draft_error_on_overflow,
                                )
                            else:
                                out = speculative_generate(
                                    draft_model=models.draft_model,
                                    target_model=models.target_model,
                                    tokenizer=models.tokenizer,
                                    prompt=doc_prompt,
                                    max_new_tokens=max_new_toks,
                                    spec_config=cur_spec_cfg,
                                    target_cache_config=active_cfg,
                                    stop_on_eos=False,
                                )
                            energy = monitor.stop(
                                num_generated_tokens=out.num_generated_tokens,
                                kv_compression_ratio=out.cache_stats.get("compression_ratio", 1.0),
                                target_passes=out.target_forward_passes,
                            )
                            run_outs.append(out)
                            run_energies.append(energy)
                            run_doc_indices.append(d_idx)
                        except Exception as exc:
                            is_oom, reason = _is_oom_or_overflow(exc)
                            if not is_oom:
                                raise
                            try:
                                monitor.stop(num_generated_tokens=0, kv_compression_ratio=1.0, target_passes=0)
                            except Exception:
                                pass
                            _clear_cuda_state(models)
                            print(f"  {mode_name:<32} | Skipped at Target={target_len} ({reason})")
                            oom_reason = reason
                            break
                    if oom_reason is not None:
                        break

                if oom_reason is not None or not run_outs:
                    continue

                tps_mean, tps_std = _mean_std([o.tokens_per_sec for o in run_outs])
                dec_tps_mean, dec_tps_std = _mean_std([o.decode_tokens_per_sec for o in run_outs])
                ttft_mean, ttft_std = _mean_std([o.ttft_ms for o in run_outs])
                tpot_mean, tpot_std = _mean_std([o.tpot_ms for o in run_outs])
                acc_mean, acc_std = _mean_std([o.acceptance_rate * 100.0 for o in run_outs])
                mean_acc_step, _ = _mean_std([o.mean_accepted_per_step for o in run_outs])
                mean_draft_step, _ = _mean_std([o.mean_drafted_per_step for o in run_outs])
                draft_ms_mean, _ = _mean_std([o.draft_time_ms for o in run_outs])
                verify_ms_mean, _ = _mean_std([o.verify_time_ms for o in run_outs])
                rollback_ms_mean, _ = _mean_std([o.rollback_time_ms for o in run_outs])
                cache_ms_mean, _ = _mean_std([o.cache_overhead_ms for o in run_outs])
                j_mean, j_std = _mean_std([e.energy_per_token_joules for e in run_energies])

                # d/v split from a separate short timing_mode=ON pass; throughput/TPOT above stay
                # from the timed runs (timing_mode OFF by default).
                dv_source = "n/a"
                d_tok_ms, d_path = 0.0, "-"
                draft_diag_row: Dict[str, Any] = {}
                draft_cache_mode_row = ""
                if use_spec:
                    dv = _dv_split_pass(models, prompt, active_cfg, cur_spec_cfg, max_new_toks)
                    if dv:
                        d_tok_ms, d_path = dv["draft_ms_per_token"], dv["draft_path"]
                        _print_draft_diag(f"{mode_name} g={cur_gamma} @ {target_len}", dv.get("draft_diag", {}))
                        draft_diag_row = dv.get("draft_diag", {})
                        draft_cache_mode_row = dv.get("draft_cache_mode", "")
                        draft_ms_mean = dv["draft_time_ms"]
                        verify_ms_mean = dv["verify_time_ms"]
                        rollback_ms_mean = dv["rollback_time_ms"]
                        cache_ms_mean = dv["cache_overhead_ms"]
                        dv_source = f"timing_mode=ON pass ({dv['dv_pass_tokens']} tok)"
                    else:
                        dv_source = "unavailable"

                f1_list: List[float] = []
                for out_i, d_idx in zip(run_outs, run_doc_indices):
                    g_ids = out_i.output_ids[0, -out_i.num_generated_tokens :].tolist()
                    if mode_id == "baseline":
                        baseline_gen_ids_per_doc[d_idx] = g_ids
                        f1_list.append(1.0)
                    else:
                        f1_list.append(compute_ngram_f1(baseline_gen_ids_per_doc.get(d_idx, []), g_ids, n=1))
                f1_mean, _ = _mean_std(f1_list)

                last_out = run_outs[-1]
                if mode_id == "baseline":
                    baseline_tps = tps_mean
                    baseline_decode_tps = dec_tps_mean
                    baseline_tpot_ms = tpot_mean
                    baseline_j = j_mean

                speedup = tps_mean / max(1e-6, baseline_tps or tps_mean)
                decode_speedup = dec_tps_mean / max(1e-6, baseline_decode_tps or dec_tps_mean)
                energy_saved_pct = (
                    (1.0 - j_mean / baseline_j) * 100.0
                    if baseline_j and baseline_j > 0
                    else 0.0
                )
                needle_hit = check_passkey_hit(run_outs[0].text, doc_passkeys[0])

                target_kv_mb = float(last_out.cache_stats["actual_memory_mb"])
                draft_kv_mb = float(last_out.draft_cache_stats.get("actual_memory_mb", 0.0))
                total_kv_mb = target_kv_mb + draft_kv_mb

                target_kv_uncomp_mb = float(last_out.cache_stats["uncompressed_memory_mb"])
                draft_kv_uncomp_mb = float(last_out.draft_cache_stats.get("uncompressed_memory_mb", 0.0))
                total_kv_uncomp_mb = target_kv_uncomp_mb + draft_kv_uncomp_mb

                comp_ratio = total_kv_uncomp_mb / max(1e-6, total_kv_mb)
                kv_saved_pct = (1.0 - (total_kv_mb / max(1e-6, total_kv_uncomp_mb))) * 100.0
                tok_comp_ratio = float(last_out.cache_stats.get("token_compression_ratio", 1.0))
                tok_ret_pct = float(last_out.cache_stats.get("token_retention_pct", 100.0))

                cost_ratio_c = 0.0
                s_theory = 1.0
                if use_spec:
                    draft_per_tok_ms = draft_ms_mean / max(1.0, mean_draft_step)
                    ref_target_ms = max(1e-3, baseline_tpot_ms or verify_ms_mean)
                    cost_ratio_c = draft_per_tok_ms / ref_target_ms
                    s_theory = _theoretical_spec_speedup(
                        acc_mean / 100.0, mean_draft_step, cost_ratio_c
                    )

                peak_mem = max(
                    max((o.peak_vram_mb for o in run_outs), default=0.0),
                    max((e.peak_gpu_memory_mb for e in run_energies), default=0.0),
                )
                label = f"{mode_name} (g={cur_gamma})" if (use_spec and len(mode_gammas) > 1) else mode_name
                ppl_val = ppl_comp_mean if use_adapt else ppl_uncomp_mean
                ppl_std_val = ppl_comp_std if use_adapt else ppl_uncomp_std

                row = {
                    "context_length": target_len,
                    "actual_prompt_tokens": actual_prompt_tokens,
                    "num_generated_tokens": last_out.num_generated_tokens,
                    "num_docs": num_docs,
                    "num_repeats": num_repeats,
                    "status": "OK",
                    "attn_implementation": str(getattr(models, "attn_implementation", "sdpa")),
                    "target_quant": str(getattr(models, "target_quant", "fp16")),
                    "timing_mode": _timing_label(cur_spec_cfg),
                    "dv_source": dv_source,
                    "draft_ms_per_token": round(d_tok_ms, 3),
                    "draft_path": d_path,
                    "draft_cache_mode": draft_cache_mode_row,
                    "draft_step_diag": draft_diag_row,
                    "compress_min_len": int(active_cfg.compress_min_len),
                    "compression_active": bool(
                        active_cfg.enabled and actual_prompt_tokens + max_new_toks >= int(active_cfg.compress_min_len)
                    ),
                    "id": mode_id,
                    "name": label,
                    "gamma": cur_gamma if use_spec else 0,
                    "tokens_per_sec": round(tps_mean, 2),
                    "tokens_per_sec_std": round(tps_std, 2),
                    "decode_tokens_per_sec": round(dec_tps_mean, 2),
                    "decode_tokens_per_sec_std": round(dec_tps_std, 2),
                    "speedup": round(speedup, 2),
                    "decode_speedup": round(decode_speedup, 2),
                    "theoretical_speedup": round(s_theory, 2),
                    "cost_ratio_c": round(cost_ratio_c, 4),
                    "ttft_ms": round(ttft_mean, 2),
                    "ttft_ms_std": round(ttft_std, 2),
                    "tpot_ms": round(tpot_mean, 2),
                    "tpot_ms_std": round(tpot_std, 2),
                    "target_forward_passes": round(
                        sum(o.target_forward_passes for o in run_outs) / len(run_outs), 1
                    ),
                    "draft_forward_passes": round(
                        sum(o.draft_forward_passes for o in run_outs) / len(run_outs), 1
                    ),
                    "acceptance_rate_pct": round(acc_mean, 1),
                    "acceptance_rate_pct_std": round(acc_std, 1),
                    "mean_accepted_per_step": round(mean_acc_step, 2),
                    "mean_drafted_per_step": round(mean_draft_step, 2),
                    "draft_time_ms": round(draft_ms_mean, 2),
                    "verify_time_ms": round(verify_ms_mean, 2),
                    "rollback_time_ms": round(rollback_ms_mean, 3),
                    "cache_overhead_ms": round(cache_ms_mean, 3),
                    "target_kv_actual_mb": round(target_kv_mb, 4),
                    "draft_kv_actual_mb": round(draft_kv_mb, 4),
                    "kv_cache_actual_mb": round(total_kv_mb, 4),
                    "kv_cache_uncompressed_mb": round(total_kv_uncomp_mb, 4),
                    "compression_ratio": round(comp_ratio, 2),
                    "token_compression_ratio": round(tok_comp_ratio, 2),
                    "token_retention_pct": round(tok_ret_pct, 1),
                    "kv_memory_saved_pct": round(kv_saved_pct, 2),
                    "peak_memory_mb": round(peak_mem, 2),
                    "energy_per_token_j": round(j_mean, 4),
                    "energy_per_token_j_std": round(j_std, 4),
                    "energy_saved_pct": round(energy_saved_pct, 2),
                    "perplexity": round(ppl_val, 4),
                    "perplexity_std": round(ppl_std_val, 4),
                    "ppl_source": ppl_source,
                    "ppl_cache": "adaptive_kv" if use_adapt else "full",
                    "ppl_check": ppl_check,
                    "token_similarity_f1": round(f1_mean, 4),
                    "passkey_hit": needle_hit,
                    "power_sensor_source": run_energies[0].sensor_source,
                }
                step_sum_ms = (
                    row["draft_time_ms"]
                    + row["verify_time_ms"]
                    + row["rollback_time_ms"]
                    + row["cache_overhead_ms"]
                )
                row["step_sum_ms"] = round(step_sum_ms, 2)
                sweep_rows.append(row)

                multi_sample = (num_docs * num_repeats) > 1
                tpot_str = (
                    f"{row['tpot_ms']:.2f}±{row['tpot_ms_std']:.2f}ms"
                    if multi_sample
                    else f"{row['tpot_ms']:>6.2f}ms"
                )
                ppl_str = (
                    f"{row['perplexity']:.2f}±{row['perplexity_std']:.2f}"
                    if num_docs > 1 and evaluate_ppl
                    else f"{row['perplexity']:>6.2f}"
                )
                acc_str = (
                    f"{row['acceptance_rate_pct']:>4.1f}±{row['acceptance_rate_pct_std']:.1f}%"
                    if multi_sample
                    else f"{row['acceptance_rate_pct']:>4.1f}%"
                )
                spec_detail = (
                    f" | Draft {row['draft_path']} {row['draft_ms_per_token']:.2f}ms/tok"
                    f" | Acc {acc_str} ({row['mean_accepted_per_step']:.2f}/step)"
                    f" | Step={step_sum_ms:.1f}ms (d:{row['draft_time_ms']:.1f} v:{row['verify_time_ms']:.1f} rb:{row['rollback_time_ms']:.2f} c:{row['cache_overhead_ms']:.2f})"
                    if use_spec
                    else f" | c_ovh:{row['cache_overhead_ms']:.2f}ms"
                )
                print(
                    f"  {label:<32} | Dec {row['decode_tokens_per_sec']:>6.2f}/s ({row['decode_speedup']:.2f}x) "
                    f"| TPOT {tpot_str} | TTFT {row['ttft_ms']:>7.1f}ms | Peak {row['peak_memory_mb']:>6.1f}MB "
                    f"| KV {row['kv_cache_actual_mb']:>6.2f}MB ({row['compression_ratio']:.2f}x) "
                    f"| PPL {ppl_str}{spec_detail}"
                )

        # Save incremental CSV checkpoint after each context length
        if sweep_rows:
            ckpt_csv = os.path.join(b_cfg.output_dir, "context_sweep.csv")
            with open(ckpt_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)

        if profile_step and models.draft_model is not None and (idx_len == 0 or idx_len == len(target_lengths) - 1):
            print(f"  [Profiler] Profiling 1 speculative step at {actual_prompt_tokens} tokens...")
            try:
                prof_res = profile_speculative_step(
                    draft_model=models.draft_model,
                    target_model=models.target_model,
                    tokenizer=models.tokenizer,
                    prompt=prompt,
                    spec_config=s_cfg,
                    target_cache_config=sweep_cache_cfg,
                )
                profiler_traces.append(prof_res)
                for op in prof_res["top_ops"][:5]:
                    print(
                        f"    [Op] {op['name'][:42]:<42} | calls={op['calls']:>4} | "
                        f"cpu={op['self_cpu_ms']:>7.2f}ms | cuda={op['self_cuda_ms']:>7.2f}ms"
                    )
            except Exception as exc:
                is_oom, reason = _is_oom_or_overflow(exc)
                if not is_oom:
                    raise
                print(f"  [Profiler] Skipped ({reason})")

    # Release static pools before batch sweep
    _clear_cuda_state(models)

    if batch_sizes:
        if batch_lengths:
            bs_target_lengths = batch_lengths
        elif len(target_lengths) == 1:
            bs_target_lengths = list(target_lengths)
        else:
            rep_len = target_lengths[min(len(target_lengths) - 1, max(0, len(target_lengths) // 2))]
            bs_target_lengths = [rep_len]
        batch_rows = _run_batch_sweep(
            models=models,
            batch_lengths=bs_target_lengths,
            batch_sizes=batch_sizes,
            sweep_cache_cfg=sweep_cache_cfg,
            uncompressed_cfg=uncompressed_cfg,
            s_cfg=s_cfg,
            max_new_toks=max_new_toks,
            monitor=monitor,
        )

    crossover = _compute_beats_baseline_ranges(sweep_rows, target_lengths)

    json_path = os.path.join(b_cfg.output_dir, "context_sweep.json")
    csv_path = os.path.join(b_cfg.output_dir, "context_sweep.csv")
    plot_path = generate_context_sweep_plots(sweep_rows, output_dir=b_cfg.output_dir)

    report = {
        "draft_model": models.draft_model.config._name_or_path if models.draft_model else "N/A",
        "target_model": models.target_model.config._name_or_path,
        "attn_implementation": str(getattr(models, "attn_implementation", "sdpa")),
        "target_quant": str(getattr(models, "target_quant", "fp16")),
        "device": str(models.device),
        "dtype": str(models.dtype),
        "sweep_lengths": target_lengths,
        "sweep_gammas": gamma_list,
        "num_docs": num_docs,
        "num_repeats": num_repeats,
        "ppl_dataset": b_cfg.ppl_dataset or "local_heldout",
        "cache_budget_ratio": sweep_cache_cfg.cache_budget_ratio,
        "crossover_analysis": crossover,
        "sweep_results": sweep_rows,
        "batch_size_results": batch_rows,
        "profiler_traces": profiler_traces,
        "plot_path": plot_path,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if sweep_rows:
        fieldnames = list(sweep_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sweep_rows)

    _print_results_table(
        f"CONTEXT SWEEP SUMMARY (budget ratio={sweep_cache_cfg.cache_budget_ratio}, "
        f"{max_new_toks} new tokens, {num_docs} docs x {num_repeats} repeats)",
        sweep_rows,
    )
    print("\n" + "=" * 96)
    print(f"  CONTEXT RANGES BEATING BASELINE: {json.dumps(crossover['beats_baseline_ranges'])}")
    print(f"  [Saved JSON] {json_path} | [CSV] {csv_path} | [Plot] {plot_path}")
    print("=" * 96 + "\n")
    return report


def _stage_configs(base_spec: SpeculativeConfig, base_cache: AdaptiveCacheConfig) -> Dict[str, Dict[str, Any]]:
    """
    Cumulative S0 -> S4 optimization stage definitions matching Steps 1..8:
      S0: Timing fix + SDPA baseline (no compile, no logits_to_keep)
      S1: S0 + FlashAttention-2 (fallback sdpa with warning)
      S2: S1 + Draft cache consistency (StaticCache %8, overflow skip, logits_to_keep=True)
      S3: S2 + Target+Draft StaticCache + torch.compile(mode='reduce-overhead') CUDA graphs
      S4: S3 + Draft sharing target's adaptive keep-set (`share_draft_keep_set=True`)
    """
    return {
        "S0": {
            "label": "S0: TimingFix + SDPA",
            "attn": "sdpa",
            "spec": replace(
                base_spec,
                timing_mode=base_spec.timing_mode,
                compile_draft=False,
                compile_target=False,
                target_use_static_cache=False,
                consistent_draft_cache=True,
                draft_error_on_overflow=True,
                round_static_cache_to_8=False,
                share_draft_keep_set=False,
            ),
            "cache": replace(base_cache, logits_to_keep=False),
        },
        "S1": {
            "label": "S1: S0 + FlashAttention-2",
            "attn": "auto",
            "spec": replace(
                base_spec,
                timing_mode=base_spec.timing_mode,
                compile_draft=False,
                compile_target=False,
                target_use_static_cache=False,
                consistent_draft_cache=True,
                draft_error_on_overflow=True,
                round_static_cache_to_8=False,
                share_draft_keep_set=False,
            ),
            "cache": replace(base_cache, logits_to_keep=False),
        },
        "S2": {
            "label": "S2: S1 + Static%8 + LogitsToKeep",
            "attn": "auto",
            "spec": replace(
                base_spec,
                timing_mode=base_spec.timing_mode,
                compile_draft=False,
                compile_target=False,
                target_use_static_cache=False,
                consistent_draft_cache=True,
                draft_error_on_overflow=True,
                round_static_cache_to_8=True,
                share_draft_keep_set=False,
            ),
            "cache": replace(base_cache, logits_to_keep=True),
        },
        "S3": {
            "label": "S3: S2 + Compile(StaticCache)",
            "attn": "auto",
            "spec": replace(
                base_spec,
                timing_mode=base_spec.timing_mode,
                compile_draft=True,
                compile_target=True,
                target_use_static_cache=True,
                consistent_draft_cache=True,
                draft_error_on_overflow=True,
                round_static_cache_to_8=True,
                share_draft_keep_set=False,
            ),
            "cache": replace(base_cache, logits_to_keep=True),
        },
        "S4": {
            "label": "S4: S3 + SharedKeepSet",
            "attn": "auto",
            "spec": replace(
                base_spec,
                timing_mode=base_spec.timing_mode,
                compile_draft=True,
                compile_target=True,
                target_use_static_cache=True,
                consistent_draft_cache=True,
                draft_error_on_overflow=True,
                round_static_cache_to_8=True,
                share_draft_keep_set=True,
            ),
            "cache": replace(base_cache, logits_to_keep=True),
        },
    }


def _save_stage_outputs(
    output_dir: str,
    models: LoadedModels,
    lengths: List[int],
    stages: List[str],
    num_docs: int,
    num_repeats: int,
    gamma: int,
    ppl_dataset: Optional[str],
    budget_ratio: Optional[float],
    stage_rows: List[Dict[str, Any]],
    batch_rows: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Saves incremental results after each stage to both stages_comparison.* and context_sweep.*."""
    os.makedirs(output_dir, exist_ok=True)
    ok_rows = [r for r in stage_rows if r.get("status", "OK") == "OK"]
    crossover = _compute_beats_baseline_ranges(ok_rows, lengths)

    # Make context_sweep-compatible view from Baseline + S4 (or latest stage) for existing Colab readers
    sweep_plot = generate_stage_comparison_plots(ok_rows, output_dir=output_dir)

    report = {
        "draft_model": models.draft_model.config._name_or_path if models.draft_model else "N/A",
        "target_model": models.target_model.config._name_or_path,
        "attn_implementation": str(getattr(models, "attn_implementation", "sdpa")),
        "target_quant": str(getattr(models, "target_quant", "fp16")),
        "device": str(models.device),
        "dtype": str(models.dtype),
        "stages": stages,
        "sweep_lengths": lengths,
        "gamma": gamma,
        "num_docs": num_docs,
        "num_repeats": num_repeats,
        "ppl_dataset": ppl_dataset or "local_heldout",
        "cache_budget_ratio": budget_ratio,
        "crossover_analysis": crossover,
        "stage_results": stage_rows,
        "sweep_results": stage_rows,
        "batch_size_results": batch_rows,
        "plot_path": sweep_plot,
    }

    for stem in ("stages_comparison", "context_sweep"):
        j_path = os.path.join(output_dir, f"{stem}.json")
        c_path = os.path.join(output_dir, f"{stem}.csv")
        with open(j_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if stage_rows:
            fieldnames = list(stage_rows[0].keys())
            with open(c_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stage_rows)

    return os.path.join(output_dir, "stages_comparison.json"), os.path.join(output_dir, "stages_comparison.csv")


def _print_unified_stage_table(
    stage_rows: List[Dict[str, Any]],
    lengths: List[int],
    crossover: Dict[str, Any],
    json_path: str,
    csv_path: str,
    plot_path: str,
) -> None:
    _print_results_table("CUMULATIVE OPTIMIZATION COMPARISON TABLE (Baseline & S0 -> S4)", stage_rows)
    print(f"  Beats-Baseline Context Ranges: {json.dumps(crossover.get('beats_baseline_ranges', {}))}")
    print(f"  [Saved JSON] {json_path} | [CSV] {csv_path} | [Plot] {plot_path}")
    print("=" * 196 + "\n")


def run_stage_sweep(
    models: LoadedModels,
    lengths: Optional[List[int]] = None,
    stages: Optional[List[str]] = None,
    bench_config: Optional[BenchmarkConfig] = None,
    cache_config: Optional[AdaptiveCacheConfig] = None,
    spec_config: Optional[SpeculativeConfig] = None,
    evaluate_ppl: bool = True,
    batch_sizes: Optional[List[int]] = None,
    batch_lengths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Default cumulative benchmark runner: executes S0 -> S4 at 1k/8k/16k (gamma=5) on FP16,
    saves outputs incrementally after each stage, skips gracefully on OOM/overflow,
    and prints one unified comparison table at the end.
    """
    b_cfg = bench_config or BenchmarkConfig()
    base_cache = cache_config or AdaptiveCacheConfig(enabled=True)
    if base_cache.cache_budget_ratio is None:
        base_cache = replace(base_cache, cache_budget_ratio=0.25)
    base_spec = spec_config or SpeculativeConfig(enabled=True, gamma=5)
    target_lengths = lengths or b_cfg.sweep_lengths or [1024, 8192, 16384]
    stage_map = _stage_configs(base_spec, base_cache)
    stage_list = [s for s in (stages or b_cfg.stages or ["S0", "S1", "S2", "S3", "S4"]) if s in stage_map]
    num_docs = max(1, int(getattr(b_cfg, "num_eval_docs", 5)))
    num_repeats = max(1, b_cfg.num_repeats)
    max_new_toks = b_cfg.sweep_max_new_tokens

    os.makedirs(b_cfg.output_dir, exist_ok=True)
    monitor = EnergyMonitor(
        sample_interval_ms=b_cfg.power_sample_interval_ms,
        fallback_gpu_tdp_watts=b_cfg.fallback_gpu_tdp_watts,
        fallback_cpu_tdp_watts=b_cfg.fallback_cpu_tdp_watts,
    )

    # Pre-build N disjoint documents per context length
    prompts_by_len: Dict[int, List[str]] = {}
    eval_docs_by_len: Dict[int, Tuple[List[str], str]] = {}
    for target_len in target_lengths:
        d_prompts: List[str] = []
        for d_idx in range(num_docs):
            dp, _ = build_long_context_prompt(
                models.tokenizer,
                target_tokens=target_len,
                passkey="73914",
                needle_depth=0.30,
                para_offset=d_idx * 25000,
            )
            d_prompts.append(dp)
        prompts_by_len[target_len] = d_prompts
        if evaluate_ppl:
            cont_len = min(b_cfg.eval_continuation_tokens, max(32, target_len // 2))
            eval_docs_by_len[target_len] = load_long_context_eval_texts(
                models.tokenizer,
                prefix_tokens=target_len,
                continuation_tokens=cont_len,
                dataset_name=b_cfg.ppl_dataset,
                num_docs=num_docs,
                fallback_prompts=d_prompts,
            )

    all_rows: List[Dict[str, Any]] = []
    cached_ppl: Dict[Tuple[int, str, bool, bool], Tuple[float, float, str, str]] = {}
    prompt_tokens_by_len: Dict[int, int] = {
        L: int(models.tokenizer(p[0], return_tensors="pt")["input_ids"].shape[1])
        for L, p in prompts_by_len.items()
    }
    cached_baseline_ref: Dict[int, Dict[str, Any]] = {}
    cached_stage_baseline: Dict[Tuple[int, str, bool, bool, bool], Dict[str, Any]] = {}

    def _eval_and_record(
        stage_id: str,
        stage_label: str,
        mode_id: str,
        target_len: int,
        use_spec: bool,
        use_adapt: bool,
        cur_spec: SpeculativeConfig,
        cur_cache: AdaptiveCacheConfig,
        actual_attn: str,
        baseline_ref: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        d_prompts = prompts_by_len[target_len]
        active_cfg = cur_cache if use_adapt else AdaptiveCacheConfig(
            enabled=False,
            prefill_chunk_size=cur_cache.prefill_chunk_size,
            logits_to_keep=cur_cache.logits_to_keep,
        )

        # Evaluate or reuse PPL (mean ± std across N docs)
        ppl_m, ppl_s, ppl_src = 0.0, 0.0, "disabled"
        ppl_check = "disabled"
        if evaluate_ppl:
            ppl_key = (target_len, actual_attn, use_adapt, bool(active_cfg.logits_to_keep))
            if ppl_key in cached_ppl:
                ppl_m, ppl_s, ppl_src, ppl_check = cached_ppl[ppl_key]
            else:
                eval_docs, ppl_src = eval_docs_by_len[target_len]
                cont_len = min(b_cfg.eval_continuation_tokens, max(32, target_len // 2))
                ppl_m, ppl_s, ppl_pruned = _ppl_through_cache(
                    models, eval_docs, active_cfg, cont_len, f"{stage_id}:{mode_id}@{target_len}"
                )
                ppl_check = "full_cache"
                if use_adapt:
                    full_key = (target_len, actual_attn, False, bool(active_cfg.logits_to_keep))
                    if full_key not in cached_ppl:
                        full_cfg = AdaptiveCacheConfig(
                            enabled=False,
                            prefill_chunk_size=cur_cache.prefill_chunk_size,
                            logits_to_keep=cur_cache.logits_to_keep,
                        )
                        f_m, f_s, _ = _ppl_through_cache(
                            models, eval_docs, full_cfg, cont_len, f"{stage_id}:full@{target_len}"
                        )
                        cached_ppl[full_key] = (f_m, f_s, ppl_src, "full_cache")
                    ppl_check = _check_ppl_compression(
                        f"{stage_id} Ctx={target_len}", cached_ppl[full_key][0], ppl_m, ppl_pruned
                    )
                cached_ppl[ppl_key] = (ppl_m, ppl_s, ppl_src, ppl_check)

        if b_cfg.num_warmup_runs > 0:
            _warmup_single_config(
                models=models,
                prompt=d_prompts[0],
                use_spec=use_spec,
                cache_cfg=active_cfg,
                spec_cfg=cur_spec,
                max_new_tokens=max_new_toks,
            )

        run_outs: List[GenerationOutput] = []
        run_energies = []
        oom_reason: Optional[str] = None
        for d_prompt in d_prompts:
            for _rep in range(num_repeats):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    monitor.start()
                    if not use_spec:
                        out = autoregressive_generate(
                            model=models.target_model,
                            tokenizer=models.tokenizer,
                            prompt=d_prompt,
                            max_new_tokens=max_new_toks,
                            cache_config=active_cfg,
                            temperature=cur_spec.temperature,
                            top_p=cur_spec.top_p,
                            stop_on_eos=False,
                            use_static_cache=bool((not use_adapt) and cur_spec.target_use_static_cache),
                            compile_model=cur_spec.compile_target,
                            compile_mode=cur_spec.compile_mode,
                            round_static_cache_to_8=cur_spec.round_static_cache_to_8,
                            logits_to_keep=active_cfg.logits_to_keep,
                            timing_mode=cur_spec.timing_mode,
                            error_on_overflow=cur_spec.draft_error_on_overflow,
                        )
                    else:
                        out = speculative_generate(
                            draft_model=models.draft_model,
                            target_model=models.target_model,
                            tokenizer=models.tokenizer,
                            prompt=d_prompt,
                            max_new_tokens=max_new_toks,
                            spec_config=cur_spec,
                            target_cache_config=active_cfg,
                            stop_on_eos=False,
                        )
                    energy = monitor.stop(
                        num_generated_tokens=out.num_generated_tokens,
                        kv_compression_ratio=out.cache_stats.get("compression_ratio", 1.0),
                        target_passes=out.target_forward_passes,
                    )
                    run_outs.append(out)
                    run_energies.append(energy)
                except Exception as exc:
                    is_oom, reason = _is_oom_or_overflow(exc)
                    if not is_oom:
                        raise
                    try:
                        monitor.stop(num_generated_tokens=0, kv_compression_ratio=1.0, target_passes=0)
                    except Exception:
                        pass
                    _clear_cuda_state(models)
                    oom_reason = reason
                    print(f"  [{stage_id} | Ctx={target_len} | {mode_id}] OOM/SKIP ({reason})")
                    break
            if oom_reason is not None:
                break

        if oom_reason is not None or not run_outs:
            return {
                "stage": stage_id,
                "stage_label": f"{stage_label} ({mode_id})",
                "stage_mode_key": "baseline" if mode_id == "baseline" else f"{stage_id}::{mode_id}",
                "id": mode_id,
                "mode_id": mode_id,
                "context_length": target_len,
                "gamma": cur_spec.gamma if use_spec else 0,
                "num_docs": num_docs,
                "status": f"OOM_SKIPPED ({oom_reason or 'empty'})",
                "actual_prompt_tokens": prompt_tokens_by_len.get(target_len, 0),
                "attn_implementation": actual_attn,
                "target_quant": str(getattr(models, "target_quant", "fp16")),
                "timing_mode": _timing_label(cur_spec),
                "dv_source": "unavailable",
                "draft_ms_per_token": 0.0,
                "draft_path": "-",
                "decode_tokens_per_sec": 0.0,
                "decode_tokens_per_sec_std": 0.0,
                "tokens_per_sec": 0.0,
                "decode_speedup": 0.0,
                "speedup": 0.0,
                "tpot_ms": 0.0,
                "tpot_ms_std": 0.0,
                "ttft_ms": 0.0,
                "step_sum_ms": 0.0,
                "draft_time_ms": 0.0,
                "verify_time_ms": 0.0,
                "rollback_time_ms": 0.0,
                "cache_overhead_ms": 0.0,
                "acceptance_rate_pct": 0.0,
                "acceptance_rate_pct_std": 0.0,
                "mean_accepted_per_step": 0.0,
                "perplexity": round(ppl_m, 4),
                "perplexity_std": round(ppl_s, 4),
                "ppl_source": ppl_src,
                "ppl_cache": "adaptive_kv" if use_adapt else "full",
                "ppl_check": ppl_check,
                "kv_cache_actual_mb": 0.0,
                "kv_cache_uncompressed_mb": 0.0,
                "compression_ratio": 1.0,
                "peak_memory_mb": _vram_mb_now(),
                "energy_per_token_j": 0.0,
            }

        tps_m, _ = _mean_std([o.tokens_per_sec for o in run_outs])
        dec_m, dec_s = _mean_std([o.decode_tokens_per_sec for o in run_outs])
        tpot_m, tpot_s = _mean_std([o.tpot_ms for o in run_outs])
        ttft_m, _ = _mean_std([o.ttft_ms for o in run_outs])
        d_ms, _ = _mean_std([o.draft_time_ms for o in run_outs])
        v_ms, _ = _mean_std([o.verify_time_ms for o in run_outs])
        rb_ms, _ = _mean_std([o.rollback_time_ms for o in run_outs])
        c_ms, _ = _mean_std([o.cache_overhead_ms for o in run_outs])
        acc_m, acc_s = _mean_std([o.acceptance_rate * 100.0 for o in run_outs])
        mean_acc_step, _ = _mean_std([o.mean_accepted_per_step for o in run_outs])
        j_m, _ = _mean_std([e.energy_per_token_joules for e in run_energies])

        # d/v split from a separate short timing_mode=ON pass (throughput/TPOT stay from timed runs)
        dv_source = "n/a"
        d_tok_ms, d_path = 0.0, "-"
        if use_spec:
            dv = _dv_split_pass(models, d_prompts[0], active_cfg, cur_spec, max_new_toks)
            if dv:
                d_tok_ms, d_path = dv["draft_ms_per_token"], dv["draft_path"]
                d_ms, v_ms = dv["draft_time_ms"], dv["verify_time_ms"]
                rb_ms, c_ms = dv["rollback_time_ms"], dv["cache_overhead_ms"]
                dv_source = f"timing_mode=ON pass ({dv['dv_pass_tokens']} tok)"
            else:
                dv_source = "unavailable"

        last_out = run_outs[-1]
        t_kv = float(last_out.cache_stats["actual_memory_mb"])
        d_kv = float(last_out.draft_cache_stats.get("actual_memory_mb", 0.0))
        tot_kv = t_kv + d_kv
        uncomp_kv = float(last_out.cache_stats["uncompressed_memory_mb"]) + float(
            last_out.draft_cache_stats.get("uncompressed_memory_mb", 0.0)
        )
        peak_mem = max(
            max((o.peak_vram_mb for o in run_outs), default=0.0),
            max((e.peak_gpu_memory_mb for e in run_energies), default=0.0),
        )
        ref_dec = float(baseline_ref["decode_tokens_per_sec"]) if baseline_ref else dec_m
        ref_tot = float(baseline_ref["tokens_per_sec"]) if baseline_ref else tps_m

        step_sum = d_ms + v_ms + rb_ms + c_ms if use_spec else v_ms + c_ms
        return {
            "stage": stage_id,
            "stage_label": "Baseline (AR FP16)" if mode_id == "baseline" else stage_label if mode_id == "combined_proposed" else f"{stage_label} (SpecOnly)",
            "stage_mode_key": "baseline" if mode_id == "baseline" else f"{stage_id}::{mode_id}",
            "id": mode_id,
            "mode_id": mode_id,
            "context_length": target_len,
            "gamma": cur_spec.gamma if use_spec else 0,
            "num_docs": num_docs,
            "status": "OK",
            "actual_prompt_tokens": prompt_tokens_by_len.get(target_len, 0),
            "attn_implementation": actual_attn,
            "target_quant": str(getattr(models, "target_quant", "fp16")),
            "timing_mode": _timing_label(cur_spec),
            "dv_source": dv_source,
            "draft_ms_per_token": round(d_tok_ms, 3),
            "draft_path": d_path,
            "decode_tokens_per_sec": round(dec_m, 2),
            "decode_tokens_per_sec_std": round(dec_s, 2),
            "tokens_per_sec": round(tps_m, 2),
            "decode_speedup": round(dec_m / max(1e-6, ref_dec), 2),
            "speedup": round(tps_m / max(1e-6, ref_tot), 2),
            "tpot_ms": round(tpot_m, 2),
            "tpot_ms_std": round(tpot_s, 2),
            "ttft_ms": round(ttft_m, 2),
            "step_sum_ms": round(step_sum, 2),
            "draft_time_ms": round(d_ms, 2),
            "verify_time_ms": round(v_ms, 2),
            "rollback_time_ms": round(rb_ms, 3),
            "cache_overhead_ms": round(c_ms, 3),
            "acceptance_rate_pct": round(acc_m, 1),
            "acceptance_rate_pct_std": round(acc_s, 1),
            "mean_accepted_per_step": round(mean_acc_step, 2),
            "perplexity": round(ppl_m, 4),
            "perplexity_std": round(ppl_s, 4),
            "ppl_source": ppl_src,
            "ppl_cache": "adaptive_kv" if use_adapt else "full",
            "ppl_check": ppl_check,
            "kv_cache_actual_mb": round(tot_kv, 4),
            "kv_cache_uncompressed_mb": round(uncomp_kv, 4),
            "compression_ratio": round(uncomp_kv / max(1e-6, tot_kv), 2),
            "peak_memory_mb": round(peak_mem, 2),
            "energy_per_token_j": round(j_m, 4),
        }

    for st_id in stage_list:
        st_cfg = stage_map[st_id]
        actual_attn = set_attn_implementation(models, st_cfg["attn"])
        print(f"\n[Stages] === Running {st_cfg['label']} (attn={actual_attn}, quant={getattr(models, 'target_quant', 'fp16')}) ===")
        for target_len in target_lengths:
            # 1. Baseline AR reference for this length (reused from S0 when baseline ref established)
            b_key = (
                target_len,
                actual_attn,
                bool(st_cfg["spec"].target_use_static_cache),
                bool(st_cfg["spec"].compile_target),
                bool(st_cfg["cache"].logits_to_keep),
            )
            if target_len not in cached_baseline_ref:
                base_row = _eval_and_record(
                    st_id, st_cfg["label"], "baseline", target_len,
                    False, False, st_cfg["spec"], st_cfg["cache"], actual_attn, None,
                )
                cached_baseline_ref[target_len] = base_row
                cached_stage_baseline[b_key] = base_row
                all_rows.append(base_row)
            elif b_key not in cached_stage_baseline:
                stage_base_row = _eval_and_record(
                    st_id, f"{st_cfg['label']} (AR)", "baseline", target_len,
                    False, False, st_cfg["spec"], st_cfg["cache"], actual_attn, cached_baseline_ref[target_len],
                )
                cached_stage_baseline[b_key] = stage_base_row

            ref_base = cached_baseline_ref[target_len]
            if ref_base and ref_base.get("status") != "OK":
                ref_base = None

            # 2. Spec + AdaptKV (primary cumulative method per stage S0 -> S4)
            prop_row = _eval_and_record(
                st_id, st_cfg["label"], "combined_proposed", target_len,
                True, True, st_cfg["spec"], st_cfg["cache"], actual_attn, ref_base,
            )
            all_rows.append(prop_row)

            # 3. Speculative Only (to compare AdaptKV impact at each stage)
            if models.draft_model is not None:
                spec_only_row = _eval_and_record(
                    st_id, st_cfg["label"], "speculative_only", target_len,
                    True, False, st_cfg["spec"], st_cfg["cache"], actual_attn, ref_base,
                )
                all_rows.append(spec_only_row)

        # Save after EACH stage so partial results survive crashes/disconnects
        _clear_cuda_state(models)
        j_path, c_path = _save_stage_outputs(
            b_cfg.output_dir,
            models,
            target_lengths,
            stage_list,
            num_docs,
            num_repeats,
            base_spec.gamma,
            b_cfg.ppl_dataset,
            base_cache.cache_budget_ratio,
            all_rows,
            [],
        )
        print(f"  [Checkpoint] Saved outputs after {st_id} -> {j_path}")

    # Optional Step 8 Batch-Size Sweep at 8k/16k/32k (Baseline vs AdaptKV until OOM)
    batch_rows: List[Dict[str, Any]] = []
    if batch_sizes:
        bs_lens = batch_lengths or b_cfg.batch_sweep_lengths
        final_attn = set_attn_implementation(models, stage_map[stage_list[-1]]["attn"])
        final_cache = stage_map[stage_list[-1]]["cache"]
        final_spec = stage_map[stage_list[-1]]["spec"]
        uncomp_final = AdaptiveCacheConfig(
            enabled=False,
            prefill_chunk_size=final_cache.prefill_chunk_size,
            logits_to_keep=final_cache.logits_to_keep,
        )
        batch_rows = _run_batch_sweep(
            models=models,
            batch_lengths=bs_lens,
            batch_sizes=batch_sizes,
            sweep_cache_cfg=final_cache,
            uncompressed_cfg=uncomp_final,
            s_cfg=final_spec,
            max_new_toks=max_new_toks,
            monitor=monitor,
        )

    j_path, c_path = _save_stage_outputs(
        b_cfg.output_dir,
        models,
        target_lengths,
        stage_list,
        num_docs,
        num_repeats,
        base_spec.gamma,
        b_cfg.ppl_dataset,
        base_cache.cache_budget_ratio,
        all_rows,
        batch_rows,
    )
    ok_rows = [r for r in all_rows if r.get("status", "OK") == "OK"]
    crossover = _compute_beats_baseline_ranges(ok_rows, target_lengths)
    plot_path = os.path.join(b_cfg.output_dir, "stages_comparison.png")
    _print_unified_stage_table(all_rows, target_lengths, crossover, j_path, c_path, plot_path)

    return {
        "draft_model": models.draft_model.config._name_or_path if models.draft_model else "N/A",
        "target_model": models.target_model.config._name_or_path,
        "attn_implementation": str(getattr(models, "attn_implementation", "sdpa")),
        "target_quant": str(getattr(models, "target_quant", "fp16")),
        "stages": stage_list,
        "sweep_lengths": target_lengths,
        "crossover_analysis": crossover,
        "stage_results": all_rows,
        "sweep_results": all_rows,
        "batch_size_results": batch_rows,
        "json_path": j_path,
        "csv_path": c_path,
        "plot_path": plot_path,
    }
