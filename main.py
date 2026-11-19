"""
Main CLI Entrypoint for Adaptive KV-Cache Compression & Speculative Decoding.

Default Models:
  - Draft Model:  Qwen/Qwen2.5-0.5B
  - Target Model: Qwen/Qwen2.5-7B

Usage Examples:
  # 1. Context sweep (1k/4k/8k/16k) for all 4 methods, FP16 target (default quant):
  python main.py --mode sweep --sweep-lengths 1024,4096,8192,16384

  # 2. Run single-prompt generation with the combined framework:
  python main.py --mode generate --prompt "Explain how attention sinks stabilize long-context LLMs:"

  # 3. Run quick verification benchmark using tiny in-memory Qwen2 architecture (no 15GB download needed):
  python main.py --mode benchmark --mock
"""

import argparse
import sys

from config import AdaptiveCacheConfig, BenchmarkConfig, ModelConfig, SpeculativeConfig
from src.benchmark.energy_monitor import EnergyMonitor
from src.benchmark.runner import run_ablation_benchmark, run_context_sweep, run_stage_sweep
from src.benchmark.task_runner import run_task_benchmark
from src.models.loader import (
    FlashAttentionUnavailableError,
    create_tiny_qwen_pair_for_testing,
    load_model_pair,
    print_model_runtime_info,
)
from src.speculative.baseline_decoder import autoregressive_generate
from src.speculative.speculative_decoder import speculative_generate
from src.speculative.static_runner import dynamo_counters_summary, get_compile_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive KV-Cache Compression and Speculative Decoding for "
            "Energy-Efficient LLM Inference on Low-Resource Hardware "
            "(Draft: Qwen/Qwen2.5-0.5B, Target: Qwen/Qwen2.5-7B)"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stages", "benchmark", "sweep", "tasks", "generate", "interactive"],
        default="stages",
        help=(
            "Execution mode: cumulative 'stages' (default S0->S4 at 1k/8k/16k), "
            "4-way 'benchmark', context-length 'sweep', real-task 'tasks' (LongBench / RULER), "
            "single 'generate', or 'interactive' CLI."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="S0,S1,S2,S3,S4",
        help="Comma-separated cumulative stages for '--mode stages' (default: S0,S1,S2,S3,S4).",
    )
    parser.add_argument(
        "--context-sweep",
        action="store_true",
        help="Shortcut for `--mode sweep` to run context-length scaling.",
    )
    parser.add_argument(
        "--sweep-lengths",
        type=str,
        default="1024,8192,16384",
        help="Comma-separated prompt token lengths for stages/sweep (default: 1024,8192,16384).",
    )
    parser.add_argument(
        "--sweep-gammas",
        type=str,
        default=None,
        help="Optional comma-separated draft lookahead gammas for sweep (e.g. '2,4,6').",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Optional comma-separated batch sizes for Baseline vs AdaptiveKV scaling (e.g. '1,2,4,8').",
    )
    parser.add_argument(
        "--batch-sweep",
        "--batch_sweep",
        action="store_true",
        help="Run batch-size sweep (1,2,4,8...) at long contexts in addition to stages/sweep.",
    )
    parser.add_argument(
        "--batch-sweep-lengths",
        "--batch_sweep_lengths",
        type=str,
        default="8192,16384,32768",
        help="Comma-separated prompt lengths for batch sweep (default: 8192,16384,32768).",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Timed repetitions per document/configuration in sweep.",
    )
    parser.add_argument(
        "--num-eval-docs",
        "--num_eval_docs",
        type=int,
        default=5,
        help="Number of disjoint evaluation documents per context length (reports mean ± std).",
    )
    parser.add_argument(
        "--ppl-dataset",
        type=str,
        default=None,
        help="Optional HF corpus for held-out PPL ('pg19', 'wikitext', or dataset id).",
    )
    parser.add_argument(
        "--no-ppl",
        action="store_true",
        help="Skip held-out perplexity evaluation (faster throughput/timing checks).",
    )
    parser.add_argument(
        "--eval-continuation-tokens",
        type=int,
        default=256,
        help="Held-out continuation token count scored for PPL (outside compressed context).",
    )
    parser.add_argument(
        "--profile-step",
        action="store_true",
        help="Run torch.profiler on a speculative decode step at shortest & longest context.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face model ID or local path for the small draft model.",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Hugging Face model ID or local path for the target LLM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute precision for model weights and full-precision KV tier.",
    )
    parser.add_argument(
        "--attn-impl",
        "--attn_impl",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        help=(
            "Attention backend. On CUDA, 'auto' requires flash_attention_2 for both models and stops "
            "with an error if flash-attn is not installed (no sdpa fallback)."
        ),
    )
    parser.add_argument(
        "--target-quant",
        "--target_quant",
        type=str,
        default=None,
        choices=["fp16", "bnb4", "awq"],
        help="Target quantization mode: 'fp16' (default), 'bnb4', or 'awq' (Marlin).",
    )
    parser.add_argument(
        "--awq-backend",
        "--awq_backend",
        type=str,
        default="marlin",
        help="AWQ kernel backend when --target-quant awq (default: marlin).",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Deprecated alias for '--target-quant bnb4' (loads target in 4-bit NF4).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load Qwen2.5-7B target model weights in 8-bit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of new tokens to generate per prompt "
            "(defaults to 128 for benchmark/generate, 256 for context sweep)."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=5,
        help="Number of draft lookahead tokens per speculative step.",
    )
    parser.add_argument(
        "--adaptive-gamma",
        action="store_true",
        help="Enable confidence-based dynamic gamma early stopping (adds per-token CPU sync).",
    )
    parser.add_argument(
        "--no-adaptive-gamma",
        action="store_true",
        help="Deprecated alias (adaptive-gamma is now disabled by default).",
    )
    parser.add_argument(
        "--timing-mode",
        "--timing_mode",
        action="store_true",
        default=None,
        help=(
            "Also sync before timer reads in the timed throughput runs (default: OFF). The d/v split is "
            "always measured in a separate short timing_mode=ON pass."
        ),
    )
    parser.add_argument(
        "--no-timing-mode",
        "--no_timing_mode",
        action="store_true",
        help="Disable synchronized step breakdown timers.",
    )
    parser.add_argument(
        "--no-static-draft-cache",
        action="store_true",
        help="Disable pre-allocated StaticCache on draft model.",
    )
    parser.add_argument(
        "--compile-draft",
        "--compile_draft",
        action="store_true",
        default=None,
        help=(
            "torch.compile(mode='reduce-overhead') / CUDA graphs on the draft's StaticCache decode "
            "(default: ON from the CLI; --no-compile-draft to disable)."
        ),
    )
    parser.add_argument(
        "--draft-attn-impl",
        "--draft_attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help=(
            "Draft attention backend (default: sdpa with packed-GQA decode, CUDA-graph friendly). The target "
            "stays on --attn-impl (FlashAttention-2)."
        ),
    )
    parser.add_argument(
        "--no-compile-draft",
        action="store_true",
        help="Disable torch.compile(mode='reduce-overhead') on draft model.",
    )
    parser.add_argument(
        "--compile-target",
        "--compile_target",
        action="store_true",
        default=None,
        help="Enable torch.compile/CUDA graphs on target bucketed decode/verify.",
    )
    parser.add_argument(
        "--no-compile-target",
        "--no_compile_target",
        action="store_true",
        help="Disable compiled target runner.",
    )
    parser.add_argument(
        "--target-static-cache",
        "--target_static_cache",
        action="store_true",
        default=None,
        help="Use pre-allocated StaticCache for non-adaptive target baseline/speculative runs.",
    )
    parser.add_argument(
        "--no-target-static-cache",
        "--no_target_static_cache",
        action="store_true",
        help="Disable StaticCache on target model.",
    )
    parser.add_argument(
        "--compile-mode",
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode when compile flags are active (default: reduce-overhead).",
    )
    parser.add_argument(
        "--no-consistent-draft-cache",
        action="store_true",
        help="Allow fallback if prompt exceeds draft StaticCache max_position_embeddings.",
    )
    parser.add_argument(
        "--no-draft-error-on-overflow",
        action="store_true",
        help="Do not raise ContextLengthOverflowError on draft position overflow.",
    )
    parser.add_argument(
        "--no-round-static-8",
        action="store_true",
        help="Disable rounding StaticCache max_cache_len up to a multiple of 8.",
    )
    parser.add_argument(
        "--share-draft-keep-set",
        "--share_draft_keep_set",
        action="store_true",
        default=None,
        help="Share target AdaptiveKV keep-set indices with draft cache during synchronized eviction.",
    )
    parser.add_argument(
        "--no-share-draft-keep-set",
        "--no_share_draft_keep_set",
        action="store_true",
        help="Disable sharing target AdaptiveKV keep-set with draft.",
    )
    parser.add_argument(
        "--draft-cache",
        "--draft_cache",
        type=str,
        choices=["full", "shared_keepset"],
        default=None,
        help=(
            "Draft KV cache in Proposed (target compressed): 'shared_keepset' (default) = the draft keeps the "
            "same token positions as the target (~25%% of the context, StaticCache + CUDA graphs kept); "
            "'full' = draft keeps the whole context. Spec-only always uses 'full'."
        ),
    )
    parser.add_argument(
        "--compress-min-len",
        "--compress_min_len",
        type=int,
        default=None,
        help=(
            "Only compress the KV cache when prompt + max_new_tokens (PPL: prefix + continuation) is at least this many tokens (default 4096). Below it "
            "AdaptKV == Baseline and Proposed == Spec-only (same code path, no eviction, no PPL change)."
        ),
    )
    parser.add_argument(
        "--no-logits-to-keep",
        action="store_true",
        help="Disable logits_to_keep slicing on prefill, decode, verify, and PPL prefix chunks.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy verification, >0.0 for rejection sampling).",
    )
    parser.add_argument(
        "--max-cache-size",
        type=int,
        default=256,
        help="Minimum/base KV cache token budget per layer before dynamic pruning/quantization.",
    )
    parser.add_argument(
        "--cache-budget-ratio",
        type=float,
        default=None,
        help="Optional proportional KV budget ratio (e.g. 0.25 = retain 25%% of context length on long prompts).",
    )
    parser.add_argument(
        "--heavy-hitter-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of mid-cache budget kept in FP16. Default 1.0 = warm-tier INT4/INT8 OFF (pure pruning); "
            "<1.0 enables mixed INT8-K/INT4-V warm tokens (ablation only)."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Chunk size for long-prompt prefill to prevent OOM on 4K-32K contexts.",
    )
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=4,
        help="Number of StreamingLLM attention sink tokens preserved in FP16.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=64,
        help="Number of local recent tokens preserved in FP16.",
    )
    parser.add_argument(
        "--quant-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Bit-width (4 or 8) for quantizing medium-importance KV cache Values.",
    )
    parser.add_argument(
        "--key-quant-bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Bit-width (4 or 8) for per-channel quantization of Warm KV cache Keys.",
    )
    parser.add_argument(
        "--no-speculative",
        action="store_true",
        help="Disable speculative decoding in 'generate'/'interactive' mode.",
    )
    parser.add_argument(
        "--no-adaptive-cache",
        action="store_true",
        help="Disable adaptive KV-cache compression in 'generate'/'interactive' mode.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain how combining Adaptive KV-Cache Compression and Speculative Decoding "
            "enables sustainable, energy-efficient LLM inference on consumer hardware:\n"
        ),
        help="Input prompt for '--mode generate'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to store benchmark JSON, CSV, and plots.",
    )
    # ---- Real-task benchmarks (--mode tasks) ----
    parser.add_argument(
        "--suite",
        type=str,
        default="longbench,ruler",
        help="Comma-separated task suites for '--mode tasks': longbench, longbench_e, ruler (default: longbench,ruler).",
    )
    parser.add_argument(
        "--longbench-tasks",
        type=str,
        default=None,
        help="Comma-separated LongBench tasks or 'all' (default: 8-task subset covering all 6 categories).",
    )
    parser.add_argument(
        "--ruler-tasks",
        type=str,
        default=None,
        help="Comma-separated RULER tasks or 'all' (default: 8 tasks: 5 NIAH variants + vt + cwe + fwe).",
    )
    parser.add_argument(
        "--ruler-lengths",
        type=str,
        default="4096,8192,16384",
        help="Comma-separated RULER context lengths in tokens (prompt + generation), default 4096,8192,16384.",
    )
    parser.add_argument(
        "--task-samples",
        type=int,
        default=30,
        help="Samples per task (per length for RULER); deterministic seeded subset (default 30).",
    )
    parser.add_argument(
        "--task-max-len",
        type=int,
        default=None,
        help="LongBench prompt truncation length in tokens (official middle truncation; default 31500 = official 32K setting).",
    )
    parser.add_argument(
        "--task-methods",
        type=str,
        default="baseline,adaptive_kv,speculative_only,combined_proposed",
        help="Comma-separated methods to run in '--mode tasks'.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="Wrap prompts in the tokenizer chat template (auto = only for Instruct/Chat models, as LongBench does).",
    )
    parser.add_argument(
        "--longbench-dir",
        type=str,
        default=None,
        help="Optional local directory with LongBench <task>.jsonl files (default: download THUDM/LongBench data.zip).",
    )
    parser.add_argument("--task-seed", type=int, default=42, help="Seed for sample selection / RULER generation.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing task_predictions.jsonl instead of resuming from it.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with miniature in-memory Qwen2 models for instant testing without downloading weights.",
    )
    return parser.parse_args()


def _run_single_prompt(
    models,
    prompt: str,
    max_new_tokens: int,
    spec_cfg: SpeculativeConfig,
    cache_cfg: AdaptiveCacheConfig,
) -> None:
    monitor = EnergyMonitor()
    monitor.start()
    if spec_cfg.enabled:
        out = speculative_generate(
            draft_model=models.draft_model,
            target_model=models.target_model,
            tokenizer=models.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            spec_config=spec_cfg,
            target_cache_config=cache_cfg,
        )
    else:
        out = autoregressive_generate(
            model=models.target_model,
            tokenizer=models.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            cache_config=cache_cfg,
            temperature=spec_cfg.temperature,
            top_p=spec_cfg.top_p,
        )
    energy = monitor.stop(
        num_generated_tokens=out.num_generated_tokens,
        kv_compression_ratio=out.cache_stats.get("compression_ratio", 1.0),
        target_passes=out.target_forward_passes,
    )

    print("\n" + "=" * 88)
    print("GENERATED OUTPUT:")
    print("-" * 88)
    print(f"{prompt}{out.text}")
    print("-" * 88)
    print(
        f"Tokens Generated      : {out.num_generated_tokens} tokens in {out.total_time_s:.2f}s "
        f"({out.tokens_per_sec:.2f} tok/s)"
    )
    print(f"Latency               : TTFT = {out.ttft_ms:.2f} ms | TPOT = {out.tpot_ms:.2f} ms/tok")
    print(
        f"Forward Passes        : Target = {out.target_forward_passes} | Draft = {out.draft_forward_passes}"
    )
    if spec_cfg.enabled:
        print(
            f"Speculative Stats     : Acceptance Rate = {out.acceptance_rate * 100:.1f}% "
            f"({out.total_accepted_tokens}/{out.total_drafted_tokens} drafted) | "
            f"Mean Accepted/Step = {out.mean_accepted_per_step:.2f}"
        )
    cs = out.cache_stats
    print(
        f"Target KV-Cache       : {cs['actual_memory_mb']:.4f} MB vs {cs['uncompressed_memory_mb']:.4f} MB "
        f"uncompressed ({cs['compression_ratio']:.2f}x compression, {cs['memory_saved_pct']:.1f}% saved)"
    )
    print(
        f"KV Token Breakdown    : Active={cs['active_seq_len']} "
        f"(FP={cs['fp_tokens']}, Quantized={cs['quantized_tokens']}) | Pruned={cs['pruned_tokens']}"
    )
    print(
        f"Energy & Sustainability: {energy.energy_per_token_joules:.4f} J/token | "
        f"Avg Power = {energy.avg_power_watts:.1f} W | "
        f"Carbon = {energy.co2_grams_per_1k_tokens:.5f} gCO2e/1k tok ({energy.sensor_source})"
    )
    print("=" * 88 + "\n")


def _resolve_bool_flag(enable_flag, disable_flag, default: bool) -> bool:
    if enable_flag is True:
        return True
    if disable_flag is True:
        return False
    return default


def main() -> int:
    args = parse_args()

    mode = "sweep" if args.context_sweep else args.mode

    target_quant = args.target_quant
    if target_quant is None:
        target_quant = "bnb4" if args.load_in_4bit else ModelConfig().target_quant

    default_spec = SpeculativeConfig()
    default_cache = AdaptiveCacheConfig()
    default_bench = BenchmarkConfig()

    model_cfg = ModelConfig(
        draft_model_name=args.draft_model,
        target_model_name=args.target_model,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_impl,
        draft_attn_implementation=args.draft_attn_impl,
        target_quant=target_quant,
        awq_backend=args.awq_backend,
        load_target_in_4bit=args.load_in_4bit,
        load_target_in_8bit=args.load_in_8bit,
    )
    budget_ratio = args.cache_budget_ratio
    if budget_ratio is None and mode in ("stages", "sweep", "tasks"):
        budget_ratio = 0.25

    cache_cfg = AdaptiveCacheConfig(
        enabled=not args.no_adaptive_cache,
        max_cache_size=args.max_cache_size,
        cache_budget_ratio=budget_ratio,
        num_sink_tokens=args.sink_tokens,
        recent_window_size=args.recent_window,
        heavy_hitter_ratio=args.heavy_hitter_ratio,
        quant_bits=args.quant_bits,
        key_quant_bits=args.key_quant_bits,
        prefill_chunk_size=args.prefill_chunk_size,
        logits_to_keep=False if args.no_logits_to_keep else default_cache.logits_to_keep,
        compress_min_len=(
            int(args.compress_min_len) if args.compress_min_len is not None else default_cache.compress_min_len
        ),
    )
    spec_cfg = SpeculativeConfig(
        enabled=not args.no_speculative,
        gamma=args.gamma,
        adaptive_gamma=bool(args.adaptive_gamma and not args.no_adaptive_gamma),
        use_static_draft_cache=not args.no_static_draft_cache,
        # Draft decode is launch-overhead bound (0.5B model): CUDA graphs ON by default for CLI runs.
        compile_draft=_resolve_bool_flag(args.compile_draft, args.no_compile_draft, True),
        compile_target=_resolve_bool_flag(args.compile_target, args.no_compile_target, default_spec.compile_target),
        target_use_static_cache=_resolve_bool_flag(
            args.target_static_cache,
            args.no_target_static_cache,
            default_spec.target_use_static_cache,
        ),
        compile_mode=args.compile_mode,
        consistent_draft_cache=not args.no_consistent_draft_cache,
        draft_error_on_overflow=not args.no_draft_error_on_overflow,
        round_static_cache_to_8=not args.no_round_static_8,
        draft_cache=args.draft_cache if args.draft_cache is not None else default_spec.draft_cache,
        share_draft_keep_set=_resolve_bool_flag(
            args.share_draft_keep_set,
            args.no_share_draft_keep_set,
            default_spec.share_draft_keep_set,
        ),
        timing_mode=_resolve_bool_flag(args.timing_mode, args.no_timing_mode, default_spec.timing_mode),
        temperature=args.temperature,
    )

    if args.mock:
        print("[Main] Initializing miniature Qwen2.5 mock pair (Draft + Target) for verification...")
        models = create_tiny_qwen_pair_for_testing()
        # Use smaller cache thresholds in mock mode so pruning & quantization trigger within short runs
        cache_cfg.max_cache_size = 32
        cache_cfg.num_sink_tokens = 4
        cache_cfg.recent_window_size = 10
        if args.compress_min_len is None:
            # Mock prompts are a few hundred tokens: scale the threshold like the other mock budgets.
            cache_cfg.compress_min_len = 0
    else:
        need_draft = (mode in ("stages", "benchmark", "sweep", "tasks")) or spec_cfg.enabled
        try:
            models = load_model_pair(model_cfg, load_draft=need_draft)
        except FlashAttentionUnavailableError as exc:
            print(f"\n[Main] ERROR: {exc}\n", file=sys.stderr)
            return 1

    print_model_runtime_info(models)
    print(
        f"[RunInfo] target_quant={getattr(models, 'target_quant', target_quant)} | "
        f"timing_mode={'ON' if spec_cfg.timing_mode else 'OFF'} for throughput runs "
        "(d/v split from a separate timing_mode=ON pass) | "
        f"compress_min_len={cache_cfg.compress_min_len} | draft_cache(Proposed)={spec_cfg.resolved_draft_cache()} | "
        f"warm-tier INT quant={'OFF (pure pruning)' if cache_cfg.heavy_hitter_ratio >= 1.0 else 'ON (heavy_hitter_ratio=' + str(cache_cfg.heavy_hitter_ratio) + ')'}"
    )
    if models.draft_model is not None:
        print(
            f"[DraftInfo] attn={getattr(models.draft_model.config, '_attn_implementation', '?')} | "
            f"static_cache={'ON (capacity bucketed, multiple of 8)' if spec_cfg.use_static_draft_cache else 'OFF'} | "
            f"compile={'ON mode=' + spec_cfg.compile_mode + (' (CUDA graphs)' if spec_cfg.compile_mode == 'reduce-overhead' and models.device.type == 'cuda' else '') if spec_cfg.compile_draft else 'OFF (eager)'} | "
            "per-row draft path + ms/token are printed in the results; compile status/graph breaks at the end."
        )

    if mode == "stages":
        sweep_lengths = [
            int(tok.strip()) for tok in args.sweep_lengths.split(",") if tok.strip()
        ]
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        batch_sizes = (
            [int(tok.strip()) for tok in args.batch_sizes.split(",") if tok.strip()]
            if args.batch_sizes
            else (list(default_bench.batch_sweep_sizes) if args.batch_sweep else None)
        )
        batch_lengths = [
            int(tok.strip()) for tok in args.batch_sweep_lengths.split(",") if tok.strip()
        ]
        default_sweep_tokens = default_bench.sweep_max_new_tokens
        sweep_new_tokens = (
            args.max_new_tokens if args.max_new_tokens is not None else default_sweep_tokens
        )
        bench_cfg = BenchmarkConfig(
            sweep_lengths=sweep_lengths,
            stages=stages,
            sweep_max_new_tokens=sweep_new_tokens if not args.mock else min(sweep_new_tokens, 32),
            num_repeats=args.num_repeats,
            num_eval_docs=args.num_eval_docs,
            ppl_dataset=args.ppl_dataset,
            eval_continuation_tokens=args.eval_continuation_tokens,
            output_dir=args.output_dir,
        )
        run_stage_sweep(
            models=models,
            lengths=sweep_lengths,
            stages=stages,
            bench_config=bench_cfg,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
            evaluate_ppl=not args.no_ppl,
            batch_sizes=batch_sizes,
            batch_lengths=batch_lengths if batch_sizes else None,
        )
    elif mode == "benchmark":
        target_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 128
        bench_cfg = BenchmarkConfig(
            max_new_tokens=target_new_tokens if not args.mock else min(target_new_tokens, 40),
            output_dir=args.output_dir,
        )
        run_ablation_benchmark(
            models=models,
            bench_config=bench_cfg,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
        )
    elif mode == "sweep":
        sweep_lengths = [
            int(tok.strip()) for tok in args.sweep_lengths.split(",") if tok.strip()
        ]
        sweep_gammas = (
            [int(tok.strip()) for tok in args.sweep_gammas.split(",") if tok.strip()]
            if args.sweep_gammas
            else None
        )
        batch_sizes = (
            [int(tok.strip()) for tok in args.batch_sizes.split(",") if tok.strip()]
            if args.batch_sizes
            else (list(default_bench.batch_sweep_sizes) if args.batch_sweep else None)
        )
        batch_lengths = [
            int(tok.strip()) for tok in args.batch_sweep_lengths.split(",") if tok.strip()
        ]
        default_sweep_tokens = default_bench.sweep_max_new_tokens
        sweep_new_tokens = (
            args.max_new_tokens if args.max_new_tokens is not None else default_sweep_tokens
        )
        bench_cfg = BenchmarkConfig(
            sweep_lengths=sweep_lengths,
            sweep_max_new_tokens=sweep_new_tokens if not args.mock else min(sweep_new_tokens, 32),
            num_repeats=args.num_repeats,
            num_eval_docs=args.num_eval_docs,
            ppl_dataset=args.ppl_dataset,
            eval_continuation_tokens=args.eval_continuation_tokens,
            output_dir=args.output_dir,
        )
        run_context_sweep(
            models=models,
            lengths=sweep_lengths,
            bench_config=bench_cfg,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
            evaluate_ppl=not args.no_ppl,
            gammas=sweep_gammas,
            batch_sizes=batch_sizes,
            batch_lengths=batch_lengths if batch_sizes else None,
            profile_step=args.profile_step,
        )
    elif mode == "tasks":
        from src.benchmark.tasks.longbench import DEFAULT_LONGBENCH_TASKS, ALL_LONGBENCH_TASKS, LONGBENCH_E_TASKS
        from src.benchmark.tasks.ruler import DEFAULT_RULER_TASKS, ALL_RULER_TASKS

        def _csv(val, default, all_vals):
            if val is None:
                return list(default)
            if val.strip().lower() == "all":
                return list(all_vals)
            return [t.strip() for t in val.split(",") if t.strip()]

        suites = [s.strip().lower() for s in args.suite.split(",") if s.strip()]
        lb_all = LONGBENCH_E_TASKS if "longbench_e" in suites else ALL_LONGBENCH_TASKS
        lb_default = [t for t in DEFAULT_LONGBENCH_TASKS if t in lb_all]
        out_dir = args.output_dir if args.output_dir != "results" else "results/tasks"
        run_task_benchmark(
            models=models,
            suites=suites,
            longbench_tasks=_csv(args.longbench_tasks, lb_default, lb_all),
            ruler_tasks=_csv(args.ruler_tasks, DEFAULT_RULER_TASKS, ALL_RULER_TASKS),
            ruler_lengths=[int(x) for x in args.ruler_lengths.split(",") if x.strip()],
            num_samples=args.task_samples,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
            methods=[m.strip() for m in args.task_methods.split(",") if m.strip()],
            max_len=args.task_max_len or 31500,
            chat_template=args.chat_template,
            model_names=(
                "mock-tiny-qwen2" if args.mock else args.target_model,
                "" if models.draft_model is None else ("mock-tiny-qwen2-draft" if args.mock else args.draft_model),
            ),
            longbench_dir=args.longbench_dir,
            output_dir=out_dir,
            seed=args.task_seed,
            resume=not args.no_resume,
        )
    elif mode == "generate":
        gen_tokens = args.max_new_tokens if args.max_new_tokens is not None else 128
        _run_single_prompt(models, args.prompt, gen_tokens, spec_cfg, cache_cfg)
    elif mode == "interactive":
        gen_tokens = args.max_new_tokens if args.max_new_tokens is not None else 128
        print("\nInteractive Mode (Draft: Qwen2.5-0.5B | Target: Qwen2.5-7B). Type 'exit' to quit.")
        while True:
            try:
                user_prompt = input("\nPrompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_prompt or user_prompt.lower() in ("exit", "quit"):
                break
            _run_single_prompt(models, user_prompt, gen_tokens, spec_cfg, cache_cfg)

    if models.draft_model is not None:
        st = get_compile_status(models.draft_model)
        dyn = dynamo_counters_summary()
        print(
            f"[DraftInfo] final draft compile status: {st.get('state')} (mode={st.get('mode')}, "
            f"graphs={st.get('graphs')}{', reason=' + st['reason'] if st.get('reason') else ''}) | "
            f"torch.compile counters: {dyn}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
