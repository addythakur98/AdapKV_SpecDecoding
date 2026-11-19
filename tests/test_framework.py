"""
Unit and Integration Tests for Adaptive KV-Cache Compression & Speculative Decoding.
"""

import json
import os
import tempfile
import torch
import pytest

from config import AdaptiveCacheConfig, BenchmarkConfig, ModelConfig, SpeculativeConfig
from src.benchmark.metrics import (
    build_long_context_prompt,
    compute_ngram_f1,
    evaluate_perplexity,
    load_long_context_eval_text,
    load_long_context_eval_texts,
)
from src.benchmark.runner import run_ablation_benchmark, run_context_sweep, run_stage_sweep
from src.kv_cache.adaptive_cache import AdaptiveKVCache
from src.kv_cache.quantizer import concat_quantized_blocks, dequantize_tensor, quantize_tensor
from src.models.loader import (
    check_vram_fit_for_sequence,
    create_tiny_qwen_pair_for_testing,
    resolve_attn_implementation,
    resolve_model_pair_for_quant,
    set_attn_implementation,
)
from src.speculative.baseline_decoder import autoregressive_generate
from src.speculative.speculative_decoder import speculative_generate
from src.speculative.static_runner import (
    ContextLengthOverflowError,
    round_up_to_multiple,
    sync_perf_counter,
)


def test_int4_and_int8_kv_quantizer():
    """Verifies packed INT4 and INT8 quantization memory reduction and reconstruction."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 32, 64, dtype=torch.float16)
    orig_bytes = x.numel() * x.element_size()

    # Test 4-bit packed quantization
    q4 = quantize_tensor(x, bits=4)
    assert q4.seq_length == 32
    assert q4.memory_bytes() < orig_bytes * 0.35  # >3x memory reduction including scale/zp
    x_rec4 = dequantize_tensor(q4)
    assert x_rec4.shape == x.shape
    assert x_rec4.dtype == x.dtype
    rel_err4 = torch.norm(x.float() - x_rec4.float()) / torch.norm(x.float())
    assert rel_err4 < 0.12

    # Test 8-bit quantization
    q8 = quantize_tensor(x, bits=8)
    assert q8.memory_bytes() < orig_bytes * 0.60
    x_rec8 = dequantize_tensor(q8)
    rel_err8 = torch.norm(x.float() - x_rec8.float()) / torch.norm(x.float())
    assert rel_err8 < 0.015


def test_adaptive_kv_cache_pruning_quantization_and_crop():
    """Verifies tiered pruning, INT4 quantization, and speculative rollback (`crop`)."""
    cfg = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=24,
        num_sink_tokens=4,
        recent_window_size=8,
        heavy_hitter_ratio=0.5,  # 6 FP heavy hitters + 6 INT4 quantized tokens
        quant_bits=4,
    )
    cache = AdaptiveKVCache(config=cfg)

    # Simulate 2 transformer layers receiving 40 tokens in chunks
    for step in range(8):
        for layer_idx in range(2):
            k = torch.randn(1, 2, 5, 32)
            v = torch.randn(1, 2, 5, 32)
            full_k, full_v = cache.update(k, v, layer_idx)
            assert full_k.shape == full_v.shape
        cache.compress_step()
    cache.compress_step(force=True)

    stats = cache.get_memory_stats()
    assert stats["total_tokens_seen"] == 40
    assert stats["active_seq_len"] <= 24
    assert cache.get_active_kv_length() == stats["active_seq_len"]
    assert cache.get_seq_length() == 40  # Monotonic RoPE sequence length preserved
    assert stats["quantized_tokens"] > 0
    assert stats["pruned_tokens"] > 0
    assert stats["fp_tokens"] + stats["quantized_tokens"] + stats["pruned_tokens"] == 40
    assert stats["compression_ratio"] > 1.5

    # Simulate speculative step adding 5 draft tokens, then rolling back 3 rejected tokens
    pre_len = cache.get_seq_length()
    expected_warm_len = stats["quantized_tokens"]
    expected_fp_len = stats["fp_tokens"]
    for layer_idx in range(2):
        k = torch.randn(1, 2, 5, 32)
        v = torch.randn(1, 2, 5, 32)
        out_k, out_v = cache.update(k, v, layer_idx)
        assert out_k.shape[-2] == expected_warm_len + expected_fp_len + 5
        assert out_v.shape[-2] == expected_warm_len + expected_fp_len + 5
    assert cache.get_seq_length() == pre_len + 5

    cache.crop(pre_len + 2)
    assert cache.get_seq_length() == pre_len + 2
    assert cache.get_active_kv_length() == expected_warm_len + expected_fp_len + 2


def test_greedy_speculative_matches_autoregressive_exactly():
    """
    Proves that when Adaptive KV-Cache compression is disabled and temperature=0,
    Speculative Decoding (with rollback) generates the EXACT same token sequence
    as standard Autoregressive Decoding on the target model.
    """
    models = create_tiny_qwen_pair_for_testing()
    prompt = "Adaptive KV cache compression and speculative decoding optimize inference"

    uncompressed_cfg = AdaptiveCacheConfig(enabled=False)
    spec_cfg = SpeculativeConfig(
        enabled=True,
        gamma=4,
        adaptive_gamma=False,
        temperature=0.0,
    )

    base_out = autoregressive_generate(
        model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        cache_config=uncompressed_cfg,
        temperature=0.0,
        stop_on_eos=False,
    )

    spec_out = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        spec_config=spec_cfg,
        target_cache_config=uncompressed_cfg,
        stop_on_eos=False,
    )

    assert base_out.num_generated_tokens == 20
    assert spec_out.num_generated_tokens == 20
    assert torch.equal(base_out.output_ids, spec_out.output_ids), (
        f"Mismatch between greedy baseline and speculative decoding!\n"
        f"Baseline:    {base_out.output_ids.tolist()}\n"
        f"Speculative: {spec_out.output_ids.tolist()}"
    )


def test_combined_speculative_and_adaptive_cache_end_to_end():
    """Tests the full combined framework (Speculative Decoding + Adaptive KV-Cache) and ablation runner."""
    models = create_tiny_qwen_pair_for_testing()
    cache_cfg = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=20,
        num_sink_tokens=2,
        recent_window_size=6,
        heavy_hitter_ratio=0.5,
        quant_bits=4,
    )
    spec_cfg = SpeculativeConfig(
        enabled=True,
        gamma=4,
        adaptive_gamma=True,
        temperature=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        bench_cfg = BenchmarkConfig(
            max_new_tokens=24,
            num_warmup_runs=1,
            output_dir=tmpdir,
            prompts=["Explain how speculative decoding accelerates large language models:"],
            perplexity_texts=["Large language models require efficient key value cache management."],
        )
        report = run_ablation_benchmark(
            models=models,
            bench_config=bench_cfg,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
        )

        assert len(report["ablation_results"]) == 4
        assert os.path.exists(os.path.join(tmpdir, "benchmark_results.json"))
        assert os.path.exists(os.path.join(tmpdir, "benchmark_results.csv"))
        assert os.path.exists(report["plot_path"])

        combined_row = report["ablation_results"][3]
        assert combined_row["compression_ratio"] > 1.0
        assert combined_row["kv_memory_saved_pct"] > 0.0


def test_warm_tier_sensitivity_and_rope_causal_mask_after_compression():
    """
    Verifies two critical correctness properties of the compressed KV cache:
    1. Attention output genuinely reads Warm-Tier INT4 tokens (perturbing Warm INT4 changes logits).
    2. Monotonic RoPE positions & causal mask offsets remain valid for multi-token queries (q_len > 1)
       after KV compression (used by speculative verification and chunked prefill).
    """
    models = create_tiny_qwen_pair_for_testing()
    target = models.target_model
    cfg = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=24,
        num_sink_tokens=4,
        recent_window_size=8,
        heavy_hitter_ratio=0.5,  # 6 FP heavy-hitters + 6 INT4 warm tokens -> 18 FP + 6 INT4
        quant_bits=4,
        prefill_chunk_size=64,
    )
    cache = AdaptiveKVCache(config=cfg)

    torch.manual_seed(7)
    prompt_ids = torch.randint(10, 200, (1, 80), device=models.device)
    with torch.no_grad():
        _ = target(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache.compress_step(force=True)

    stats = cache.get_memory_stats()
    assert stats["total_tokens_seen"] == 80
    assert stats["quantized_tokens"] == 6
    assert cache.get_seq_length() == 80
    assert cache.get_active_kv_length() == 24

    # Verify mask offset invariant before feeding a multi-token verification query (q_len = 4)
    q_offset = cache.get_query_offset(0)
    kv_len, kv_offset = cache.get_mask_sizes(query_length=4, layer_idx=0)
    assert q_offset == 80
    assert kv_len == 24 + 4
    assert kv_offset + cache.get_active_kv_length(0) == q_offset

    # Run a q_len > 1 verification forward pass (cloning cache state indirectly via logits diff)
    probe_ids = torch.randint(10, 200, (1, 4), device=models.device)
    with torch.no_grad():
        logits_before = target(input_ids=probe_ids, past_key_values=cache, use_cache=True).logits.clone()
    cache.crop(80)

    # Zero out Warm INT4 blocks across all layers and confirm logits change -> attention reads INT4!
    for idx in range(len(cache.quant_value_blocks)):
        if cache.quant_value_blocks[idx] is not None:
            cache.quant_value_blocks[idx].q_data.zero_()
            cache.quant_value_blocks[idx].scale.zero_()
            cache.quant_value_blocks[idx].zero_point.zero_()
    with torch.no_grad():
        logits_after = target(input_ids=probe_ids, past_key_values=cache, use_cache=True).logits.clone()

    assert not torch.allclose(logits_before, logits_after, atol=1e-5), (
        "Logits did not change after zeroing Warm INT4 blocks; attention is ignoring Warm Tier!"
    )


def test_proportional_budget_and_context_sweep_mock():
    """Tests proportional cache scaling and the context-length sweep pipeline end-to-end."""
    models = create_tiny_qwen_pair_for_testing()
    cache_cfg = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=32,
        cache_budget_ratio=0.25,
        num_sink_tokens=4,
        recent_window_size=16,
        heavy_hitter_ratio=0.5,
        quant_bits=4,
        prefill_chunk_size=128,
    )
    cache = AdaptiveKVCache(config=cache_cfg)
    eff_max, _, _, _, _ = cache._effective_budgets(1024)
    assert eff_max == 256

    prompt_512, passkey = build_long_context_prompt(models.tokenizer, target_tokens=512, passkey="73914")
    tok_len = int(models.tokenizer(prompt_512, return_tensors="pt")["input_ids"].shape[1])
    assert 480 <= tok_len <= 560
    assert passkey == "73914"

    with tempfile.TemporaryDirectory() as tmpdir:
        bench_cfg = BenchmarkConfig(
            num_warmup_runs=0,
            sweep_lengths=[128, 256, 512],
            sweep_max_new_tokens=16,
            output_dir=tmpdir,
        )
        spec_cfg = SpeculativeConfig(enabled=True, gamma=4, adaptive_gamma=True, temperature=0.0)
        report = run_context_sweep(
            models=models,
            lengths=[128, 256, 512],
            bench_config=bench_cfg,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
            evaluate_ppl=True,
        )

        assert len(report["sweep_results"]) == 12
        assert "crossover_analysis" in report
        assert os.path.exists(os.path.join(tmpdir, "context_sweep.json"))
        assert os.path.exists(os.path.join(tmpdir, "context_sweep.csv"))
        assert os.path.exists(report["plot_path"])
        first_spec = next(r for r in report["sweep_results"] if r["id"] == "combined_proposed")
        assert "decode_tokens_per_sec" in first_spec
        assert "draft_time_ms" in first_spec
        assert "verify_time_ms" in first_spec
        assert "cache_overhead_ms" in first_spec
        assert first_spec["draft_kv_actual_mb"] > 0.0


def test_per_channel_key_quantization_under_qwen_outliers():
    """
    Verifies KIVI-style per-channel Key quantization prevents softmax collapse when Keys
    carry large fixed channel outliers (as in Qwen2.5 `k_proj.bias`).
    """
    torch.manual_seed(7)
    B, H, T, D = 1, 4, 256, 128
    q = torch.randn(B, H, 1, D, dtype=torch.float16)
    k = torch.randn(B, H, T, D, dtype=torch.float16) * 0.5
    v = torch.randn(B, H, T, D, dtype=torch.float16) * 0.5
    # Inject strong fixed channel-wise outliers on Keys (channels 0, 17, 91)
    k[:, :, :, 0] += 40.0
    k[:, :, :, 17] -= 30.0
    k[:, :, :, 91] += 25.0

    # Baseline FP16 attention output & distribution
    scale = 1.0 / (D ** 0.5)
    logits_fp = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    probs_fp = torch.softmax(logits_fp, dim=-1)
    out_fp = torch.matmul(probs_fp, v.float())

    # Broken per-token INT4 on Keys explodes softmax KL
    k_tok4 = dequantize_tensor(quantize_tensor(k, bits=4, axis="per_token"))
    logits_bad = torch.matmul(q.float(), k_tok4.float().transpose(-2, -1)) * scale
    probs_bad = torch.softmax(logits_bad, dim=-1)
    kl_bad = float((probs_fp * (torch.log(probs_fp + 1e-9) - torch.log(probs_bad + 1e-9))).sum() / (B * H))
    assert kl_bad > 0.5

    # Mixed K-per-channel INT8 + V-per-token INT4 preserves attention distribution cleanly
    k_blk_a = quantize_tensor(k[:, :, :128, :], bits=8, axis="per_channel")
    k_blk_b = quantize_tensor(k[:, :, 128:, :], bits=8, axis="per_channel")
    k_cat = concat_quantized_blocks(k_blk_a, k_blk_b, max_seq_len=256)
    k_ch8 = dequantize_tensor(k_cat)
    v_tok4 = dequantize_tensor(quantize_tensor(v, bits=4, axis="per_token"))

    logits_good = torch.matmul(q.float(), k_ch8.float().transpose(-2, -1)) * scale
    probs_good = torch.softmax(logits_good, dim=-1)
    out_good = torch.matmul(probs_good, v_tok4.float())

    kl_good = float((probs_fp * (torch.log(probs_fp + 1e-9) - torch.log(probs_good + 1e-9))).sum() / (B * H))
    rel_out_err = float(torch.norm(out_fp - out_good) / torch.norm(out_fp))
    assert kl_good < 0.01
    assert rel_out_err < 0.15


def test_preallocated_buffer_and_nonrepeating_prompt():
    """Verifies O(1) preallocated FP buffer crop/update and non-repeating sweep prompts."""
    models = create_tiny_qwen_pair_for_testing()
    prompt_a, _ = build_long_context_prompt(models.tokenizer, target_tokens=768, passkey="73914")
    tok_count = int(models.tokenizer(prompt_a, return_tensors="pt")["input_ids"].shape[1])
    assert tok_count <= 768
    lines = [
        ln.strip()
        for ln in prompt_a.splitlines()
        if ln.strip() and not ln.startswith(("Question:", "Answer:"))
    ]
    assert len(lines) >= 6
    assert len(set(lines)) == len(lines)

    cache = AdaptiveKVCache(AdaptiveCacheConfig(enabled=True, compress_min_len=0, max_cache_size=256))
    k_init = torch.randn(1, 2, 16, 16, dtype=torch.float16)
    v_init = torch.randn(1, 2, 16, 16, dtype=torch.float16)
    cache.update(k_init, v_init, layer_idx=0)
    buf_ptr = cache._k_buf[0].data_ptr()
    for step in range(8):
        k_one = torch.randn(1, 2, 1, 16, dtype=torch.float16)
        v_one = torch.randn(1, 2, 1, 16, dtype=torch.float16)
        k_out, _ = cache.update(k_one, v_one, layer_idx=0)
        assert k_out.shape[2] == 16 + step + 1
    assert cache._k_buf[0].data_ptr() == buf_ptr  # zero reallocations inside initial capacity
    cache.crop(18)
    assert cache._fp_len[0] == 18
    assert cache._k_buf[0].data_ptr() == buf_ptr


def test_heldout_ppl_token_budget_repeats_gamma_and_batch():
    """
    Verifies held-out continuation PPL scoring, token_compression_ratio vs byte compression,
    multi-repeat std telemetry, gamma sweep, theoretical speedup sanity metrics, batch-size
    scaling, and torch.profiler step diagnostics.
    """
    models = create_tiny_qwen_pair_for_testing()
    cache_cfg = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=32,
        cache_budget_ratio=0.25,
        num_sink_tokens=4,
        recent_window_size=16,
        heavy_hitter_ratio=0.5,
        quant_bits=4,
        key_quant_bits=8,
        prefill_chunk_size=128,
    )

    eval_text, src = load_long_context_eval_text(
        models.tokenizer, prefix_tokens=256, continuation_tokens=64, dataset_name=None
    )
    assert src == "local:heldout"
    res_ppl = evaluate_perplexity(
        models.target_model,
        models.tokenizer,
        [eval_text],
        cache_config=cache_cfg,
        continuation_tokens=64,
    )
    assert res_ppl["evaluated_tokens"] == 64

    with tempfile.TemporaryDirectory() as tmpdir:
        bench_cfg = BenchmarkConfig(
            num_warmup_runs=0,
            num_repeats=2,
            sweep_lengths=[192],
            sweep_max_new_tokens=16,
            eval_continuation_tokens=48,
            output_dir=tmpdir,
        )
        spec_cfg = SpeculativeConfig(enabled=True, gamma=4, adaptive_gamma=False, temperature=0.0)
        report = run_context_sweep(
            models=models,
            lengths=[192],
            bench_config=bench_cfg,
            cache_config=cache_cfg,
            spec_config=spec_cfg,
            evaluate_ppl=True,
            gammas=[2, 4],
            batch_sizes=[1, 2],
            profile_step=True,
        )

        # 2 non-speculative modes + 2 speculative modes * 2 gammas = 6 rows
        assert len(report["sweep_results"]) == 6
        prop_rows = [r for r in report["sweep_results"] if r["id"] == "combined_proposed"]
        assert {r["gamma"] for r in prop_rows} == {2, 4}
        r0 = prop_rows[0]
        assert "theoretical_speedup" in r0 and r0["theoretical_speedup"] >= 0.0
        assert "cost_ratio_c" in r0
        assert "tpot_ms_std" in r0
        assert "rollback_time_ms" in r0
        assert 3.5 <= r0["token_compression_ratio"] <= 4.5
        assert len(report["batch_size_results"]) == 4  # 2 batch sizes * (baseline + adaptive_kv)
        assert len(report["profiler_traces"]) >= 1
        assert len(report["profiler_traces"][0]["top_ops"]) > 0


def test_query_hooks_static_draft_cache_exact_budget_and_segments():
    """
    Verifies:
    1. SnapKV-style `Q_obs @ K^T` query hooks + default `heavy_hitter_ratio=1.0` retain exact 25% (4.00x).
    2. `StaticCache` on the draft model generates bit-exact greedy tokens matching dynamic draft cache.
    3. Segment-based `concat_quantized_blocks(..., axis='per_channel')` never re-quantizes older chunks.
    """
    models = create_tiny_qwen_pair_for_testing()
    prompt, _ = build_long_context_prompt(models.tokenizer, target_tokens=256, passkey="73914")
    tok_ids = models.tokenizer(prompt, return_tensors="pt")["input_ids"]
    assert tok_ids.shape[1] <= 256

    # 1. Exact 25% (4.00x) retention + query hook execution
    cfg_fp = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=32,
        cache_budget_ratio=0.25,
        num_sink_tokens=4,
        recent_window_size=16,
        heavy_hitter_ratio=1.0,
        obs_window_size=8,
        prefill_chunk_size=128,
    )
    cache = AdaptiveKVCache(config=cfg_fp)
    cache.register_query_hooks(models.target_model)
    try:
        prompt_chunk_len = tok_ids.shape[1]
        with torch.no_grad():
            _ = models.target_model(input_ids=tok_ids, past_key_values=cache, use_cache=True)
        assert cache._obs_inputs[0] is not None
        cache.compress_step(force=True)
    finally:
        cache.remove_query_hooks()

    stats = cache.get_memory_stats()
    assert stats["quantized_tokens"] == 0
    assert stats["active_seq_len"] == int(round(prompt_chunk_len * 0.25))
    assert abs(stats["compression_ratio"] - 4.0) < 0.05
    assert abs(stats["token_compression_ratio"] - 4.0) < 0.05

    # 2. StaticCache draft vs Adaptive/Dynamic draft greedy equivalence
    uncomp_cfg = AdaptiveCacheConfig(enabled=False, prefill_chunk_size=128)
    s_static = SpeculativeConfig(
        enabled=True, gamma=4, adaptive_gamma=False, use_static_draft_cache=True, compile_draft=False
    )
    s_dyn = SpeculativeConfig(
        enabled=True, gamma=4, adaptive_gamma=False, use_static_draft_cache=False, compile_draft=False
    )
    out_static = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt="Speculative decoding with StaticCache rollback preserves exact tokens",
        max_new_tokens=18,
        spec_config=s_static,
        target_cache_config=uncomp_cfg,
        stop_on_eos=False,
    )
    out_dyn = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt="Speculative decoding with StaticCache rollback preserves exact tokens",
        max_new_tokens=18,
        spec_config=s_dyn,
        target_cache_config=uncomp_cfg,
        stop_on_eos=False,
    )
    assert out_static.draft_cache_stats.get("cache_type") == "StaticCache"
    assert torch.equal(out_static.output_ids, out_dyn.output_ids)

    # 3. Segment-based per_channel concat leaves newer segment bit-identical
    torch.manual_seed(11)
    x1 = torch.randn(1, 2, 16, 32, dtype=torch.float16)
    x2 = torch.randn(1, 2, 16, 32, dtype=torch.float16)
    b1 = quantize_tensor(x1, bits=8, axis="per_channel")
    b2 = quantize_tensor(x2, bits=8, axis="per_channel")
    merged = concat_quantized_blocks(b1, b2, max_seq_len=24)
    assert merged.seq_length == 24
    assert torch.equal(dequantize_tensor(merged)[:, :, -16:, :], dequantize_tensor(b2))


def test_step1_timing_mode_flag_and_parity(monkeypatch):
    """Step 1: --timing_mode syncs CUDA before timer reads without changing outputs or acceptance."""
    models = create_tiny_qwen_pair_for_testing()
    uncomp_cfg = AdaptiveCacheConfig(enabled=False)
    prompt = "Synchronized timers must never alter greedy speculative decoding outputs."

    s_off = SpeculativeConfig(enabled=True, gamma=4, timing_mode=False, compile_draft=False)
    s_on = SpeculativeConfig(enabled=True, gamma=4, timing_mode=True, compile_draft=False)

    out_off = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=16,
        spec_config=s_off,
        target_cache_config=uncomp_cfg,
        stop_on_eos=False,
    )
    out_on = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=16,
        spec_config=s_on,
        target_cache_config=uncomp_cfg,
        stop_on_eos=False,
    )
    assert torch.equal(out_off.output_ids, out_on.output_ids)
    assert out_off.acceptance_rate == out_on.acceptance_rate
    assert out_on.draft_time_ms >= 0.0 and out_on.verify_time_ms >= 0.0

    # Verify sync_perf_counter invokes torch.cuda.synchronize only when timing_mode=True on CUDA
    sync_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev=None: sync_calls.append(dev))
    sync_perf_counter(torch.device("cuda:0"), timing_mode=False)
    assert len(sync_calls) == 0
    sync_perf_counter(torch.device("cuda:0"), timing_mode=True)
    assert len(sync_calls) == 1


def test_step2_flash_attn_fallback_and_adaptive_kv_fa2_contract(monkeypatch, capsys):
    """Step 2: flash_attention_2 is required on CUDA (raises, no sdpa fallback), and AdaptiveKVCache matches FA2 attributes."""
    import src.models.loader as loader_mod

    monkeypatch.setattr(loader_mod, "is_flash_attn_available", lambda: False)
    with pytest.raises(loader_mod.FlashAttentionUnavailableError, match="flash-attn"):
        resolve_attn_implementation("auto", torch.float16, "cuda")

    monkeypatch.setattr(loader_mod, "is_flash_attn_available", lambda: True)
    resolved_ok = resolve_attn_implementation("auto", torch.float16, "cuda")
    assert resolved_ok == "flash_attention_2"

    models = create_tiny_qwen_pair_for_testing()
    actual = set_attn_implementation(models, "auto")
    assert actual == "sdpa"  # CPU mock environment resolves to sdpa

    # Verify AdaptiveKVCache exposes FA2/Transformers v5 attributes and crop rollback semantics
    cache = AdaptiveKVCache(
        AdaptiveCacheConfig(enabled=True, compress_min_len=0, max_cache_size=32, num_sink_tokens=4, recent_window_size=12),
    )
    assert cache.is_sliding == []
    assert cache.is_linear == []
    assert cache.is_dummy == []
    assert cache.is_compileable is False

    for layer_idx in range(2):
        k = torch.randn(1, 2, 16, 16, dtype=torch.float32)
        v = torch.randn(1, 2, 16, 16, dtype=torch.float32)
        k_out, _ = cache.update(k, v, layer_idx=layer_idx)
        assert k_out.shape[2] == 16
    assert cache.get_seq_length() == 16
    cache.crop(10)
    assert cache.get_seq_length() == 10


def test_step3_quantization_resolution_and_base_instruct_guard():
    """Step 3: Default FP16 Base, AWQ Instruct pairing, bnb4 compatibility, and Base+Instruct mismatch guard."""
    assert resolve_model_pair_for_quant(ModelConfig()) == (
        "fp16",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-0.5B",
    )
    assert resolve_model_pair_for_quant(ModelConfig(target_quant="bnb4")) == (
        "bnb4",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-0.5B",
    )
    assert resolve_model_pair_for_quant(ModelConfig(target_quant="awq")) == (
        "awq",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "Qwen/Qwen2.5-0.5B-Instruct",
    )
    with pytest.raises(ValueError, match="Do not mix Base and Instruct"):
        resolve_model_pair_for_quant(
            ModelConfig(
                target_model_name="Qwen/Qwen2.5-7B-Instruct",
                draft_model_name="Qwen/Qwen2.5-0.5B",
            )
        )

    models = create_tiny_qwen_pair_for_testing()
    vram_info = check_vram_fit_for_sequence(
        models.target_model, models.draft_model, "fp16", seq_len=16384
    )
    assert isinstance(vram_info["fits"], bool)
    assert vram_info["estimated_total_gb"] > 0
    assert vram_info["kv_gb"] > 0


def test_step4_and_step5_static_alignment_overflow_and_shared_keepset():
    """Steps 4 & 5: StaticCache %8 rounding, target StaticCache parity, strict overflow error, and shared keep-set."""
    assert round_up_to_multiple(129, 8) == 136
    assert round_up_to_multiple(128, 8) == 128

    models = create_tiny_qwen_pair_for_testing()
    uncomp_cfg = AdaptiveCacheConfig(enabled=False)
    prompt = "Verifying target and draft static cache parity and %8 alignment on Qwen2."

    # Compile enabled falls back cleanly to eager on CPU/Windows without crashing
    s_static_all = SpeculativeConfig(
        enabled=True,
        gamma=4,
        use_static_draft_cache=True,
        target_use_static_cache=True,
        compile_draft=True,
        compile_target=True,
        round_static_cache_to_8=True,
        consistent_draft_cache=True,
        draft_error_on_overflow=True,
    )
    s_dyn_all = SpeculativeConfig(
        enabled=True,
        gamma=4,
        use_static_draft_cache=False,
        target_use_static_cache=False,
        compile_draft=False,
        compile_target=False,
    )
    out_static = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=16,
        spec_config=s_static_all,
        target_cache_config=uncomp_cfg,
        stop_on_eos=False,
    )
    out_dyn = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=prompt,
        max_new_tokens=16,
        spec_config=s_dyn_all,
        target_cache_config=uncomp_cfg,
        stop_on_eos=False,
    )
    assert torch.equal(out_static.output_ids, out_dyn.output_ids)
    assert out_static.draft_cache_stats.get("max_cache_len", 0) % 8 == 0
    assert out_static.cache_stats.get("cache_type") == "StaticCache"
    assert out_static.cache_stats.get("max_cache_len", 0) % 8 == 0

    # Strict overflow: artificially small draft max_position_embeddings raises ContextLengthOverflowError
    orig_max_pos = models.draft_model.config.max_position_embeddings
    try:
        models.draft_model.config.max_position_embeddings = 24
        long_prompt, _ = build_long_context_prompt(models.tokenizer, target_tokens=40)
        with pytest.raises(ContextLengthOverflowError):
            speculative_generate(
                draft_model=models.draft_model,
                target_model=models.target_model,
                tokenizer=models.tokenizer,
                prompt=long_prompt,
                max_new_tokens=8,
                spec_config=s_static_all,
                target_cache_config=uncomp_cfg,
                stop_on_eos=False,
            )
    finally:
        models.draft_model.config.max_position_embeddings = orig_max_pos

    # Step 5: share_draft_keep_set with AdaptiveKV compression
    adapt_cfg = AdaptiveCacheConfig(
        enabled=True,
        compress_min_len=0,
        max_cache_size=40,
        num_sink_tokens=4,
        recent_window_size=12,
        heavy_hitter_ratio=1.0,
        prefill_chunk_size=32,
        logits_to_keep=True,
    )
    s_shared = SpeculativeConfig(
        enabled=True,
        gamma=4,
        share_draft_keep_set=True,
        use_static_draft_cache=True,
        compile_draft=False,
    )
    long_p, _ = build_long_context_prompt(models.tokenizer, target_tokens=96)
    out_shared = speculative_generate(
        draft_model=models.draft_model,
        target_model=models.target_model,
        tokenizer=models.tokenizer,
        prompt=long_p,
        max_new_tokens=16,
        spec_config=s_shared,
        target_cache_config=adapt_cfg,
        stop_on_eos=False,
    )
    assert out_shared.num_generated_tokens == 16
    assert out_shared.cache_stats["pruned_tokens"] > 0


def test_step6_logits_to_keep_ppl_parity_and_oom_skip(monkeypatch):
    """Step 6: logits_to_keep preserves exact baseline decode & PPL while saving logit allocations."""
    models = create_tiny_qwen_pair_for_testing()
    prompt, _ = build_long_context_prompt(models.tokenizer, target_tokens=110)
    uncomp_no_ltk = AdaptiveCacheConfig(enabled=False, prefill_chunk_size=48, logits_to_keep=False)
    uncomp_ltk = AdaptiveCacheConfig(enabled=False, prefill_chunk_size=48, logits_to_keep=True)

    out_ref = autoregressive_generate(
        models.target_model,
        models.tokenizer,
        prompt=prompt,
        max_new_tokens=14,
        cache_config=uncomp_no_ltk,
        stop_on_eos=False,
        logits_to_keep=False,
    )
    out_fast = autoregressive_generate(
        models.target_model,
        models.tokenizer,
        prompt=prompt,
        max_new_tokens=14,
        cache_config=uncomp_ltk,
        stop_on_eos=False,
        logits_to_keep=True,
    )
    assert torch.equal(out_ref.output_ids, out_fast.output_ids)

    eval_text, _ = load_long_context_eval_text(
        models.tokenizer, prefix_tokens=130, continuation_tokens=50
    )
    ppl_ref = evaluate_perplexity(
        model=models.target_model,
        tokenizer=models.tokenizer,
        texts=[eval_text],
        cache_config=uncomp_no_ltk,
        chunk_size=48,
        continuation_tokens=48,
        logits_to_keep=False,
    )
    ppl_fast = evaluate_perplexity(
        model=models.target_model,
        tokenizer=models.tokenizer,
        texts=[eval_text],
        cache_config=uncomp_ltk,
        chunk_size=48,
        continuation_tokens=48,
        logits_to_keep=True,
    )
    assert ppl_ref["evaluated_tokens"] == ppl_fast["evaluated_tokens"] > 0
    assert abs(ppl_ref["perplexity"] - ppl_fast["perplexity"]) < 1e-4

    # Mock CUDA OOM inside speculative_generate on second context length to ensure run_stage_sweep skips without crashing
    import src.benchmark.runner as runner_mod

    orig_spec_gen = runner_mod.speculative_generate

    def flaky_spec_gen(*args, **kwargs):
        p = kwargs.get("prompt")
        if isinstance(p, str):
            tok_len = models.tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
            if tok_len >= 180:
                raise torch.cuda.OutOfMemoryError("Simulated OOM at >=180 tokens")
        return orig_spec_gen(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "speculative_generate", flaky_spec_gen)
    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_stage_sweep(
            models=models,
            lengths=[96, 192],
            stages=["S0", "S1"],
            bench_config=BenchmarkConfig(
                sweep_lengths=[96, 192],
                sweep_max_new_tokens=8,
                num_eval_docs=1,
                num_repeats=1,
                eval_continuation_tokens=32,
                output_dir=tmpdir,
            ),
            cache_config=AdaptiveCacheConfig(
                enabled=True,
                compress_min_len=0, max_cache_size=32, num_sink_tokens=4, recent_window_size=10,
                cache_budget_ratio=0.25, prefill_chunk_size=48,
            ),
            spec_config=SpeculativeConfig(enabled=True, gamma=3),
            evaluate_ppl=False,
        )
        rows = report["stage_results"]
        assert any(r["context_length"] == 96 and r["status"] == "OK" for r in rows)
        assert any(r["context_length"] == 192 and "OOM_SKIPPED" in r["status"] for r in rows)
        assert os.path.exists(os.path.join(tmpdir, "stages_comparison.json"))
        assert os.path.exists(os.path.join(tmpdir, "stages_comparison.csv"))


def test_step7_and_step8_multidoc_stage_sweep_and_batch_oom_skip(monkeypatch):
    """Steps 7 & 8: 5 disjoint docs, mean±std reporting, S0->S4 stage sweep, winning ranges, and batch OOM stop."""
    models = create_tiny_qwen_pair_for_testing()
    docs, doc_src = load_long_context_eval_texts(
        models.tokenizer, prefix_tokens=110, continuation_tokens=50, num_docs=5
    )
    assert len(docs) == 5
    assert len(set(docs)) == 5
    assert doc_src == "local:heldout"

    # Mock OOM at B >= 4 inside batch sweep
    import src.benchmark.runner as runner_mod

    orig_ar_gen = runner_mod.autoregressive_generate

    def batch_oom_ar(*args, **kwargs):
        p = kwargs.get("prompt")
        if isinstance(p, list) and len(p) >= 4:
            raise torch.cuda.OutOfMemoryError("Simulated batch OOM at B>=4")
        return orig_ar_gen(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "autoregressive_generate", batch_oom_ar)

    with tempfile.TemporaryDirectory() as tmpdir:
        report = run_stage_sweep(
            models=models,
            lengths=[128, 192],
            stages=["S0", "S2", "S4"],
            bench_config=BenchmarkConfig(
                sweep_lengths=[128, 192],
                stages=["S0", "S2", "S4"],
                sweep_max_new_tokens=12,
                num_eval_docs=2,
                num_repeats=1,
                eval_continuation_tokens=32,
                output_dir=tmpdir,
            ),
            cache_config=AdaptiveCacheConfig(
                enabled=True,
                compress_min_len=0,
                max_cache_size=32,
                num_sink_tokens=4,
                recent_window_size=10,
                cache_budget_ratio=0.25,
                heavy_hitter_ratio=1.0,
                prefill_chunk_size=48,
            ),
            spec_config=SpeculativeConfig(enabled=True, gamma=3),
            evaluate_ppl=True,
            batch_sizes=[1, 2, 4, 8],
            batch_lengths=[128],
        )
        assert os.path.exists(os.path.join(tmpdir, "stages_comparison.json"))
        assert os.path.exists(os.path.join(tmpdir, "stages_comparison.csv"))
        assert os.path.exists(os.path.join(tmpdir, "context_sweep.json"))
        assert os.path.exists(os.path.join(tmpdir, "stages_comparison.png"))

        with open(os.path.join(tmpdir, "stages_comparison.json"), encoding="utf-8") as f:
            saved = json.load(f)
        assert "beats_baseline_ranges" in saved["crossover_analysis"]
        ok_rows = [r for r in saved["stage_results"] if r.get("status") == "OK"]
        assert len(ok_rows) > 0
        for row in ok_rows:
            assert "tpot_ms_std" in row
            assert "perplexity_std" in row
            assert "peak_memory_mb" in row

        b_rows = saved["batch_size_results"]
        assert any(br["batch_size"] == 1 and br["status"] == "OK" for br in b_rows)
        assert any(br["batch_size"] == 2 and br["status"] == "OK" for br in b_rows)
        assert any(br["batch_size"] == 4 and "OOM_SKIPPED" in br["status"] for br in b_rows)
        assert not any(br["batch_size"] == 8 for br in b_rows)  # Stopped cleanly on OOM at B=4!


def test_split_kv_draft_attention_parity():
    """Draft decode attention (split-KV, packed GQA) == SDPA with repeat_kv, for q_len 1..3 and masked tails."""
    import torch.nn.functional as F
    from src.speculative.draft_attention import split_kv_decode_attention

    torch.manual_seed(0)
    for (b, hq, hkv, q, L, d) in [(1, 14, 2, 1, 2048, 64), (1, 14, 2, 2, 1024, 64), (2, 8, 2, 1, 96, 16), (1, 4, 4, 3, 100, 8)]:
        Q, K, V = torch.randn(b, hq, q, d), torch.randn(b, hkv, L, d), torch.randn(b, hkv, L, d)
        filled = L - 37  # static buffer with a stale (masked) tail
        qpos = torch.arange(q).view(1, 1, q, 1) + filled - q
        mask = (torch.arange(L).view(1, 1, 1, L) <= qpos).expand(b, 1, q, L)
        ref = F.scaled_dot_product_attention(
            Q, K.repeat_interleave(hq // hkv, 1), V.repeat_interleave(hq // hkv, 1), attn_mask=mask
        ).transpose(1, 2)
        out = split_kv_decode_attention(Q, K, V, mask, hq // hkv, None)
        assert out.shape == ref.shape
        assert (out - ref).abs().max().item() < 1e-5


def _mock_long_prompt(models, tokens):
    prompt, _ = build_long_context_prompt(models.tokenizer, target_tokens=tokens)
    return prompt


def _mock_adapt_cfg(min_len=0):
    return AdaptiveCacheConfig(
        enabled=True, compress_min_len=min_len, max_cache_size=40, cache_budget_ratio=0.25,
        num_sink_tokens=4, recent_window_size=12, prefill_chunk_size=64,
    )


def test_shared_keepset_static_draft_and_lossless_outputs():
    """
    --draft-cache shared_keepset: the draft StaticCache is compacted to the target's consensus keep-set
    (same count as each target layer, ~25% of context), RoPE positions stay logical, and the target's output is
    identical for every draft cache mode (the draft only changes acceptance).
    """
    models = create_tiny_qwen_pair_for_testing()
    prompt = _mock_long_prompt(models, 300)
    adapt = _mock_adapt_cfg()
    modes = {
        "shared_static": dict(draft_cache="shared_keepset"),
        "full_static": dict(draft_cache="full"),
        "shared_legacy_adaptive": dict(draft_cache="shared_keepset", use_static_draft_cache=False),
    }
    outs = {}
    for name, kw in modes.items():
        s = SpeculativeConfig(enabled=True, gamma=4, compile_draft=False, **kw)
        outs[name] = speculative_generate(
            models.draft_model, models.target_model, models.tokenizer, prompt, max_new_tokens=40,
            spec_config=s, target_cache_config=adapt, stop_on_eos=False,
        )
    ref = outs["full_static"].output_ids
    for o in outs.values():
        assert torch.equal(o.output_ids, ref)

    sh = outs["shared_static"]
    ds = sh.draft_cache_stats
    assert sh.draft_cache_mode == "shared_keepset"
    assert ds["cache_type"] == "StaticCache(shared_keepset)"
    # Same kept count as the target and as the legacy shared (eager AdaptiveKVCache) draft
    assert ds["active_seq_len"] == sh.cache_stats["active_seq_len"]
    assert ds["total_tokens_seen"] == sh.cache_stats["total_tokens_seen"]
    assert ds["active_seq_len"] == outs["shared_legacy_adaptive"].draft_cache_stats["active_seq_len"]
    assert ds["token_compression_ratio"] >= 3.5
    # In-place static compaction keeps the exact token set of the eager shared draft -> identical acceptance
    assert sh.acceptance_rate == outs["shared_legacy_adaptive"].acceptance_rate
    # The draft buffer is sized for ~25% of the context, not the full context
    assert ds["max_cache_len"] < outs["full_static"].draft_cache_stats["max_cache_len"]


def test_shared_keepset_indices_match_target_budget():
    """The draft keep-set has exactly the target layer's kept count, includes sinks + recent window, sorted."""
    torch.manual_seed(0)
    cfg = AdaptiveCacheConfig(enabled=True, max_cache_size=32, num_sink_tokens=4, recent_window_size=8)
    cache = AdaptiveKVCache(config=cfg)
    for layer in range(3):
        cache.update(torch.randn(1, 2, 100, 16), torch.randn(1, 2, 100, 16), layer)
    cache.compress_step(force=True)
    idx = cache.get_shared_keep_indices()
    assert idx is not None
    assert idx.numel() == cache.get_active_kv_length(0) == cache.get_active_kv_length(2)
    assert torch.equal(idx[:4], torch.arange(4))
    assert torch.equal(idx[-8:], torch.arange(92, 100))
    assert torch.all(idx[1:] > idx[:-1])


def test_compress_min_len_gates_compression():
    """Below compress_min_len: Proposed == Spec-only exactly (same path), AdaptKV == Baseline, PPL equal."""
    models = create_tiny_qwen_pair_for_testing()
    prompt = _mock_long_prompt(models, 200)
    gated = _mock_adapt_cfg(min_len=10 ** 6)
    full = AdaptiveCacheConfig(enabled=False, prefill_chunk_size=64)
    s = SpeculativeConfig(enabled=True, gamma=4, compile_draft=False)

    def run(cfg):
        return speculative_generate(
            models.draft_model, models.target_model, models.tokenizer, prompt, max_new_tokens=24,
            spec_config=s, target_cache_config=cfg, stop_on_eos=False,
        )

    p, sp = run(gated), run(full)
    assert torch.equal(p.output_ids, sp.output_ids)
    assert p.acceptance_rate == sp.acceptance_rate
    assert p.cache_stats["pruned_tokens"] == 0 and p.draft_cache_mode == "full"

    a = autoregressive_generate(models.target_model, models.tokenizer, prompt, max_new_tokens=24,
                                cache_config=gated, stop_on_eos=False)
    b = autoregressive_generate(models.target_model, models.tokenizer, prompt, max_new_tokens=24,
                                cache_config=full, stop_on_eos=False)
    assert torch.equal(a.output_ids, b.output_ids)

    ppl_g = evaluate_perplexity(models.target_model, models.tokenizer, [prompt], cache_config=gated,
                                continuation_tokens=32)
    ppl_f = evaluate_perplexity(models.target_model, models.tokenizer, [prompt], cache_config=full,
                                continuation_tokens=32)
    assert abs(ppl_g["perplexity"] - ppl_f["perplexity"]) < 1e-6
    # With the threshold met, compression is active and PPL goes through the compressed cache
    ppl_on = evaluate_perplexity(models.target_model, models.tokenizer, [prompt], cache_config=_mock_adapt_cfg(),
                                 continuation_tokens=32)
    assert ppl_on.get("pruned_tokens", 0) > 0
    assert abs(ppl_on["perplexity"] - ppl_f["perplexity"]) > 1e-6


def test_draft_step_profile_diagnostics():
    """profile_draft_step reports the draft decode path, mask vs capacity, static-address marking; output unchanged."""
    from dataclasses import replace as _replace

    models = create_tiny_qwen_pair_for_testing()
    prompt = _mock_long_prompt(models, 200)
    full = AdaptiveCacheConfig(enabled=False, prefill_chunk_size=64)
    base = SpeculativeConfig(enabled=True, gamma=4, compile_draft=False)
    o1 = speculative_generate(models.draft_model, models.target_model, models.tokenizer, prompt, max_new_tokens=24,
                              spec_config=base, target_cache_config=full, stop_on_eos=False)
    o2 = speculative_generate(models.draft_model, models.target_model, models.tokenizer, prompt, max_new_tokens=24,
                              spec_config=_replace(base, profile_draft_step=True), target_cache_config=full,
                              stop_on_eos=False)
    assert torch.equal(o1.output_ids, o2.output_ids)
    d = o2.draft_diag
    assert d["eager_attention"]["path"] == "split_kv_matmul"
    assert d["eager_attention"]["q_len"] == 1
    assert d["eager_attention"]["mask_shape"][-1] == d["static_capacity"]
    assert d["kv_static_address"] is True
    assert d["step_ms"] > 0
