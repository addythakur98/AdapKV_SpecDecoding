"""
Publication-Quality Visualization Generator for Benchmark Results.

Generates a 2x2 summary dashboard (`benchmark_comparison.png`) comparing:
  1. Throughput (tokens/s) & Speedup over Baseline
  2. Decode Latency (TPOT in ms/token)
  3. Target KV-Cache Memory Footprint (MB) & Compression Ratio
  4. Energy Consumption per Token (Joules/token) & Perplexity (PPL)
"""

import os
import tempfile
from typing import Any, Dict, List

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mplconfig_adaptive_kv")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_benchmark_plots(
    summary_rows: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Renders and saves the 4-panel ablation comparison chart."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "benchmark_comparison.png")

    labels = [r["short_name"] for r in summary_rows]
    throughputs = [r.get("decode_tokens_per_sec", r["tokens_per_sec"]) for r in summary_rows]
    tpots = [r["tpot_ms"] for r in summary_rows]
    kv_mem_mb = [r["kv_cache_actual_mb"] for r in summary_rows]
    kv_uncomp_mb = [r["kv_cache_uncompressed_mb"] for r in summary_rows]
    energy_per_tok = [r["energy_per_token_j"] for r in summary_rows]
    ppls = [r["perplexity"] for r in summary_rows]

    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    fig.suptitle(
        "Adaptive KV-Cache Compression & Speculative Decoding\n"
        "Draft: Qwen2.5-0.5B | Target: Qwen2.5-7B",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Decode Throughput (Tokens / sec, steady-state excluding prefill/TTFT)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, throughputs, color=colors[: len(labels)], width=0.55, edgecolor="black")
    ax1.set_title("Decode Throughput (Steady-State, Higher is Better)", fontweight="bold")
    ax1.set_ylabel("Decode Tokens / Second")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, r in zip(bars1, summary_rows):
        h = bar.get_height()
        sp = r.get("decode_speedup", r["speedup"])
        ax1.annotate(
            f"{h:.1f} tok/s\n({sp:.2f}x)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(throughputs) * 1.28 if max(throughputs) > 0 else 1.0)

    # Panel 2: Latency per Output Token (TPOT ms)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, tpots, color=colors[: len(labels)], width=0.55, edgecolor="black")
    ax2.set_title("Time Per Output Token - TPOT (Lower is Better)", fontweight="bold")
    ax2.set_ylabel("Latency (ms / token)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_ylim(0, max(tpots) * 1.25 if max(tpots) > 0 else 1.0)

    # Panel 3: KV-Cache Memory Footprint (Uncompressed vs Compressed)
    ax3 = axes[1, 0]
    w = 0.34
    b_uncomp = ax3.bar(
        x - w / 2,
        kv_uncomp_mb,
        width=w,
        label="Uncompressed FP16 Equivalent",
        color="#B0BEC5",
        edgecolor="black",
    )
    b_actual = ax3.bar(
        x + w / 2,
        kv_mem_mb,
        width=w,
        label="Actual KV-Cache Footprint",
        color="#2E7D32",
        edgecolor="black",
    )
    ax3.set_title("Target KV-Cache Memory Footprint (Lower is Better)", fontweight="bold")
    ax3.set_ylabel("Memory (MB)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.legend(fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, r in zip(b_actual, summary_rows):
        h = bar.get_height()
        ax3.annotate(
            f"{r['compression_ratio']:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    max_mem = max(max(kv_uncomp_mb, default=1.0), max(kv_mem_mb, default=1.0))
    ax3.set_ylim(0, max_mem * 1.25 if max_mem > 0 else 1.0)

    # Panel 4: Energy per Token (Joules) & Perplexity Overlay
    ax4 = axes[1, 1]
    bars4 = ax4.bar(
        x,
        energy_per_tok,
        color=colors[: len(labels)],
        width=0.55,
        edgecolor="black",
        label="Energy / Token (J)",
    )
    ax4.set_title("Energy per Token (J) & Perplexity (Lower is Better)", fontweight="bold")
    ax4.set_ylabel("Energy per Token (Joules)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha="right")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, r in zip(bars4, summary_rows):
        h = bar.get_height()
        ax4.annotate(
            f"{h:.3f} J\n(PPL: {r['perplexity']:.2f})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax4.set_ylim(0, max(energy_per_tok) * 1.30 if max(energy_per_tok) > 0 else 1.0)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path


def generate_context_sweep_plots(
    sweep_rows: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Renders 2x2 crossover curves across context lengths (e.g. 1K -> 32K)."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "context_sweep.png")

    method_styles = [
        ("baseline", "1. Baseline (Standard KV)", "#4C72B0", "o", "--"),
        ("adaptive_kv", "2. Adaptive KV Only", "#55A868", "s", "-."),
        ("speculative_only", "3. Speculative Only", "#DD8452", "^", ":"),
        ("combined_proposed", "4. Proposed (Spec + AdaptKV)", "#C44E52", "D", "-"),
    ]

    lengths = sorted({int(r["context_length"]) for r in sweep_rows})
    tick_labels = [
        f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n) for n in lengths
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    fig.suptitle(
        "Context-Length Scaling & Crossover Analysis (1K -> 32K)\n"
        "Draft: Qwen2.5-0.5B | Target: Qwen2.5-7B",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (
            axes[0, 0],
            "decode_tokens_per_sec",
            "Decode Throughput vs Context Length (Steady-State, Higher is Better)",
            "Decode Tokens / Second",
        ),
        (axes[0, 1], "tpot_ms", "Decode Latency TPOT vs Context Length (Lower is Better)", "TPOT (ms / token)"),
        (axes[1, 0], "kv_cache_actual_mb", "Active KV-Cache Memory vs Context Length (Lower is Better)", "KV Cache Memory (MB)"),
        (axes[1, 1], "energy_per_token_j", "Energy per Token vs Context Length (Lower is Better)", "Energy / Token (Joules)"),
    ]

    for ax, metric_key, title, ylabel in panels:
        for mode_id, label, color, marker, linestyle in method_styles:
            raw_pts = [r for r in sweep_rows if r["id"] == mode_id]
            if not raw_pts:
                continue
            # When sweeping multiple gammas, plot the best-decode-speed gamma at each context length.
            by_len: Dict[int, Dict[str, Any]] = {}
            for r in raw_pts:
                ctx = int(r["context_length"])
                cur_best = by_len.get(ctx)
                if cur_best is None or float(r.get("decode_tokens_per_sec", r.get("tokens_per_sec", 0.0))) > float(
                    cur_best.get("decode_tokens_per_sec", cur_best.get("tokens_per_sec", 0.0))
                ):
                    by_len[ctx] = r
            pts = [by_len[k] for k in sorted(by_len.keys())]
            xs = [int(r["context_length"]) for r in pts]
            ys = [
                float(r.get(metric_key, r.get("tokens_per_sec", 0.0)))
                for r in pts
            ]
            lw = 2.6 if mode_id == "combined_proposed" else 1.8
            ms = 7 if mode_id == "combined_proposed" else 5.5
            ax.plot(
                xs,
                ys,
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=lw,
                markersize=ms,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Prompt Context Length (Tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.legend(fontsize=8.2, loc="best")

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path


def generate_stage_comparison_plots(
    stage_rows: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Renders cumulative S0 -> S4 progression across TPOT, Decode tok/s, D/V Split, and Peak VRAM."""
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "stages_comparison.png")
    prop_rows = [
        r for r in stage_rows
        if r.get("status", "OK") == "OK" and r.get("mode_id", r.get("id")) in ("combined_proposed", "baseline")
    ]
    if not prop_rows:
        return plot_path

    lengths = sorted({int(r["context_length"]) for r in prop_rows})
    stages = []
    for r in prop_rows:
        st = str(r.get("stage", r.get("short_name", r.get("id"))))
        if st not in stages:
            stages.append(st)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160)
    fig.suptitle(
        "Cumulative Optimization Stages (S0 -> S4) vs Baseline (FP16)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    panels = [
        (axes[0, 0], "tpot_ms", "TPOT Across Optimization Stages (Lower is Better)", "TPOT (ms / token)"),
        (axes[0, 1], "decode_tokens_per_sec", "Decode Throughput Across Stages (Higher is Better)", "Decode tok/s"),
        (axes[1, 0], "verify_time_ms", "Target Verify Step Time Across Stages (Lower is Better)", "Verify (ms / step)"),
        (axes[1, 1], "peak_memory_mb", "Peak GPU Memory Across Stages (Lower is Better)", "Peak VRAM (MB)"),
    ]
    x = np.arange(len(stages))
    for ax, metric, title, ylabel in panels:
        for length in lengths:
            ys = []
            for st in stages:
                match = next(
                    (
                        r for r in prop_rows
                        if str(r.get("stage", r.get("id"))) == st and int(r["context_length"]) == length
                    ),
                    None,
                )
                ys.append(float(match.get(metric, 0.0)) if match else float("nan"))
            lbl = f"{length // 1024}K" if length >= 1024 and length % 1024 == 0 else str(length)
            ax.plot(x, ys, marker="o", linewidth=2.0, label=f"Ctx={lbl}")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=8.5)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------------------------
# Real-task benchmark plots (LongBench / RULER). Palette: validated categorical slots 1-4
# (adjacent CVD dE >= 9.1). Slots 3-4 are < 3:1 on white, so every chart carries a legend and the
# numbers are in task_summary.md (table view).
# ---------------------------------------------------------------------------------------------
_TASK_METHOD_STYLE = {
    "baseline": ("Baseline", "#2a78d6", "o"),
    "adaptive_kv": ("AdaptKV", "#eb6834", "s"),
    "speculative_only": ("Spec-only", "#1baf7a", "^"),
    "combined_proposed": ("Proposed", "#eda100", "D"),
}


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


def generate_task_plots(summary: Dict[str, Any], output_dir: str) -> List[str]:
    """Writes tasks_scores.png (per-task scores), tasks_ruler_length.png and tasks_efficiency.png."""
    methods = [m for m in summary.get("methods", []) if m in _TASK_METHOD_STYLE]
    paths: List[str] = []

    # 1. Per-task scores (LongBench / LongBench-E), grouped bars, one axis (score 0-100).
    lb = [t for t in summary["tasks"] if t["suite"] in ("longbench", "longbench_e")]
    if lb and methods:
        fig, ax = plt.subplots(figsize=(max(7.0, 1.25 * len(lb) + 2), 4.2))
        x = np.arange(len(lb))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            label, color, _ = _TASK_METHOD_STYLE[m]
            vals = [t["methods"].get(m, {}).get("score", float("nan")) for t in lb]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w * 0.92, color=color, label=label,
                   edgecolor="white", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([t["task"] for t in lb], rotation=25, ha="right")
        ax.set_ylabel("Task score (official metric, 0-100)", color="#52514e")
        ax.set_title("LongBench: task quality by method", loc="left", fontsize=12, color="#0b0b0b")
        _style_axes(ax)
        ax.legend(frameon=False, ncol=len(methods), loc="upper left", bbox_to_anchor=(0, -0.42), fontsize=9)
        fig.tight_layout()
        p = os.path.join(output_dir, "tasks_scores.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    # 2. RULER: average score vs context length, one line per method.
    ruler = {k: v for k, v in summary.get("suite_averages", {}).items() if k.startswith("ruler@")}
    if ruler and methods:
        lengths = sorted(int(k.split("@")[1]) for k in ruler)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for m in methods:
            label, color, marker = _TASK_METHOD_STYLE[m]
            ys = [ruler[f"ruler@{L}"].get(m, {}).get("score", float("nan")) for L in lengths]
            ax.plot(lengths, ys, color=color, marker=marker, markersize=7, linewidth=2, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels([f"{L // 1024}K" for L in lengths])
        ax.set_xlabel("Context length (tokens)", color="#52514e")
        ax.set_ylabel("RULER average score (0-100)", color="#52514e")
        ax.set_title("RULER: retrieval/tracking accuracy vs context", loc="left", fontsize=12, color="#0b0b0b")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        p = os.path.join(output_dir, "tasks_ruler_length.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)

    # 3. Efficiency on the same generations: decode speedup vs Baseline per suite (one axis, ratio).
    avgs = summary.get("suite_averages", {})
    if avgs and methods:
        names = list(avgs.keys())
        fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(names) + 2), 4.0))
        x = np.arange(len(names))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            label, color, _ = _TASK_METHOD_STYLE[m]
            vals = [avgs[n].get(m, {}).get("decode_speedup", float("nan")) for n in names]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, width=w * 0.92, color=color, label=label,
                   edgecolor="white", linewidth=1.0)
            if m == "combined_proposed":
                for xi, v in zip(x, vals):
                    if not _nan(v):
                        ax.annotate(f"{v:.2f}x", (xi + (i - (len(methods) - 1) / 2) * w, v),
                                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8,
                                    color="#0b0b0b")
        ax.axhline(1.0, color="#9a9994", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Decode speedup vs Baseline (x)", color="#52514e")
        ax.set_title("Decode speedup on real task prompts", loc="left", fontsize=12, color="#0b0b0b")
        _style_axes(ax)
        ax.legend(frameon=False, ncol=len(methods), loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=9)
        fig.tight_layout()
        p = os.path.join(output_dir, "tasks_efficiency.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    return paths
