"""Tests for the real-task benchmarks (LongBench / RULER): official scorers, generators, runner."""

import json
import os
import tempfile

import pytest

from config import AdaptiveCacheConfig, SpeculativeConfig
from src.benchmark.tasks.common import middle_truncate, count_tokens
from src.benchmark.tasks.longbench import build_longbench_samples, LONGBENCH_PROMPTS
from src.benchmark.tasks.ruler import ALL_RULER_TASKS, build_ruler_samples
from src.benchmark.tasks.scoring import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    retrieval_score,
    rouge_score,
    score_longbench_sample,
    string_match_all,
)
from src.models.loader import create_tiny_qwen_pair_for_testing


# ---------------------------------------------------------------- official LongBench scorers
def test_qa_f1_matches_squad_normalisation():
    assert qa_f1_score("The Eiffel Tower!", "eiffel tower") == pytest.approx(1.0)
    # pred tokens {paris, france}, gt {paris}: P=0.5 R=1 -> F1=2/3
    assert qa_f1_score("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert qa_f1_score("London", "Paris") == 0.0


def test_classification_retrieval_count_code_scores():
    classes = ["Location", "Other location", "Individual"]
    assert classification_score("Other location", "Other location", all_classes=classes) == 1.0
    assert classification_score("Individual", "Other location", all_classes=classes) == 0.0
    assert retrieval_score("Paragraph 15", "Paragraph 15") == 1.0
    assert retrieval_score("Paragraph 1 or Paragraph 15", "Paragraph 15") == 0.5
    assert retrieval_score("no idea", "Paragraph 15") == 0.0
    assert count_score("There are 12 unique paragraphs", "12") == 1.0
    # code_sim scores the first line that is not a comment / fence
    assert code_sim_score("# comment\nreturn x + 1\nfoo()", "return x + 1") == pytest.approx(1.0)
    assert rouge_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0, abs=1e-6)


def test_first_line_rule_for_fewshot_tasks():
    # trec / triviaqa / samsum are scored on the first generated line only (official eval.py)
    pred = "Paris\nQuestion: something else London"
    assert score_longbench_sample("triviaqa", "qa_f1", pred, ["Paris"]) == pytest.approx(1.0)
    assert score_longbench_sample("hotpotqa", "qa_f1", pred, ["Paris"]) < 1.0


def test_ruler_string_match_all():
    assert string_match_all("The numbers are 123 and 456.", ["123", "456"]) == 1.0
    assert string_match_all("Only 123", ["123", "456"]) == 0.5
    assert string_match_all("", ["1"]) == 0.0


# ---------------------------------------------------------------- prompt construction
def test_middle_truncation_keeps_both_ends():
    tok = create_tiny_qwen_pair_for_testing().tokenizer
    text = "HEAD " + " ".join(f"w{i}" for i in range(500)) + " TAIL"
    out = middle_truncate(tok, text, 100)
    assert count_tokens(tok, out) <= 101  # +1 for the test tokenizer's BOS
    assert middle_truncate(tok, "short text", 100) == "short text"


def test_longbench_builder_from_local_dir():
    tok = create_tiny_qwen_pair_for_testing().tokenizer
    with tempfile.TemporaryDirectory() as d:
        recs = [
            {"input": f"Question {i}?", "context": " ".join(["ctx"] * (50 + i * 400)), "answers": [f"ans{i}"],
             "length": 50 + i * 400, "all_classes": None, "_id": f"id{i}"}
            for i in range(5)
        ]
        with open(os.path.join(d, "hotpotqa.jsonl"), "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r) for r in recs))
        s1 = build_longbench_samples(tok, "hotpotqa", 3, max_length=600, data_dir=d, seed=1)
        s2 = build_longbench_samples(tok, "hotpotqa", 3, max_length=600, data_dir=d, seed=1)
        assert [x.sample_id for x in s1] == [x.sample_id for x in s2]  # deterministic subset
        assert all(x.prompt_tokens <= 601 for x in s1)
        assert all(x.max_new_tokens == 32 and x.metric == "qa_f1" for x in s1)
        # (The test tokenizer cannot decode back to text, so only untruncated prompts are checked verbatim.)
        assert all(x.prompt.endswith("Answer:") for x in s1 if not x.meta["truncated"])
        assert any(x.meta["truncated"] for x in s1)
        assert "hotpotqa" in LONGBENCH_PROMPTS


@pytest.mark.parametrize("task", ALL_RULER_TASKS)
def test_ruler_generators_fit_length_and_contain_answers(task):
    tok = create_tiny_qwen_pair_for_testing().tokenizer
    samples = build_ruler_samples(tok, task, 2048, 3, seed=7)
    again = build_ruler_samples(tok, task, 2048, 3, seed=7)
    assert [s.prompt for s in samples] == [s.prompt for s in again]  # byte-identical per seed
    assert len({s.prompt for s in samples}) == 3  # samples differ from each other
    for s in samples:
        assert s.prompt_tokens <= 2048 - s.max_new_tokens
        assert s.prompt_tokens >= 0.8 * (2048 - s.max_new_tokens)  # haystack actually fills the context
        assert s.answers and all(a in s.prompt for a in s.answers)


# ---------------------------------------------------------------- runner (mock models, offline)
def test_task_runner_end_to_end_and_resume():
    from src.benchmark.task_runner import run_task_benchmark

    models = create_tiny_qwen_pair_for_testing()
    cache_cfg = AdaptiveCacheConfig(enabled=True, max_cache_size=32, num_sink_tokens=4, recent_window_size=10,
                                    cache_budget_ratio=0.25, compress_min_len=0)
    spec_cfg = SpeculativeConfig(enabled=True, gamma=3)
    with tempfile.TemporaryDirectory() as d:
        kw = dict(models=models, suites=["ruler"], longbench_tasks=[], ruler_tasks=["niah_single_1", "vt"],
                  ruler_lengths=[1024], num_samples=2, cache_config=cache_cfg, spec_config=spec_cfg,
                  model_names=("mock", "mock-draft"), output_dir=d)
        summary = run_task_benchmark(**kw)
        for f in ("task_predictions.jsonl", "task_summary.json", "task_summary.md", "task_summary.csv"):
            assert os.path.exists(os.path.join(d, f))
        assert len(summary["tasks"]) == 2
        t0 = summary["tasks"][0]
        assert set(t0["methods"]) == {"baseline", "adaptive_kv", "speculative_only", "combined_proposed"}
        assert t0["methods"]["adaptive_kv"]["compressed_frac"] == 1.0
        assert t0["methods"]["baseline"]["compressed_frac"] == 0.0
        # Greedy speculative decoding is lossless vs. the same cache without speculation.
        assert t0["lossless"]["speculative_only_vs_baseline"]["exact_match_pct"] == 100.0
        n_rows = sum(1 for _ in open(os.path.join(d, "task_predictions.jsonl"), encoding="utf-8"))
        assert n_rows == 2 * 2 * 4
        # Resume: nothing is regenerated, summary is rebuilt from the JSONL.
        summary2 = run_task_benchmark(**kw)
        n_rows2 = sum(1 for _ in open(os.path.join(d, "task_predictions.jsonl"), encoding="utf-8"))
        assert n_rows2 == n_rows
        assert summary2["suite_averages"].keys() == summary["suite_averages"].keys()
