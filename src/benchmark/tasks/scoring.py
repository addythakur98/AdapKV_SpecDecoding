"""
Task scoring functions.

LongBench metrics are ports of the official `LongBench/metrics.py` + `eval.py`
(https://github.com/THUDM/LongBench), so numbers are comparable with published LongBench tables
(H2O / SnapKV / PyramidKV / KIVI all report on this scorer):

  qa_f1_score          token-level F1 after SQuAD normalisation (lower, strip punctuation/articles)
  rouge_score          ROUGE-L F from the `rouge` package (same package LongBench uses)
  classification_score TREC-style label matching with the official substring de-duplication
  retrieval_score      "Paragraph N" retrieval accuracy
  count_score          passage-count accuracy
  code_sim_score       fuzzy edit similarity of the first code line (fuzzywuzzy/rapidfuzz `ratio`)

RULER metrics are the official `string_match_all` (fraction of reference strings found in the
prediction, case-insensitive) and `string_match_part`.

Optional dependencies (`rouge`, `rapidfuzz`) have pure-python fallbacks; the fallback in use is
reported by `scorer_backends()` and written into every results file.
"""

from collections import Counter
import difflib
import re
import string
from typing import Callable, Dict, List, Optional, Sequence

try:  # Official LongBench uses `rouge` (pltrdy) for ROUGE-L.
    from rouge import Rouge as _Rouge

    _ROUGE = _Rouge()
except Exception:  # pragma: no cover - depends on environment
    _ROUGE = None

try:  # Official LongBench uses fuzzywuzzy.fuzz.ratio; rapidfuzz.fuzz.ratio is the same InDel ratio.
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover - depends on environment
    _fuzz = None


def scorer_backends() -> Dict[str, str]:
    return {
        "rouge": "rouge(pltrdy) [official]" if _ROUGE is not None else "builtin-LCS (fallback, not official)",
        "code_sim": "rapidfuzz.fuzz.ratio [official-equivalent]" if _fuzz is not None else "difflib (fallback, not official)",
    }


# --------------------------------------------------------------------------------------------
# LongBench metrics (English subset)
# --------------------------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1_tokens(prediction: Sequence[str], ground_truth: Sequence[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    return _f1_tokens(normalize_answer(prediction).split(), normalize_answer(ground_truth).split())


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _rouge_l_fallback(prediction: str, ground_truth: str) -> float:
    p, g = prediction.split(), ground_truth.split()
    lcs = _lcs_len(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    if _ROUGE is None:
        return _rouge_l_fallback(prediction, ground_truth)
    try:
        scores = _ROUGE.get_scores([prediction], [ground_truth], avg=True)
    except Exception:  # empty hypothesis etc. -> official scorer returns 0.0
        return 0.0
    return float(scores["rouge-l"]["f"])


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    em_match_list: List[str] = []
    all_classes = kwargs.get("all_classes") or []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    # Official quirk preserved: remove labels that are substrings of the ground truth (iterating a
    # list while removing from it, exactly as the reference implementation does).
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else float(right_num / len(numbers))


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    if _fuzz is not None:
        return float(_fuzz.ratio(prediction, ground_truth) / 100.0)
    return float(difflib.SequenceMatcher(None, prediction, ground_truth).ratio())


LONGBENCH_METRICS: Dict[str, Callable[..., float]] = {
    "qa_f1": qa_f1_score,
    "rouge": rouge_score,
    "classification": classification_score,
    "retrieval": retrieval_score,
    "count": count_score,
    "code_sim": code_sim_score,
}

# Official eval.py: these few-shot tasks are scored on the first generated line only.
LONGBENCH_FIRST_LINE_TASKS = ("trec", "triviaqa", "samsum", "lsht")


def score_longbench_sample(
    task: str,
    metric: str,
    prediction: str,
    answers: Sequence[str],
    all_classes: Optional[Sequence[str]] = None,
) -> float:
    """Per-sample LongBench score in [0, 1]: max over reference answers (official `scorer`)."""
    if task in LONGBENCH_FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    fn = LONGBENCH_METRICS[metric]
    best = 0.0
    for gt in answers:
        best = max(best, fn(prediction, gt, all_classes=all_classes))
    return float(best)


def longbench_e_bucket(length: int) -> str:
    """Official LongBench-E length buckets (on the dataset's `length` field)."""
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


# --------------------------------------------------------------------------------------------
# RULER metrics
# --------------------------------------------------------------------------------------------
def _ruler_postprocess(pred: str) -> str:
    # Official RULER `postprocess_pred`: strip, drop non-printable chars.
    pred = pred.strip()
    return re.sub(r"[\x00-\x1f]", "\n", pred).strip()


def string_match_all(prediction: str, references: Sequence[str]) -> float:
    pred = _ruler_postprocess(prediction).lower()
    if not references:
        return 0.0
    return sum(1.0 if r.lower() in pred else 0.0 for r in references) / len(references)


def string_match_part(prediction: str, references: Sequence[str]) -> float:
    pred = _ruler_postprocess(prediction).lower()
    return max((1.0 if r.lower() in pred else 0.0) for r in references) if references else 0.0
