"""
LongBench (THUDM, ACL 2024) loader and prompt builder.

Prompts, per-task generation lengths, metrics, middle truncation and the "no chat template for
few-shot / code tasks" rule are taken verbatim from the official repo
(config/dataset2prompt.json, config/dataset2maxlen.json, pred.py, eval.py) so the scores are
directly comparable with published LongBench numbers for KV-cache compression methods.

Data comes from the official `THUDM/LongBench` HF dataset repo (`data.zip`, jsonl per task), read
directly with `huggingface_hub` so it works with every `datasets` version (the repo uses a loading
script that `datasets>=4` refuses to run). `--longbench-dir` can point at an extracted copy instead.
"""

import json
import os
import random
import zipfile
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizerBase

from src.benchmark.tasks.common import TaskSample, apply_chat, count_tokens, middle_truncate
from src.benchmark.tasks.scoring import longbench_e_bucket

# Official dataset2prompt.json (English tasks). The "asconcisely" typo is in the official prompt.
LONGBENCH_PROMPTS: Dict[str, str] = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news articles. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

# Official dataset2maxlen.json
LONGBENCH_MAX_NEW_TOKENS: Dict[str, int] = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "hotpotqa": 32, "2wikimqa": 32,
    "musique": 32, "gov_report": 512, "qmsum": 512, "multi_news": 512, "trec": 64, "triviaqa": 32,
    "samsum": 128, "passage_count": 32, "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64,
}

# Official dataset2metric (eval.py)
LONGBENCH_METRIC: Dict[str, str] = {
    "narrativeqa": "qa_f1", "qasper": "qa_f1", "multifieldqa_en": "qa_f1", "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1", "musique": "qa_f1", "gov_report": "rouge", "qmsum": "rouge",
    "multi_news": "rouge", "trec": "classification", "triviaqa": "qa_f1", "samsum": "rouge",
    "passage_count": "count", "passage_retrieval_en": "retrieval", "lcc": "code_sim",
    "repobench-p": "code_sim",
}

LONGBENCH_CATEGORY: Dict[str, str] = {
    "narrativeqa": "Single-Doc QA", "qasper": "Single-Doc QA", "multifieldqa_en": "Single-Doc QA",
    "hotpotqa": "Multi-Doc QA", "2wikimqa": "Multi-Doc QA", "musique": "Multi-Doc QA",
    "gov_report": "Summarization", "qmsum": "Summarization", "multi_news": "Summarization",
    "trec": "Few-shot", "triviaqa": "Few-shot", "samsum": "Few-shot",
    "passage_count": "Synthetic", "passage_retrieval_en": "Synthetic",
    "lcc": "Code", "repobench-p": "Code",
}

# Official pred.py: no chat template for few-shot and code-completion tasks.
LONGBENCH_NO_CHAT = ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p")

# Tasks shipped in LongBench-E (length-balanced variant with 0-4k / 4-8k / 8k+ buckets).
LONGBENCH_E_TASKS = (
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec",
    "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
)

ALL_LONGBENCH_TASKS = tuple(LONGBENCH_PROMPTS.keys())

# Default subset: covers all six LongBench categories with a manageable GPU budget.
DEFAULT_LONGBENCH_TASKS = (
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "triviaqa",
    "passage_retrieval_en", "lcc",
)

# Official model2maxlen for 32K-context models (prompt tokens; generation comes on top).
LONGBENCH_DEFAULT_MAX_LEN = 31500


def _official_zip_path() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")


def load_longbench_records(task: str, use_e: bool = False, data_dir: Optional[str] = None) -> List[dict]:
    """Reads every record of one LongBench(-E) task from a local dir or the official data.zip."""
    fname = f"{task}_e.jsonl" if use_e else f"{task}.jsonl"
    if data_dir:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            path = os.path.join(data_dir, "data", fname)
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    zpath = _official_zip_path()
    with zipfile.ZipFile(zpath) as z:
        with z.open(f"data/{fname}") as f:
            return [json.loads(line) for line in f.read().decode("utf-8").splitlines() if line.strip()]


def select_indices(n_total: int, n: Optional[int], seed: int) -> List[int]:
    """Deterministic subset (same samples for every method / run / machine)."""
    if n is None or n <= 0 or n >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), n))


def build_longbench_samples(
    tokenizer: PreTrainedTokenizerBase,
    task: str,
    num_samples: Optional[int],
    max_length: int = LONGBENCH_DEFAULT_MAX_LEN,
    use_chat: bool = False,
    use_e: bool = False,
    data_dir: Optional[str] = None,
    seed: int = 42,
) -> List[TaskSample]:
    if task not in LONGBENCH_PROMPTS:
        raise ValueError(f"Unknown/unsupported LongBench task '{task}'. Supported: {', '.join(ALL_LONGBENCH_TASKS)}")
    if use_e and task not in LONGBENCH_E_TASKS:
        raise ValueError(f"'{task}' has no LongBench-E split. LongBench-E tasks: {', '.join(LONGBENCH_E_TASKS)}")
    records = load_longbench_records(task, use_e=use_e, data_dir=data_dir)
    if use_e and num_samples:
        # Stratify the subset across the three length buckets so every bucket is represented.
        by_bucket: Dict[str, List[int]] = {}
        for i, r in enumerate(records):
            by_bucket.setdefault(longbench_e_bucket(int(r.get("length", 0))), []).append(i)
        idx: List[int] = []
        per = max(1, num_samples // max(1, len(by_bucket)))
        for b_i, (bucket, members) in enumerate(sorted(by_bucket.items())):
            idx += [members[j] for j in select_indices(len(members), per, seed + b_i)]
        idx = sorted(idx)
    else:
        idx = select_indices(len(records), num_samples, seed)

    tmpl = LONGBENCH_PROMPTS[task]
    chat_ok = use_chat and task not in LONGBENCH_NO_CHAT
    samples: List[TaskSample] = []
    for i in idx:
        r = records[i]
        raw = tmpl.format(context=r["context"], input=r["input"])
        truncated = middle_truncate(tokenizer, raw, max_length)
        prompt = apply_chat(tokenizer, truncated, chat_ok)
        length = int(r.get("length", 0))
        samples.append(
            TaskSample(
                suite="longbench_e" if use_e else "longbench",
                task=task,
                sample_id=str(r.get("_id", i)),
                prompt=prompt,
                answers=list(r["answers"]),
                max_new_tokens=LONGBENCH_MAX_NEW_TOKENS[task],
                metric=LONGBENCH_METRIC[task],
                prompt_tokens=count_tokens(tokenizer, prompt),
                context_length=length,
                length_bucket=longbench_e_bucket(length),
                all_classes=r.get("all_classes"),
                meta={
                    "category": LONGBENCH_CATEGORY[task],
                    "truncated": truncated is not raw,
                    "chat_template": chat_ok,
                },
            )
        )
    return samples
