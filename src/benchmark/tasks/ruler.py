"""
RULER (Hsieh et al., NVIDIA, COLM 2024) synthetic long-context tasks at exact token lengths.

Re-implementation of the official generators (scripts/data/synthetic/*.py) with the official
templates, answer prefixes, needle formats, task configs (synthetic.yaml) and generation lengths.
Implemented: the 8 NIAH variants, variable tracking (vt), common-words (cwe) and
frequent-words (fwe) extraction = 11 of RULER's 13 tasks. The two QA tasks are omitted because
LongBench already covers real QA (SQuAD/HotpotQA-based RULER QA would duplicate it).

Differences from the official data pipeline (also recorded in every results file):
  * "essay" haystack = Project Gutenberg PG-19 test books (official: Paul Graham essays).
    If PG-19 cannot be downloaded the noise haystack is used and the sample is tagged.
  * Sentence splitting for needle insertion uses a regex instead of nltk.sent_tokenize.
Everything is seeded per (task, length, index): every method sees byte-identical prompts.
"""

import hashlib
import random
import re
import string
import uuid
from typing import Callable, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from src.benchmark.tasks.common import TaskSample, apply_chat, count_tokens

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
    "I will quiz you about the {type_needle_v} afterwards.\n{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
)
NIAH_ANSWER_PREFIX = " The special magic {type_needle_v} for {query} mentioned in the provided text are"
NIAH_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."

VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
)
VT_ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are "
    "assgined the value {query}, they are: "
)

CWE_TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more often than others. "
    "Memorize the ones that appear most often.\n{context}\n"
    "Question: What are the 10 most common words in the above list?"
)
CWE_ANSWER_PREFIX = " Answer: The top 10 words that appear most often in the list are:"

FWE_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. Find the three most "
    "frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore "
    "the dots '....'. What are the three most frequently appeared words in the above coded text?"
)
FWE_ANSWER_PREFIX = " Answer: According to the coded text above, the three most frequently appeared words are:"

# Official synthetic.yaml task configs.
RULER_TASKS: Dict[str, Dict] = {
    "niah_single_1": dict(kind="niah", haystack="noise", key="words", value="numbers", nk=1, nv=1, nq=1),
    "niah_single_2": dict(kind="niah", haystack="essay", key="words", value="numbers", nk=1, nv=1, nq=1),
    "niah_single_3": dict(kind="niah", haystack="essay", key="words", value="uuids", nk=1, nv=1, nq=1),
    "niah_multikey_1": dict(kind="niah", haystack="essay", key="words", value="numbers", nk=4, nv=1, nq=1),
    "niah_multikey_2": dict(kind="niah", haystack="needle", key="words", value="numbers", nk=1, nv=1, nq=1),
    "niah_multikey_3": dict(kind="niah", haystack="needle", key="uuids", value="uuids", nk=1, nv=1, nq=1),
    "niah_multivalue": dict(kind="niah", haystack="essay", key="words", value="numbers", nk=1, nv=4, nq=1),
    "niah_multiquery": dict(kind="niah", haystack="essay", key="words", value="numbers", nk=4, nv=1, nq=4),
    "vt": dict(kind="vt", num_chains=1, num_hops=4),
    "cwe": dict(kind="cwe", freq_cw=30, freq_ucw=3, num_cw=10),
    "fwe": dict(kind="fwe", alpha=2.0),
}
RULER_MAX_NEW_TOKENS = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50}
ALL_RULER_TASKS = tuple(RULER_TASKS.keys())
DEFAULT_RULER_TASKS = (
    "niah_single_2", "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
)

# --------------------------------------------------------------------------------------------
# Vocabularies / haystacks
# --------------------------------------------------------------------------------------------
_WORDS_CACHE: Dict[str, List[str]] = {}
_ESSAY_CACHE: Dict[str, List[str]] = {}


def _wonderwords_lists() -> Tuple[List[str], List[str], List[str], str]:
    """nouns, adjectives, verbs (the lists RULER uses). Falls back to pseudo-words if unavailable."""
    if "nouns" not in _WORDS_CACHE:
        try:
            from wonderwords import random_word as _rw

            _WORDS_CACHE["nouns"] = _rw._get_words_from_text_file("nounlist.txt")
            _WORDS_CACHE["adjs"] = _rw._get_words_from_text_file("adjectivelist.txt")
            _WORDS_CACHE["verbs"] = _rw._get_words_from_text_file("verblist.txt")
            _WORDS_CACHE["src"] = ["wonderwords"]
        except Exception:
            rng = random.Random(0)
            syl = [c + v for c in "bcdfghjklmnprstvwz" for v in "aeiou"]

            def mk(n: int) -> List[str]:
                out = set()
                while len(out) < n:
                    out.add("".join(rng.choice(syl) for _ in range(rng.randint(2, 4))))
                return sorted(out)

            _WORDS_CACHE["nouns"], _WORDS_CACHE["adjs"], _WORDS_CACHE["verbs"] = mk(6000), mk(900), mk(1000)
            _WORDS_CACHE["src"] = ["pseudo-words (wonderwords not installed)"]
    return _WORDS_CACHE["nouns"], _WORDS_CACHE["adjs"], _WORDS_CACHE["verbs"], _WORDS_CACHE["src"][0]


def word_source() -> str:
    return _wonderwords_lists()[3]


def load_essay_words(min_words: int = 200_000) -> Tuple[Optional[List[str]], str]:
    """Word list from PG-19 test books (cached). Returns (None, reason) if unavailable."""
    if "words" in _ESSAY_CACHE:
        return _ESSAY_CACHE["words"], "pg19"
    try:
        from datasets import load_dataset

        try:
            ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
        except Exception:
            ds = load_dataset("deepmind/pg19", split="test", streaming=True)
        words: List[str] = []
        for item in ds:
            text = item.get("text") or ""
            # Drop the Gutenberg header/licence boilerplate at the start of each book.
            words.extend(text.split()[500:])
            if len(words) >= min_words:
                break
        if len(words) < 50_000:
            return None, "pg19 too short"
        _ESSAY_CACHE["words"] = words
        return words, "pg19"
    except Exception as exc:
        return None, f"pg19 unavailable ({type(exc).__name__})"


# --------------------------------------------------------------------------------------------
# Generators. Each returns (user_prompt_without_answer_prefix, answer_prefix, answers).
# `n` is the haystack size knob that the length fitter searches over.
# --------------------------------------------------------------------------------------------
def _gen_key(kind: str, rng: random.Random) -> str:
    if kind == "words":
        nouns, adjs, _, _ = _wonderwords_lists()
        return f"{rng.choice(adjs)}-{rng.choice(nouns)}"
    if kind == "numbers":
        return str(rng.randint(10**6, 10**7 - 1))
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _niah(cfg: Dict, n: int, seed: int, essay: Optional[List[str]]) -> Tuple[str, str, List[str], Dict]:
    rng = random.Random(seed)
    tv = cfg["value"]
    keys, values, needles = [], [], []
    for _ in range(cfg["nk"]):
        k = _gen_key(cfg["key"], rng)
        while k in keys:
            k = _gen_key(cfg["key"], rng)
        keys.append(k)
        vals = [_gen_key(tv, rng) for _ in range(cfg["nv"])]
        values.append(vals)
        needles += [NIAH_NEEDLE.format(type_needle_v=tv, key=k, value=v) for v in vals]
    rng.shuffle(needles)

    haystack = cfg["haystack"]
    meta = {"haystack": haystack}
    if haystack == "essay" and essay is None:
        haystack = "noise"
        meta["haystack"] = "noise (fallback: essay haystack unavailable)"

    place_rng = random.Random(seed + 1)
    n = max(n, len(needles))
    if haystack == "noise":
        sentences = [NOISE] * n
        for idx, nd in zip(sorted(place_rng.sample(range(n), len(needles)), reverse=True), needles):
            sentences.insert(idx, nd)
        context = "\n".join(sentences)
    elif haystack == "needle":
        sentences = []
        for _ in range(n):
            dk = _gen_key(cfg["key"], rng)
            while dk in keys:
                dk = _gen_key(cfg["key"], rng)
            sentences.append(NIAH_NEEDLE.format(type_needle_v=tv, key=dk, value=_gen_key(tv, rng)))
        for idx, nd in zip(sorted(place_rng.sample(range(n), len(needles)), reverse=True), needles):
            sentences.insert(idx, nd)
        context = "\n".join(sentences)
    else:  # essay
        start = (seed * 7919) % max(1, len(essay) - n - 1)
        sentences = _SENT_SPLIT.split(" ".join(essay[start : start + n]))
        depths = sorted(place_rng.random() for _ in needles)
        pieces, prev = [], 0
        for d, nd in zip(depths, needles):
            pos = int(len(sentences) * d)
            pieces += sentences[prev:pos] + [nd]
            prev = pos
        pieces += sentences[prev:]
        context = " ".join(pieces)

    q_idx = rng.sample(range(cfg["nk"]), cfg["nq"])
    queries = [keys[i] for i in q_idx]
    answers = [a for i in q_idx for a in values[i]]
    query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]
    prompt = NIAH_TEMPLATE.format(type_needle_v=tv, context=context, query=query)
    return prompt, NIAH_ANSWER_PREFIX.format(type_needle_v=tv, query=query), answers, meta


def _vt_body(n: int, rng: random.Random, num_chains: int, num_hops: int, var_len: int) -> Tuple[str, str, List[str]]:
    total = (num_hops + 1) * num_chains
    names = set()
    while len(names) < total:
        names.add("".join(rng.choices(string.ascii_uppercase, k=var_len)))
    names = list(names)
    rng.shuffle(names)
    chains, var_sets = [], []
    for c in range(num_chains):
        vs = names[c * (num_hops + 1) : (c + 1) * (num_hops + 1)]
        var_sets.append(vs)
        chain = [f"VAR {vs[0]} = {rng.randint(10000, 99999)}"]
        chain += [f"VAR {vs[j + 1]} = VAR {vs[j]} " for j in range(num_hops)]
        chains.append(chain)
    sentences = [NOISE] * n
    for chain in chains:
        positions = sorted(rng.sample(range(len(sentences)), len(chain)))
        for j, p in enumerate(positions):
            sentences.insert(p + j, chain[j])
    context = "\n".join(sentences).replace(". \n", ".\n")
    value = chains[0][0].split("=")[-1].strip()
    return context, value, var_sets[0]


def _vt(cfg: Dict, n: int, seed: int, essay=None) -> Tuple[str, str, List[str], Dict]:
    nh = cfg["num_hops"]
    ex_ctx, ex_val, ex_vars = _vt_body(5, random.Random(seed + 10_000), 1, nh, 3)
    icl = (
        VT_TEMPLATE.format(context=ex_ctx, query=ex_val)
        + VT_ANSWER_PREFIX.format(num_v=nh + 1, query=ex_val)
        + " ".join(ex_vars)
    )
    ctx, val, vars_ = _vt_body(max(len(ex_vars) + 1, n), random.Random(seed), cfg["num_chains"], nh, 5)
    prompt = icl + "\n\n" + VT_TEMPLATE.format(context=ctx, query=val)
    return prompt, VT_ANSWER_PREFIX.format(num_v=nh + 1, query=val), vars_, {"haystack": "noise"}


def _cwe_words(seed: int) -> List[str]:
    nouns, adjs, verbs, _ = _wonderwords_lists()
    words = sorted(set(nouns + adjs + verbs))
    random.Random(seed).shuffle(words)
    return words


def _cwe_body(num_words: int, rng: random.Random, words: List[str], cw: int, ucw: int, ncw: int):
    full = rng.sample(words, min(num_words, len(words)))
    common, uncommon = full[:ncw], full[ncw:]
    wl = common * cw + uncommon * ucw
    rng.shuffle(wl)
    return " ".join(f"{i + 1}. {w}" for i, w in enumerate(wl)), common


def _cwe(cfg: Dict, n: int, seed: int, essay=None) -> Tuple[str, str, List[str], Dict]:
    words = _cwe_words(42)
    ex_ctx, ex_common = _cwe_body(20, random.Random(seed + 10_000), words, 3, 1, 10)
    icl = CWE_TEMPLATE.format(context=ex_ctx) + CWE_ANSWER_PREFIX + " " + " ".join(
        f"{i + 1}. {w}" for i, w in enumerate(ex_common)
    )
    ctx, common = _cwe_body(max(cfg["num_cw"] + 1, n), random.Random(seed), words, cfg["freq_cw"], cfg["freq_ucw"], cfg["num_cw"])
    return icl + "\n\n" + CWE_TEMPLATE.format(context=ctx), CWE_ANSWER_PREFIX, common, {"haystack": "word-list"}


def _fwe_body(num_words: int, seed: int, alpha: float, vocab_size: int = 2000, wordlen: int = 6):
    rng = random.Random(seed)
    vocab = set()
    while len(vocab) < vocab_size:
        vocab.add("".join(rng.choices(string.ascii_lowercase, k=wordlen)))
    vocab = sorted(vocab)
    rng.shuffle(vocab)
    vocab[0] = "..."  # official: the top-ranked "word" is noise
    zeta = sum(k ** -alpha for k in range(1, 100_000))
    sampled: List[str] = []
    for k, w in enumerate(vocab, start=1):
        sampled += [w] * int(num_words * (k ** -alpha) / zeta)
    rng.shuffle(sampled)
    return " ".join(sampled), vocab[1:4]


def _fwe(cfg: Dict, n: int, seed: int, essay=None) -> Tuple[str, str, List[str], Dict]:
    ex_ctx, ex_ans = _fwe_body(200, seed + 10_000, cfg["alpha"])
    icl = FWE_TEMPLATE.format(context=ex_ctx) + FWE_ANSWER_PREFIX + " " + " ".join(ex_ans)
    ctx, ans = _fwe_body(max(40, n), seed, cfg["alpha"])
    return icl + "\n\n" + FWE_TEMPLATE.format(context=ctx), FWE_ANSWER_PREFIX, ans, {"haystack": "coded-words"}


_GENERATORS: Dict[str, Callable] = {"niah": _niah, "vt": _vt, "cwe": _cwe, "fwe": _fwe}


def _fit_length(
    build: Callable[[int], str],
    tokenizer: PreTrainedTokenizerBase,
    budget: int,
    n_init: int,
    n_max: Optional[int] = None,
) -> Tuple[int, str, int]:
    """Largest haystack size n whose prompt fits in `budget` tokens (secant search + exact step-down)."""
    n = max(1, n_init)
    cache: Dict[int, Tuple[str, int]] = {}

    def measure(k: int) -> int:
        if k not in cache:
            txt = build(k)
            cache[k] = (txt, count_tokens(tokenizer, txt))
        return cache[k][1]

    lo_n, lo_t = 1, measure(1)
    if lo_t > budget:
        raise ValueError(f"RULER prompt cannot fit {budget} tokens (minimum is {lo_t}).")
    for _ in range(8):
        if n_max is not None:
            n = min(n, n_max)
        t = measure(n)
        if t <= budget and (budget - t) < max(16, budget // 200):
            break
        per = max(1e-6, (t - lo_t) / max(1, n - lo_n))
        n_new = max(1, lo_n + int((budget - lo_t) / per))
        if t <= budget:
            lo_n, lo_t = n, t
        if n_new == n:
            break
        n = n_new
    while measure(n) > budget and n > 1:
        n = max(1, int(n * 0.98) if n > 100 else n - 1)
    txt, t = cache[n]
    return n, txt, t


def build_ruler_samples(
    tokenizer: PreTrainedTokenizerBase,
    task: str,
    context_length: int,
    num_samples: int,
    use_chat: bool = False,
    seed: int = 42,
) -> List[TaskSample]:
    if task not in RULER_TASKS:
        raise ValueError(f"Unknown RULER task '{task}'. Supported: {', '.join(ALL_RULER_TASKS)}")
    cfg = RULER_TASKS[task]
    kind = cfg["kind"]
    gen = _GENERATORS[kind]
    max_new = RULER_MAX_NEW_TOKENS[kind]
    budget = int(context_length) - max_new  # official: prompt + generation <= max_seq_length

    essay = None
    if kind == "niah" and cfg.get("haystack") == "essay":
        essay, _ = load_essay_words()

    # Initial haystack-size guess in tokens per unit (noise sentence / needle line / essay word /
    # distinct list word (x3 repeats) / coded word). The fitter corrects it after one measurement.
    if kind == "niah":
        hay = cfg["haystack"] if (essay is not None or cfg["haystack"] != "essay") else "noise"
        per_unit = {"noise": 22.0, "needle": 20.0, "essay": 1.35}[hay]
    else:
        per_unit = {"vt": 22.0, "cwe": 10.0, "fwe": 2.2}[kind]
    n_init = max(1, int(budget / per_unit))
    n_max = len(essay) - 1 if (essay is not None and kind == "niah") else None

    samples: List[TaskSample] = []
    for i in range(num_samples):
        # Stable across processes (Python's str hash is salted, so it is not used here).
        s_seed = int(hashlib.sha256(f"{task}|{context_length}|{seed}|{i}".encode()).hexdigest()[:8], 16)
        holder: Dict[str, object] = {}

        def build(n: int, _s=s_seed) -> str:
            user, prefix, answers, meta = gen(cfg, n, _s, essay)
            holder.update(answers=answers, meta=meta)
            return apply_chat(tokenizer, user, use_chat, prefix)

        n, prompt, ntok = _fit_length(build, tokenizer, budget, n_init, n_max)
        build(n)  # make sure `holder` reflects the chosen n
        n_init = n
        samples.append(
            TaskSample(
                suite="ruler",
                task=task,
                sample_id=f"{task}-{context_length}-{i}",
                prompt=prompt,
                answers=list(holder["answers"]),
                max_new_tokens=max_new,
                metric="string_match_all",
                prompt_tokens=ntok,
                context_length=int(context_length),
                length_bucket=str(context_length),
                meta={**dict(holder["meta"]), "haystack_units": n, "words": word_source()},
            )
        )
    return samples
