"""
Quality & Perplexity Evaluation Metrics.

Evaluates:
1. Streaming Autoregressive Perplexity (PPL) with and without Adaptive KV-Cache Compression,
   quantifying how well the compressed KV cache preserves predictive log-likelihood over
   long sequences.
2. Token-level F1 / N-gram overlap between baseline and compressed/speculative outputs.
"""

from collections import Counter
import math
import random
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from config import AdaptiveCacheConfig, cache_config_for_context
from src.kv_cache.adaptive_cache import AdaptiveKVCache
from src.speculative.static_runner import (
    get_peak_vram_mb,
    model_forward_with_logits_to_keep,
    reset_peak_vram,
    supports_logits_to_keep,
)


@torch.no_grad()
def evaluate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    cache_config: Optional[AdaptiveCacheConfig] = None,
    chunk_size: int = 256,
    continuation_tokens: int = 0,
    logits_to_keep: Optional[bool] = None,
) -> Dict[str, float]:
    """
    Computes exact streaming autoregressive Perplexity (PPL) using `AdaptiveKVCache`
    with chunked prefill & compression between chunks so long (1K-32K) sequences run
    without O(N) Python loops or peak-logits OOM.

    When `logits_to_keep` is enabled (default), prefix chunks skip materializing full-chunk
    LM-head logits (`logits_to_keep=1`), dramatically reducing peak VRAM on 8k-32k sequences
    without changing evaluated log-likelihood by a single bit.
    """
    device = next(model.parameters()).device
    base_cfg = cache_config or AdaptiveCacheConfig(enabled=False)
    use_ltk = (
        (base_cfg.logits_to_keep if logits_to_keep is None else bool(logits_to_keep))
        and supports_logits_to_keep(model)
    )

    total_nll = 0.0
    total_tokens = 0
    final_cache_stats: Dict[str, float] = {}
    reset_peak_vram(device)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids: torch.Tensor = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])
        if seq_len < 2:
            continue

        # Target token indices [eval_start, seq_len) are scored; prefix [0, eval_start) populates cache.
        eval_start = (
            max(1, seq_len - continuation_tokens)
            if continuation_tokens > 0 and seq_len > continuation_tokens + 16
            else 1
        )

        # --compress-min-len: a prefix (context) shorter than the threshold is scored through an uncompressed
        # cache, with the same chunking as Baseline, so AdaptKV PPL == Baseline PPL there by construction.
        cfg = cache_config_for_context(base_cfg, seq_len)
        eval_chunk = (
            max(16, min(chunk_size, cfg.prefill_chunk_size, cfg.max_cache_size))
            if cfg.enabled
            else max(16, min(chunk_size, cfg.prefill_chunk_size))
        )
        kv_cache = AdaptiveKVCache(config=cfg)
        if cfg.enabled:
            kv_cache.register_query_hooks(model)
        prev_last_logit: Optional[torch.Tensor] = None

        try:
            for start in range(0, seq_len, eval_chunk):
                end = min(seq_len, start + eval_chunk)
                chunk_ids = input_ids[:, start:end]
                chunk_len = end - start

                if end < eval_start:
                    # Pure prefix chunk whose last logit isn't predicting an evaluated token yet
                    _ = model_forward_with_logits_to_keep(
                        model,
                        logits_to_keep_count=1 if use_ltk else None,
                        input_ids=chunk_ids,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
                    prev_last_logit = None
                elif end == eval_start:
                    # Last position predicts token `eval_start` in the next chunk
                    out = model_forward_with_logits_to_keep(
                        model,
                        logits_to_keep_count=1 if use_ltk else None,
                        input_ids=chunk_ids,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
                    prev_last_logit = out.logits[:, -1, :].float()
                else:
                    # Scored continuation (either partial or full chunk)
                    need_from = max(0, (eval_start - 1) - start)
                    keep_n = chunk_len - need_from
                    out = model_forward_with_logits_to_keep(
                        model,
                        logits_to_keep_count=keep_n if use_ltk else None,
                        input_ids=chunk_ids,
                        past_key_values=kv_cache,
                        use_cache=True,
                    )
                    raw_logits = out.logits
                    if not use_ltk and need_from > 0:
                        raw_logits = raw_logits[:, need_from:, :]

                    # Cross-chunk boundary prediction: previous chunk's final logit predicts token `start`
                    if prev_last_logit is not None and start >= eval_start:
                        boundary_loss = F.cross_entropy(
                            prev_last_logit, chunk_ids[:, 0], reduction="sum"
                        )
                        total_nll += float(boundary_loss.item())
                        total_tokens += 1

                    # Intra-chunk next-token predictions for target positions [start + need_from + 1, end)
                    if chunk_len > need_from + 1 and (end - 1) >= eval_start:
                        shift_logits = raw_logits[:, :-1, :].float().reshape(-1, raw_logits.shape[-1])
                        shift_targets = chunk_ids[:, need_from + 1 :].reshape(-1)
                        if shift_targets.numel() > 0:
                            intra_loss = F.cross_entropy(shift_logits, shift_targets, reduction="sum")
                            total_nll += float(intra_loss.item())
                            total_tokens += int(shift_targets.numel())

                    prev_last_logit = raw_logits[:, -1, :].float()

                if cfg.enabled:
                    kv_cache.compress_step(force=True)

            kv_cache.compress_step(force=True)
            final_cache_stats = kv_cache.get_memory_stats()
        finally:
            kv_cache.remove_query_hooks()

    mean_nll = total_nll / max(1, total_tokens)
    ppl = math.exp(min(20.0, mean_nll))  # Clamp exponent to avoid overflow on random init models

    return {
        "perplexity": round(ppl, 4),
        "mean_nll": round(mean_nll, 4),
        "evaluated_tokens": total_tokens,
        "cache_compression_ratio": final_cache_stats.get("compression_ratio", 1.0),
        "pruned_tokens": int(final_cache_stats.get("pruned_tokens", 0) or 0),
        "kept_tokens": int(final_cache_stats.get("active_seq_len", 0) or 0),
        "peak_vram_mb": get_peak_vram_mb(device),
    }


_PROSE_TOPICS = [
    ("paleoclimatology", "Antarctic ice-core isotopes", "Holocene monsoon shifts", "benthic foraminifera records"),
    ("maritime cartography", "sixteenth-century portolan charts", "magnetic declination tables", "chronometer longitude trials"),
    ("urban hydrology", "permeable basalt pavements", "subterranean retention cisterns", "tidal estuary floodgates"),
    ("metallurgical forensics", "crucible slag inclusions", "annealed bronze microstructures", "trace arsenic alloys"),
    ("observational astronomy", "interferometric baseline calibration", "circumstellar dust disks", "millisecond pulsar timing"),
    ("acoustic architecture", "coffered timber ceilings", "reverberation decay curves", "helmholtz cavity resonators"),
    ("agricultural genetics", "drought-tolerant root architecture", "diploid landrace accessions", "photoperiod flowering loci"),
    ("railway signaling", "track-circuit relay interlocking", "gradient braking curves", "cab-signaling transponders"),
    ("glassmaking chemistry", "soda-lime batch fining", "annealing lehr temperature gradients", "cobalt silicate pigments"),
    ("Alpine glaciology", "subglacial basal sliding", "crevasse ablation stakes", "radar-sounded bed topography"),
    ("bookbinding conservation", "alum-tawed pigskin spines", "laid-paper chain lines", "iron-gall ink corrosion"),
    ("estuarine ecology", "salt-marsh cordgrass accretion", "bivalve filtration rates", "dissolved oxygen stratification"),
]

_CITIES = [
    "Bergen", "Trieste", "Valdivia", "Oaxaca", "Kanazawa", "Inverness",
    "Plovdiv", "Coimbra", "Ghent", "Arequipa", "Szeged", "Tartu",
    "Dunedin", "Cadiz", "Aomori", "Brno", "Galway", "Ljubljana",
]

_ADJECTIVES = [
    "measured", "gradual", "irregular", "archived", "preliminary", "verified",
    "layered", "tilted", "brittle", "weathered", "coherent", "subtle",
    "calibrated", "episodic", "regional", "structural", "seasonal", "sparse",
]

_VERBS = [
    "altered", "documented", "stabilized", "complicated", "revealed", "constrained",
    "accelerated", "offset", "preserved", "reshuffled", "delimited", "corroborated",
]

_HELDOUT_TOPICS = [
    ("numismatic cataloging", "silver tetradrachm dies", "overstruck magistrate monograms", "hoard provenance ledgers"),
    ("piano acoustics", "duplex string scaling", "felt hammer voicing needles", "spruce soundboard crown"),
    ("karst geomorphology", "phreatic limestone conduits", "stalagmite growth laminae", "epikarst dye traces"),
    ("mycological taxonomy", "spore print ornamentation", "ectomycorrhizal birch hosts", "clamp-connection hyphae"),
    ("clockwork horology", "deadbeat anchor escapements", "bimetallic temperature balances", "fused brass lantern pinions"),
    ("textile weaving", "drawloom pattern harnesses", "madder-dyed worsted weft", "selvage tension weights"),
]


def _generate_unique_paragraph(idx: int) -> str:
    """
    Generates a deterministic, fluent, non-repeating paragraph with high lexical and
    structural variety. Context indices (`idx < 100000`) and held-out indices
    (`idx >= 100000`) draw from completely disjoint domain/template pools so model
    perplexity does not artificially collapse over long contexts.
    """
    rng = random.Random(1009 + idx * 7919)
    if idx >= 100000:
        topic, obj_a, obj_b, obj_c = _HELDOUT_TOPICS[(idx - 100000) % len(_HELDOUT_TOPICS)]
        place = rng.choice(_CITIES)
        adj = rng.choice(_ADJECTIVES)
        verb = rng.choice(_VERBS)
        yr = rng.randint(1840, 2023)
        val1 = round(rng.uniform(3.1, 89.7), 2)
        val2 = rng.randint(14, 480)
        return (
            f"During a {yr} survey in {place} concerning {topic}, curators compared {obj_a} against {obj_b} "
            f"across {val2} specimens. A {adj} shift averaging {val1} percent {verb} earlier assumptions drawn from {obj_c}, "
            f"prompting archivists to re-index the surviving plates before cataloging the next folio.\n"
        )

    topic, obj_a, obj_b, obj_c = _PROSE_TOPICS[idx % len(_PROSE_TOPICS)]
    city_a, city_b = rng.sample(_CITIES, 2)
    adj_a, adj_b = rng.sample(_ADJECTIVES, 2)
    verb_a, verb_b = rng.sample(_VERBS, 2)
    year = rng.randint(1785, 2024)
    n1 = rng.randint(11, 950)
    n2 = round(rng.uniform(0.4, 74.9), 2)
    pattern = idx % 5
    if pattern == 0:
        return (
            f"Section {idx + 1} ({topic}): Fieldwork conducted near {city_a} in {year} examined {n1} samples of {obj_a}. "
            f"Observers noted a {adj_a} pattern that {verb_a} standard interpretations of {obj_b}, while laboratory checks in {city_b} "
            f"showed {obj_c} varying by {n2} units across successive strata.\n"
        )
    if pattern == 1:
        return (
            f"Archive entry {idx + 1}: Correspondence sent from {city_b} to {city_a} around {year} described {obj_b} in detail. "
            f"Rather than following uniform intervals, {adj_b} readings near {n2} {verb_b} how {topic} specialists paired {obj_a} with {obj_c} across {n1} trials.\n"
        )
    if pattern == 2:
        return (
            f"Working paper {idx + 1} addresses {topic} through comparative measurements gathered over {n1} weeks outside {city_a}. "
            f"Because {obj_c} exhibited {adj_a} sensitivity above threshold {n2}, analysts revised the {year} baseline for {obj_a} and {obj_b}.\n"
        )
    if pattern == 3:
        return (
            f"Chronicle {idx + 1}: Inspectors stationed in {city_a} and {city_b} documented how {obj_a} {verb_a} during seasonal transitions in {year}. "
            f"Closer inspection of {obj_b} revealed {n1} distinct markers whose {adj_b} spacing differed by {n2} percent from {obj_c}.\n"
        )
    return (
        f"Monograph note {idx + 1} summarizes {year} investigations into {topic} led by practitioners from {city_b}. "
        f"Integrating {obj_c} alongside {n1} preserved instances of {obj_a} {verb_b} prior debates over {adj_a} {obj_b} near {n2} mm.\n"
    )


def build_long_context_prompt(
    tokenizer: PreTrainedTokenizerBase,
    target_tokens: int,
    passkey: Optional[str] = "73914",
    needle_depth: float = 0.30,
    para_offset: int = 0,
) -> Tuple[str, Optional[str]]:
    """
    Constructs a deterministic, non-repeating long-context prompt whose tokenized length
    lands cleanly at `<= target_tokens` (exact power-of-2 lengths such as 16384 instead of
    off-by-1 16385), optionally embedding a passkey needle at `needle_depth`.
    """
    needle_sentence = (
        f" [ARCHIVE RECORD: The confidential verification passkey is {passkey}. Remember {passkey}.] "
        if passkey
        else ""
    )
    suffix = (
        "\nQuestion: What is the confidential verification passkey stated in the archive record above, "
        "and summarize the empirical findings discussed in the notes?\nAnswer:"
        if passkey
        else "\nQuestion: Summarize the key observations described above.\nAnswer:"
    )

    sample_block = _generate_unique_paragraph(para_offset) + _generate_unique_paragraph(para_offset + 1)
    suffix_len = int(tokenizer(suffix, return_tensors="pt")["input_ids"].shape[1])
    needle_len = (
        int(tokenizer(needle_sentence, return_tensors="pt")["input_ids"].shape[1])
        if needle_sentence
        else 0
    )
    body_target = max(16, target_tokens - suffix_len - needle_len)

    # Estimate tokens per paragraph from a 2-paragraph sample
    sample_ids = tokenizer(sample_block, return_tensors="pt")["input_ids"][0].tolist()
    tokens_per_para = max(1, (len(sample_ids) - 1) // 2)
    num_paras = max(2, (body_target // tokens_per_para) + 2)

    needle_para_idx = min(num_paras - 1, max(1, int(round(num_paras * needle_depth))))
    parts: List[str] = []
    para_counter = para_offset
    for idx in range(num_paras):
        if passkey and idx == needle_para_idx:
            parts.append(needle_sentence)
        parts.append(_generate_unique_paragraph(para_counter))
        para_counter += 1

    # Fine-tune exact token length so prompt never overshoots `target_tokens`
    candidate = "".join(parts) + suffix
    cur_len = int(tokenizer(candidate, return_tensors="pt")["input_ids"].shape[1])
    while cur_len < target_tokens:
        parts.append(_generate_unique_paragraph(para_counter))
        para_counter += 1
        candidate = "".join(parts) + suffix
        cur_len = int(tokenizer(candidate, return_tensors="pt")["input_ids"].shape[1])

    while len(parts) > 2 and cur_len > target_tokens + tokens_per_para:
        drop_idx = len(parts) - 1
        if passkey and parts[drop_idx] == needle_sentence:
            drop_idx -= 1
        parts.pop(drop_idx)
        candidate = "".join(parts) + suffix
        cur_len = int(tokenizer(candidate, return_tensors="pt")["input_ids"].shape[1])

    # Binary-search trim the final paragraph so tokenized length <= target_tokens
    while cur_len > target_tokens and len(parts) > 1:
        trim_idx = len(parts) - 1
        if passkey and parts[trim_idx] == needle_sentence:
            trim_idx -= 1
        last_words = parts[trim_idx].split()
        if len(last_words) <= 1:
            parts.pop(trim_idx)
            candidate = "".join(parts) + suffix
            cur_len = int(tokenizer(candidate, return_tensors="pt")["input_ids"].shape[1])
            continue
        lo, hi = 1, len(last_words)
        best_words: Optional[List[str]] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            parts[trim_idx] = " ".join(last_words[:mid]) + "\n"
            test_len = int(
                tokenizer("".join(parts) + suffix, return_tensors="pt")["input_ids"].shape[1]
            )
            if test_len <= target_tokens:
                best_words = last_words[:mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if best_words is None:
            parts.pop(trim_idx)
        else:
            parts[trim_idx] = " ".join(best_words) + "\n"
        candidate = "".join(parts) + suffix
        cur_len = int(tokenizer(candidate, return_tensors="pt")["input_ids"].shape[1])
        if best_words is not None:
            break

    return candidate, passkey


def load_long_context_eval_texts(
    tokenizer: PreTrainedTokenizerBase,
    prefix_tokens: int,
    continuation_tokens: int = 256,
    dataset_name: Optional[str] = None,
    num_docs: int = 5,
    fallback_prompts: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """
    Step 7: Loads `num_docs` disjoint documents containing `prefix_tokens + continuation_tokens`
    tokens each (streaming non-overlapping spans from HF `pg19`/`wikitext` or deterministic disjoint
    local held-out texts) for reporting `mean ± std` PPL across multiple evaluation documents.
    """
    docs_needed = max(1, int(num_docs))
    total_needed = prefix_tokens + continuation_tokens
    docs: List[str] = []
    source_name = "local:heldout"

    if dataset_name:
        try:
            from datasets import load_dataset

            ds_norm = dataset_name.strip().lower()
            if ds_norm in ("pg19", "pg-19"):
                try:
                    ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
                except Exception:
                    ds = load_dataset("deepmind/pg19", split="test", streaming=True)
                field = "text"
            elif ds_norm in ("wikitext", "wikitext-103", "wikitext2"):
                try:
                    ds = load_dataset(
                        "Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True
                    )
                except Exception:
                    ds = load_dataset(
                        "wikitext", "wikitext-103-raw-v1", split="test", streaming=True
                    )
                field = "text"
            else:
                ds = load_dataset(dataset_name, split="test", streaming=True)
                field = "text"

            token_buf: List[int] = []
            for item in ds:
                chunk_text = (item.get(field) or "").strip()
                if len(chunk_text) < 200:
                    continue
                chunk_ids = tokenizer(chunk_text, return_tensors="pt")["input_ids"][0].tolist()
                token_buf.extend(chunk_ids)
                while len(token_buf) >= total_needed and len(docs) < docs_needed:
                    doc_ids = token_buf[:total_needed]
                    token_buf = token_buf[total_needed + 32 :]
                    docs.append(tokenizer.decode(doc_ids, skip_special_tokens=False))
                if len(docs) >= docs_needed:
                    return docs, f"hf:{dataset_name}"
            if docs:
                source_name = f"hf:{dataset_name}+local"
        except Exception as exc:
            print(f"[Metrics] Corpus '{dataset_name}' unavailable ({exc}); falling back to local held-out text.")

    while len(docs) < docs_needed:
        doc_idx = len(docs)
        if fallback_prompts and doc_idx < len(fallback_prompts):
            prefix_text = fallback_prompts[doc_idx]
        else:
            prefix_text, _ = build_long_context_prompt(
                tokenizer,
                target_tokens=prefix_tokens,
                passkey="73914",
                para_offset=doc_idx * 25000,
            )
        cont_text, _ = build_long_context_prompt(
            tokenizer,
            target_tokens=continuation_tokens,
            passkey=None,
            para_offset=100000 + doc_idx * 25000,
        )
        docs.append(prefix_text + "\n" + cont_text)

    return docs, source_name


def load_long_context_eval_text(
    tokenizer: PreTrainedTokenizerBase,
    prefix_tokens: int,
    continuation_tokens: int = 256,
    dataset_name: Optional[str] = None,
    fallback_prompt: Optional[str] = None,
) -> Tuple[str, str]:
    """Single-document convenience wrapper preserving backward compatibility."""
    docs, src = load_long_context_eval_texts(
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        continuation_tokens=continuation_tokens,
        dataset_name=dataset_name,
        num_docs=1,
        fallback_prompts=[fallback_prompt] if fallback_prompt is not None else None,
    )
    return docs[0], src


def check_passkey_hit(generated_text: str, passkey: Optional[str]) -> Optional[bool]:
    """Checks whether the embedded passkey appears in generated text."""
    if not passkey:
        return None
    return passkey in generated_text


def compute_ngram_f1(reference_ids: List[int], candidate_ids: List[int], n: int = 1) -> float:
    """Computes token-level n-gram F1 similarity between baseline and candidate sequences."""
    if not reference_ids and not candidate_ids:
        return 1.0
    if not reference_ids or not candidate_ids:
        return 0.0

    def _get_ngrams(seq: List[int], k: int) -> Counter:
        if len(seq) < k:
            return Counter([tuple(seq)])
        return Counter(tuple(seq[i : i + k]) for i in range(len(seq) - k + 1))

    ref_counts = _get_ngrams(reference_ids, n)
    cand_counts = _get_ngrams(candidate_ids, n)

    overlap = sum((ref_counts & cand_counts).values())
    precision = overlap / max(1, sum(cand_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)
