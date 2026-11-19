"""Shared types and prompt helpers for the real-task benchmarks (LongBench, RULER)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from transformers import PreTrainedTokenizerBase


@dataclass
class TaskSample:
    """One fully-formatted evaluation example, identical for every method."""

    suite: str  # "longbench" | "longbench_e" | "ruler"
    task: str  # e.g. "hotpotqa", "niah_multikey_2"
    sample_id: str
    prompt: str  # final model input (chat template already applied if enabled)
    answers: List[str]
    max_new_tokens: int
    metric: str  # LongBench metric name or "string_match_all" / "string_match_part"
    prompt_tokens: int = 0
    context_length: Optional[int] = None  # RULER target length / LongBench raw length
    length_bucket: Optional[str] = None  # LongBench-E bucket or RULER length label
    all_classes: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def encode(tokenizer: PreTrainedTokenizerBase, text: str) -> List[int]:
    """Token ids of `text` without any tokenizer-added BOS (Qwen adds none; the test tokenizer does)."""
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return ids


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Number of tokens the generators will actually feed the model for `text`."""
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])


def middle_truncate(tokenizer: PreTrainedTokenizerBase, prompt: str, max_length: int) -> str:
    """
    Official LongBench truncation: if the prompt is longer than `max_length` tokens keep the first
    and last `max_length // 2` tokens (instructions sit at both ends, the middle is document body).
    """
    ids = encode(tokenizer, prompt)
    if len(ids) <= max_length:
        return prompt
    half = int(max_length / 2)
    return tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
        ids[-half:], skip_special_tokens=True
    )


def resolve_chat_template(mode: str, model_name: str, tokenizer: PreTrainedTokenizerBase) -> bool:
    """`auto` -> use the chat template only for Instruct/Chat checkpoints that ship one."""
    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if mode == "on":
        if not has_template:
            print("[Tasks] --chat-template on requested but the tokenizer has no chat template; using raw prompts.")
        return has_template
    if mode == "off":
        return False
    name = (model_name or "").lower()
    return has_template and ("instruct" in name or "chat" in name)


def apply_chat(
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    use_chat: bool,
    answer_prefix: str = "",
) -> str:
    """Wraps `user_prompt` as a single user turn (+ generation prompt) and appends `answer_prefix`."""
    if not use_chat:
        return user_prompt + answer_prefix
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True
    )
    return text + answer_prefix.lstrip()
