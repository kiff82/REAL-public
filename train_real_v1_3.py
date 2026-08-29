"""
REAL: Recursive External Alignment Layer
---------------------------------------

This script is a Colab-friendly training bed for REAL as a contact system:
a frozen pretrained LLM manifold in contact with a small input-conditioned head.

Key idea:
- Frozen LLM weights = static manifold / pretrained basins
- The learned head is a contact-making signal, not a replacement model
- Train the head to:
    - extracts an "internal vector field" proxy from the prompt (pooled embeddings)
    - samples "thought space" candidates (particles) before collapsing into an intent latent
    - iteratively refines the latent (dynamic basin)
    - projects the latent into prefix embeddings (contact signal)
    - predicts an energy per step that matches true per-step supervised loss
    - supports an inference-time adaptive refinement policy

This is not weight-space "deepening". It is state-space contact refinement.

Run (Colab):
  !pip -q install -U "transformers>=4.51.0" accelerate datasets bitsandbytes
  python train_real_v1_3.py

Notes:
- Some models (e.g., Gemma) require accepting license terms on Hugging Face.
- Qwen3 requires transformers >= 4.51.0.
"""

import argparse
import json
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from real.core.dtype_ops import auto_dtype
from real.core.contact_loss import average_candidate_logprob
from real.core.loss_ops import causal_ce_per_sample
from real.core.prefix_ops import build_inputs_embeds_with_prefix, clamp_prefix_norm
from utils.hotpot_reasoning_contracts import (
    HOTPOT_REASONING_OUTPUT_CONTRACTS,
    hotpot_reasoning_prompt_prefix as _shared_hotpot_reasoning_prompt_prefix,
    hotpot_reasoning_response_cue as _shared_hotpot_reasoning_response_cue,
    hotpot_reasoning_target_continuation_text as _shared_hotpot_reasoning_target_continuation_text,
    hotpot_reasoning_target_lines as _shared_hotpot_reasoning_target_lines,
    normalize_hotpot_reasoning_output_contract as _shared_normalize_hotpot_reasoning_output_contract,
    normalize_hotpot_reasoning_prompt_variant as _shared_normalize_hotpot_reasoning_prompt_variant,
    normalize_hotpot_reasoning_response_cue_variant as _shared_normalize_hotpot_reasoning_response_cue_variant,
    select_hotpot_key_facts,
)
from utils.insufficiency_contact_path_selection import (
    clean_insufficiency_candidate_text,
    dedupe_candidate_records,
    question_terms,
    select_hard_negative_by_scores,
)


# -------------------------
# 0) Utilities
# -------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def temp_seed(seed: int):
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    x:   [B, L, D]
    mask:[B, L] bool/0-1
    """
    mask_f = mask.float()
    denom = mask_f.sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=dim) / denom


@torch.no_grad()
def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    a, b: any shape, treated as flattened vectors
    """
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.std(unbiased=False) * b.std(unbiased=False)).clamp(min=eps)
    return float((a * b).mean() / denom)





def _truncate_tokens_head_tail(tokens: List[int], max_len: int, head_ratio: float = 0.35, min_head: int = 32) -> List[int]:
    """Truncate a token list by keeping both a head and a tail segment.

    This is a safe default for prompts that have:
      - important instructions / question near the beginning, AND
      - an answer delimiter near the end (e.g., 'Answer: ' / 'Summary: ' / 'ORIGINAL:').

    If tokens <= max_len, returns tokens unchanged.
    """
    if len(tokens) <= max_len:
        return tokens
    if max_len <= 0:
        return []

    head_len = int(max_len * head_ratio)
    head_len = max(min_head, head_len)
    head_len = min(head_len, max_len)
    tail_len = max_len - head_len

    if tail_len <= 0:
        return tokens[:max_len]

    return tokens[:head_len] + tokens[-tail_len:]


# Probe prompt delimiters: include a trailing separator to avoid tokenization edge effects.
ANSWER_DELIM = "Answer: "
SUMMARY_DELIM = "Summary: "
DEFAULT_CONTEXT_ABSTAIN_TEXT = "Cannot be inferred from the provided context."
LEGACY_TRUTHFULQA_ABSTAIN_TEXT = "I don't know"
FEVER_REASONING_NEI_TEXT = "Not enough information from the provided evidence"


def _truncate_text_head_tail(text: str, max_chars: int, head_chars: int = 2000, tail_chars: int = 1000) -> str:
    """Cheap char-level truncation to avoid pathological tokenization costs."""
    if max_chars is None or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head = text[: min(head_chars, max_chars)]
    remain = max_chars - len(head)
    if remain <= 0:
        return head
    tail = text[-min(tail_chars, remain):]
    return head + "\n...\n" + tail


def _normalize_eval_text(text: str) -> str:
    """Small normalization helper for answer-vs-abstain comparisons."""
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _crop_context_around_answers(
    context: str,
    answers: List[str],
    *,
    window_before: int = 600,
    window_after: int = 600,
    fallback_max_chars: int = 2400,
    fallback_head_chars: int = 1600,
    fallback_tail_chars: int = 800,
) -> str:
    """Keep a stable context slice around the first matched answer span when possible."""
    ctx_l = context.lower()
    for answer in answers:
        ans = (answer or "").strip()
        if not ans:
            continue
        pos = ctx_l.find(ans.lower())
        if pos == -1:
            continue
        start = max(0, pos - window_before)
        end = min(len(context), pos + len(ans) + window_after)
        return context[start:end]
    return _truncate_text_head_tail(
        context,
        max_chars=fallback_max_chars,
        head_chars=fallback_head_chars,
        tail_chars=fallback_tail_chars,
    )


def build_sft_example(
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
    answer_loss_tokens: Optional[int],
    prompt_head_ratio: float = 0.35,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Dict[str, List[int]]:
    """Build a causal-LM SFT example with a *prompt* and a *short supervised answer window*.

    Guarantees:
      - The supervised answer window is preserved under truncation.
      - prompt_len is consistent with the returned tokenization.
      - There is at least 1 supervised token (falls back to keeping 1 token).

    Rationale:
      The project relies on an answer-start loss window (ANSWER_LOSS_TOKENS) to keep the
      task strongly prompt-conditioned. When sequences overflow, we must avoid accidentally
      truncating away the supervised answer tokens (which can create zero-loss / misleading
      evaluation).
    """

    bos = tokenizer.bos_token_id if add_bos else None
    eos = tokenizer.eos_token_id if add_eos else None

    prompt_ids: List[int] = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids: List[int] = tokenizer(answer, add_special_tokens=False).input_ids

    if eos is not None:
        answer_ids = answer_ids + [eos]

    if answer_loss_tokens is not None:
        # Keep only the supervised answer window. Tokens beyond this do not affect loss.
        keep_n = max(1, int(answer_loss_tokens))
        answer_ids = answer_ids[:keep_n]

    # Ensure we can fit at least BOS + 1 answer token.
    bos_len = 1 if bos is not None else 0
    if max_length <= bos_len + 1:
        max_length = bos_len + 1

    # Reserve space for BOS + answer window; truncate prompt to fit.
    reserved = bos_len + len(answer_ids)
    if reserved > max_length:
        # Rare edge case: answer window itself is too large; shrink it.
        max_ans = max(1, max_length - bos_len)
        answer_ids = answer_ids[:max_ans]
        reserved = bos_len + len(answer_ids)

    prompt_budget = max_length - reserved
    if len(prompt_ids) > prompt_budget:
        prompt_ids = _truncate_tokens_head_tail(prompt_ids, prompt_budget, head_ratio=prompt_head_ratio)

    input_ids = ([] if bos is None else [bos]) + prompt_ids + answer_ids
    prompt_len = bos_len + len(prompt_ids)

    labels = ([-100] * prompt_len) + answer_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_len,
    }


def _find_answer_token_span(offsets: List[Tuple[int, int]], ans_start: int, ans_end: int) -> Optional[Tuple[int, int]]:
    token_start = None
    token_end = None
    for i, (s, e) in enumerate(offsets):
        if e <= ans_start:
            continue
        if s >= ans_end:
            token_end = i
            break
        if token_start is None:
            token_start = i
    if token_start is None:
        return None
    if token_end is None:
        token_end = len(offsets)
    return token_start, token_end


def build_extractive_qa_example(
    tokenizer,
    context: str,
    question: str,
    answer: str,
    max_length: int,
    answer_loss_tokens: Optional[int],
    add_bos: bool = True,
    add_eos: bool = True,
) -> Dict[str, List[int]]:
    bos = tokenizer.bos_token_id if add_bos else None
    eos = tokenizer.eos_token_id if add_eos else None

    answer_ids: List[int] = tokenizer(answer, add_special_tokens=False).input_ids
    if eos is not None:
        answer_ids = answer_ids + [eos]

    if answer_loss_tokens is not None:
        keep_n = max(1, int(answer_loss_tokens))
        answer_ids = answer_ids[:keep_n]

    ctx_prefix_ids = tokenizer("Context: ", add_special_tokens=False).input_ids
    q_prefix_ids = tokenizer("\nQuestion: ", add_special_tokens=False).input_ids
    a_prefix_ids = tokenizer("\n" + ANSWER_DELIM, add_special_tokens=False).input_ids

    question_ids = tokenizer(question, add_special_tokens=False).input_ids

    bos_len = 1 if bos is not None else 0
    if max_length <= bos_len + 1:
        max_length = bos_len + 1

    fixed_prefix_len = len(ctx_prefix_ids) + len(q_prefix_ids) + len(a_prefix_ids)
    max_question = max_length - bos_len - fixed_prefix_len - 1
    if max_question < 0:
        max_question = 0
    if len(question_ids) > max_question:
        question_ids = question_ids[:max_question]

    max_answer = max_length - bos_len - fixed_prefix_len - len(question_ids)
    if max_answer < 1:
        max_answer = 1
    if len(answer_ids) > max_answer:
        answer_ids = answer_ids[:max_answer]

    ctx_kwargs = {"add_special_tokens": False}
    offsets = None
    if getattr(tokenizer, "is_fast", False):
        ctx_kwargs["return_offsets_mapping"] = True
    try:
        ctx_enc = tokenizer(context, **ctx_kwargs)
        offsets = ctx_enc.get("offset_mapping")
    except TypeError:
        ctx_enc = tokenizer(context, add_special_tokens=False)
        offsets = None

    context_ids = ctx_enc["input_ids"]

    fixed_prompt_len = len(ctx_prefix_ids) + len(q_prefix_ids) + len(question_ids) + len(a_prefix_ids)
    reserved = bos_len + fixed_prompt_len + len(answer_ids)
    context_budget = max_length - reserved
    if context_budget < 0:
        context_budget = 0

    if len(context_ids) > context_budget:
        span = None
        if offsets is not None:
            ctx_l = context.lower()
            ans_l = answer.lower()
            pos = ctx_l.find(ans_l)
            if pos != -1:
                span = _find_answer_token_span(offsets, pos, pos + len(answer))
        if span is not None:
            span_start, span_end = span
            span_len = max(1, span_end - span_start)
            if span_len >= context_budget:
                start = span_start
                end = span_start + context_budget
            else:
                left = (context_budget - span_len) // 2
                start = max(0, span_start - left)
                end = start + context_budget
                if end > len(context_ids):
                    end = len(context_ids)
                    start = max(0, end - context_budget)
            context_ids = context_ids[start:end]
        else:
            context_ids = _truncate_tokens_head_tail(context_ids, context_budget, head_ratio=0.50)

    prompt_ids = ctx_prefix_ids + context_ids + q_prefix_ids + question_ids + a_prefix_ids
    input_ids = ([] if bos is None else [bos]) + prompt_ids + answer_ids
    prompt_len = bos_len + len(prompt_ids)
    labels = ([-100] * prompt_len) + answer_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_len,
    }


# -------------------------
# 1) Dataset: text -> corrupted_text, target = original_text
# -------------------------

class DenoiseToOriginalDataset(Dataset):
    """
    Builds samples like:
      PROMPT:  Restore the original text.
               CORRUPTED: ...
               ORIGINAL:
      TARGET:  <original text>

    This is SFT-style (causal LM):
      - prompt tokens have label = -100
      - answer tokens have label = token_id
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 256,
        p_drop_word: float = 0.18,
        p_swap_adj: float = 0.06,
        seed: int = 0,
        max_samples: int = 20000,
        answer_loss_tokens: Optional[int] = None,
        deterministic_corruption: bool = False,
        dataset_name: str = "wikitext",
        dataset_config_name: str = "wikitext-2-raw-v1",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens
        self.p_drop_word = p_drop_word
        self.p_swap_adj = p_swap_adj
        self.seed = seed
        self.rng = random.Random(seed)
        self.deterministic_corruption = deterministic_corruption

        ds = load_dataset(dataset_name, dataset_config_name, split=split)
        texts = []
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if len(t) < 60:
                continue
            texts.append(t)
            if len(texts) >= max_samples:
                break
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def _corrupt(self, text: str, rng: random.Random) -> str:
        words = text.split()
        if len(words) < 8:
            return text

        kept = [w for w in words if rng.random() > self.p_drop_word]
        if len(kept) < 3:
            kept = words[: max(3, len(words) // 2)]

        i = 0
        while i < len(kept) - 1:
            if rng.random() < self.p_swap_adj:
                kept[i], kept[i + 1] = kept[i + 1], kept[i]
                i += 2
            else:
                i += 1

        return " ".join(kept)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        clean = self.texts[idx]
        rng = random.Random(self.seed + idx) if self.deterministic_corruption else self.rng
        noisy = self._corrupt(clean, rng)

        prompt = (
            "Restore the original text.\n"
            "CORRUPTED:\n"
            f"{noisy}\n"
            "ORIGINAL:\n"
        )

        # Preserve the supervised answer window; truncate prompt if needed.
        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=clean,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.50,
            add_bos=True,
            add_eos=True,
        )

class BoolQShortAnswerDataset(Dataset):
    """
    BoolQ-style yes/no classification framed as short-form generation.

    PROMPT:
      Answer the question with yes or no.
      Question: ...
      "Answer: "
    TARGET: "yes" | "no"
    """

    def __init__(
        self,
        tokenizer,
        split: str = "validation",
        max_length: int = 160,
        max_samples: int = 1000,
        answer_loss_tokens: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        ds = load_dataset("boolq", split=split)
        self.examples = []
        for ex in ds:
            q = (ex.get("question") or "").strip()
            if not q:
                continue
            ans = "yes" if ex.get("answer") else "no"
            p = (ex.get("passage") or "").strip()
            if not p:
                continue
            self.examples.append((q, p, ans))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        question, passage, answer = self.examples[idx]
        # Avoid pathological tokenization costs on very long passages.
        passage = _truncate_text_head_tail(passage, max_chars=4000, head_chars=2400, tail_chars=1200)

        prompt = (
            "Answer the question with yes or no.\n"
            f"Question: {question}\n"
            f"Passage: {passage}\n"
            + ANSWER_DELIM
        )

        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=answer,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.45,
            add_bos=True,
            add_eos=True,
        )

class MultipleChoiceLetterDataset(Dataset):
    """
    Multiple-choice QA framed as predicting a single option letter.

    PROMPT:
      Question: ...
      Options:
      A) ...
      B) ...
      ...
      "Answer: "
    TARGET: "A" | "B" | ...
    """

    def __init__(
        self,
        tokenizer,
        split: str = "validation",
        max_length: int = 224,
        max_samples: int = 1000,
        dataset_name: str = "ai2_arc",
        dataset_config_name: str = "ARC-Challenge",
        answer_loss_tokens: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        ds = load_dataset(dataset_name, dataset_config_name, split=split)
        self.examples = []
        for ex in ds:
            question = (ex.get("question") or "").strip()
            choices = ex.get("choices", {})
            texts = choices.get("text") or []
            labels = choices.get("label") or []
            answer_key = (ex.get("answerKey") or "").strip()
            if not question or not answer_key or answer_key not in labels:
                continue
            options = []
            for lbl, txt in zip(labels, texts):
                if not lbl or not txt:
                    continue
                options.append((lbl.strip(), txt.strip()))
            if len(options) < 2:
                continue
            self.examples.append((question, options, answer_key))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        question, options, answer_key = self.examples[idx]
        options_str = "\n".join([f"{lbl}) {txt}" for lbl, txt in options])
        prompt = (
            f"Question: {question}\n"
            "Options:\n"
            f"{options_str}\n"
            + ANSWER_DELIM
        )

        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=answer_key,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.40,
            add_bos=True,
            add_eos=True,
        )

class MMLUProLetterDataset(Dataset):
    """
    MMLU-Pro multiple-choice framed as predicting a single option letter.
    Typically 10-way (A-J), but we allow variable option counts.
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "test",
        max_length: int = 256,
        max_samples: int = 1000,
        answer_loss_tokens: int = 4,
        dataset_name: str = "TIGER-Lab/MMLU-Pro",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        ds = load_dataset(dataset_name, split=split)
        self.examples = []

        for ex in ds:
            q = (ex.get("question") or "").strip()
            options = ex.get("options") or ex.get("choices") or ex.get("answer_choices")
            ans = ex.get("answer")

            if not q or not options or ans is None:
                continue

            # Normalize options to a list of strings.
            if isinstance(options, dict):
                # Try common dict formats: {"text":[...], "label":[...]} or {"A":"..", ...}
                if ("text" in options) and isinstance(options.get("text"), list):
                    options = options.get("text") or []
                else:
                    # stable order by key if it looks like A/B/C...
                    keys = list(options.keys())
                    keys_sorted = sorted(keys)
                    options = [options[k] for k in keys_sorted]

            if not isinstance(options, list) or len(options) < 2:
                continue

            letters = [chr(ord("A") + i) for i in range(len(options))]
            opt_pairs = [(letters[i], str(options[i]).strip()) for i in range(len(options))]

            answer_key = None
            if isinstance(ans, int):
                if 0 <= ans < len(letters):
                    answer_key = letters[ans]
            else:
                ans_s = str(ans).strip()
                if ans_s.isdigit():
                    idx = int(ans_s)
                    if 0 <= idx < len(letters):
                        answer_key = letters[idx]
                elif ans_s in letters:
                    answer_key = ans_s
                else:
                    # Sometimes answer is the option text; try to match.
                    ans_l = ans_s.lower()
                    for i, _opt in enumerate(options):
                        if str(_opt).strip().lower() == ans_l:
                            answer_key = letters[i]
                            break

            if answer_key is None:
                continue

            self.examples.append((q, opt_pairs, answer_key))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        question, options, answer_key = self.examples[idx]
        options_str = "\n".join([f"{lbl}) {txt}" for lbl, txt in options])
        prompt = (
            f"Question: {question}\n"
            "Options:\n"
            f"{options_str}\n"
            + ANSWER_DELIM
        )
        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=answer_key,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.40,
            add_bos=True,
            add_eos=True,
        )


class GPQADiamondLetterDataset(Dataset):
    """
    GPQA-diamond multiple-choice framed as predicting a single option letter (A-D).
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "train",
        max_length: int = 256,
        max_samples: int = 1000,
        answer_loss_tokens: int = 4,
        dataset_name: str = "jinulee-v/gpqa-diamond",
        dataset_config_name: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        if dataset_config_name:
            ds = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        self.examples = []
        for ex in ds:
            q = (ex.get("question") or "").strip()
            options = ex.get("choices") or ex.get("options")
            ans = ex.get("answer") or ex.get("label") or ex.get("answer_key") or ex.get("correct")

            if not q:
                continue

            opt_pairs = None
            if options is not None:
                if isinstance(options, dict):
                    texts = options.get("text")
                    labels = options.get("label")
                    if isinstance(texts, list) and isinstance(labels, list) and len(texts) == len(labels):
                        opt_pairs = [(str(lbl).strip(), str(txt).strip()) for lbl, txt in zip(labels, texts)]
                    else:
                        # dict like {"A": "...", ...}
                        keys = sorted(list(options.keys()))
                        opt_pairs = [(str(k).strip(), str(options[k]).strip()) for k in keys]
                elif isinstance(options, list):
                    letters = [chr(ord("A") + i) for i in range(len(options))]
                    opt_pairs = [(letters[i], str(options[i]).strip()) for i in range(len(options))]

            answer_key = None
            if ans is not None:
                if isinstance(ans, int):
                    if 0 <= ans <= 25:
                        answer_key = chr(ord("A") + ans)
                else:
                    ans_s = str(ans).strip()
                    if ans_s.isdigit():
                        idx = int(ans_s)
                        if 0 <= idx <= 25:
                            answer_key = chr(ord("A") + idx)
                    else:
                        answer_key = ans_s

            # If answer_key is not a letter, try to match by option text.
            if answer_key and opt_pairs:
                letters_present = [lbl for lbl, _ in opt_pairs]
                if answer_key not in letters_present:
                    ans_l = str(answer_key).strip().lower()
                    for lbl, txt in opt_pairs:
                        if str(txt).strip().lower() == ans_l:
                            answer_key = lbl
                            break

            if not answer_key or answer_key not in {"A", "B", "C", "D"}:
                continue

            # Some GPQA variants embed options inside the question string. Allow missing opt_pairs.
            if not opt_pairs:
                opt_pairs = [("A", ""), ("B", ""), ("C", ""), ("D", "")]

            self.examples.append((q, opt_pairs, answer_key))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        question, options, answer_key = self.examples[idx]
        options_str = "\n".join([f"{lbl}) {txt}" for lbl, txt in options if txt])
        prompt = f"Question: {question}\n"
        if options_str:
            prompt += "Options:\n" + options_str + "\n"
        prompt += ANSWER_DELIM

        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=answer_key,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.40,
            add_bos=True,
            add_eos=True,
        )


class AIMEShortAnswerDataset(Dataset):
    """
    AIME-style math problems framed as short answer generation.
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "train",
        max_length: int = 256,
        max_samples: int = 30,
        answer_loss_tokens: int = 8,
        dataset_name: str = "HuggingFaceH4/aime_2024",
        dataset_config_name: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        if dataset_config_name:
            ds = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        self.examples = []
        for ex in ds:
            q = (ex.get("problem") or ex.get("question") or "").strip()
            a = (str(ex.get("answer") or "")).strip()
            if not q or not a:
                continue
            prompt = (
                "Solve the problem. Answer with a single integer.\n"
                f"Problem: {q}\n"
                + ANSWER_DELIM
            )
            self.examples.append((prompt, a))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        prompt, answer = self.examples[idx]
        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=answer,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.55,
            add_bos=True,
            add_eos=True,
        )


class HotpotQADataset(Dataset):
    """
    HotpotQA 2-hop QA framed as extractive QA (loss-only).
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "validation",
        max_length: int = 320,
        max_samples: int = 1000,
        answer_loss_tokens: int = 16,
        dataset_name: str = "hotpotqa/hotpot_qa",
        dataset_config_name: Optional[str] = "distractor",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        if dataset_config_name:
            ds = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        self.examples = []
        for ex in ds:
            q = (ex.get("question") or "").strip()
            a = (ex.get("answer") or "").strip()
            ctx = ex.get("context")
            if not q or not a or not ctx:
                continue

            chunks: List[str] = []
            if isinstance(ctx, dict):
                titles = ctx.get("title") or []
                sents = ctx.get("sentences") or []
                for t, ss in zip(titles, sents):
                    ss_txt = " ".join([str(s).strip() for s in (ss or []) if s])
                    if ss_txt:
                        chunks.append(f"[{t}] {ss_txt}")
            elif isinstance(ctx, list):
                # Sometimes it's a list of (title, [sentences]) pairs.
                for item in ctx:
                    if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    t, ss = item[0], item[1]
                    ss_txt = " ".join([str(s).strip() for s in (ss or []) if s])
                    if ss_txt:
                        chunks.append(f"[{t}] {ss_txt}")

            context_text = "\n".join(chunks)
            if not context_text:
                continue

            context_text = _truncate_text_head_tail(context_text, max_chars=7000, head_chars=5000, tail_chars=1500)
            self.examples.append((context_text, q, a))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        context, question, answer = self.examples[idx]
        return build_extractive_qa_example(
            tokenizer=self.tokenizer,
            context=context,
            question=question,
            answer=answer,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            add_bos=True,
            add_eos=True,
        )


def _iter_hotpot_context_sections(raw_context: Any) -> List[Tuple[str, List[str]]]:
    """Normalize HotpotQA context payloads into (title, [sentences]) sections."""
    sections: List[Tuple[str, List[str]]] = []
    if isinstance(raw_context, dict):
        titles = raw_context.get("title") or []
        sentences = raw_context.get("sentences") or []
        for title, sent_list in zip(titles, sentences):
            title_s = str(title).strip()
            cleaned = [str(sent).strip() for sent in (sent_list or []) if str(sent).strip()]
            if title_s and cleaned:
                sections.append((title_s, cleaned))
        return sections

    if isinstance(raw_context, list):
        for item in raw_context:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = str(item[0]).strip()
            sent_list = item[1]
            cleaned = [str(sent).strip() for sent in (sent_list or []) if str(sent).strip()]
            if title and cleaned:
                sections.append((title, cleaned))
    return sections


def _hotpot_support_fact_keys(raw_supporting_facts: Any) -> List[Tuple[str, int]]:
    """Normalize HotpotQA supporting-facts payloads into (title, sent_id) pairs."""
    pairs: List[Tuple[str, int]] = []
    if isinstance(raw_supporting_facts, dict):
        titles = raw_supporting_facts.get("title") or []
        sent_ids = raw_supporting_facts.get("sent_id") or raw_supporting_facts.get("sent_ids") or []
        for title, sent_id in zip(titles, sent_ids):
            title_s = str(title).strip()
            try:
                sent_idx = int(sent_id)
            except (TypeError, ValueError):
                continue
            if title_s and sent_idx >= 0:
                pairs.append((title_s, sent_idx))
        return pairs

    if isinstance(raw_supporting_facts, list):
        for item in raw_supporting_facts:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title_s = str(item[0]).strip()
            try:
                sent_idx = int(item[1])
            except (TypeError, ValueError):
                continue
            if title_s and sent_idx >= 0:
                pairs.append((title_s, sent_idx))
    return pairs


def _hotpot_supporting_facts_from_sections(
    sections: List[Tuple[str, List[str]]],
    raw_supporting_facts: Any,
    *,
    max_facts: int = 2,
    max_fact_chars: int = 160,
) -> List[str]:
    """Resolve gold supporting-fact strings from HotpotQA context + indices."""
    support_keys = _hotpot_support_fact_keys(raw_supporting_facts)
    if not support_keys:
        return []

    title_to_sents: Dict[str, List[str]] = {}
    for title, sent_list in sections:
        title_to_sents[title] = list(sent_list)

    facts: List[str] = []
    seen = set()
    for title, sent_idx in support_keys:
        sentences = title_to_sents.get(title)
        if not sentences or not (0 <= sent_idx < len(sentences)):
            continue
        fact_text = sentences[sent_idx].strip()
        if max_fact_chars > 0 and len(fact_text) > max_fact_chars:
            fact_text = fact_text[:max_fact_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
        fact = f"[{title}] {fact_text}"
        fact_norm = _normalize_eval_text(fact)
        if not fact_norm or fact_norm in seen:
            continue
        seen.add(fact_norm)
        facts.append(fact)
        if len(facts) >= max_facts:
            break
    return facts


def _normalize_hotpot_reasoning_output_contract(output_contract: Optional[str]) -> str:
    """Canonicalize the eval-only Hotpot reasoning output contract name."""
    return _shared_normalize_hotpot_reasoning_output_contract(output_contract)


def _normalize_hotpot_reasoning_prompt_variant(prompt_variant: Optional[str]) -> str:
    return _shared_normalize_hotpot_reasoning_prompt_variant(prompt_variant)


def _hotpot_reasoning_prompt_prefix(output_contract: str, *, prompt_variant: Optional[str] = None) -> str:
    """Render the Hotpot prompt instructions for one output contract."""
    return _shared_hotpot_reasoning_prompt_prefix(output_contract, prompt_variant=prompt_variant)


def _normalize_hotpot_reasoning_response_cue_variant(response_cue_variant: Optional[str]) -> str:
    return _shared_normalize_hotpot_reasoning_response_cue_variant(response_cue_variant)


def _hotpot_reasoning_response_cue(output_contract: str, *, response_cue_variant: Optional[str] = None) -> str:
    return _shared_hotpot_reasoning_response_cue(output_contract, response_cue_variant=response_cue_variant)


def _hotpot_reasoning_target_continuation_text(
    gold_supporting_facts: List[str],
    answer: str,
    *,
    output_contract: str,
    response_cue_variant: Optional[str] = None,
) -> str:
    return _shared_hotpot_reasoning_target_continuation_text(
        gold_supporting_facts,
        answer,
        output_contract=output_contract,
        response_cue_variant=response_cue_variant,
    )


def _hotpot_reasoning_target_lines(
    gold_supporting_facts: List[str],
    answer: str,
    *,
    output_contract: str,
) -> List[str]:
    """Render the supervised target for one Hotpot reasoning output contract."""
    return _shared_hotpot_reasoning_target_lines(
        gold_supporting_facts,
        answer,
        output_contract=output_contract,
    )


class HotpotQAReasoningDataset(Dataset):
    """
    HotpotQA reasoning eval with a lightly structured output:

    Supporting facts:
    - <fact 1>
    - <fact 2>
    Final answer: <answer>

    Source ownership is recorded in metadata so context-pull probing does not
    need to recover the evidence span from prompt markers.
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "validation",
        max_length: int = 512,
        max_samples: int = 128,
        answer_loss_tokens: int = 96,
        dataset_name: str = "hotpotqa/hotpot_qa",
        dataset_config_name: Optional[str] = "distractor",
        output_contract: str = "default",
        prompt_variant: str = "default",
        response_cue_variant: str = "default",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens
        self.output_contract = _normalize_hotpot_reasoning_output_contract(output_contract)
        self.prompt_variant = _normalize_hotpot_reasoning_prompt_variant(prompt_variant)
        self.response_cue_variant = _normalize_hotpot_reasoning_response_cue_variant(response_cue_variant)

        if dataset_config_name:
            ds = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        self.examples = []
        for ex in ds:
            question = (ex.get("question") or "").strip()
            answer = (ex.get("answer") or "").strip()
            sections = _iter_hotpot_context_sections(ex.get("context"))
            gold_supporting_facts = _hotpot_supporting_facts_from_sections(sections, ex.get("supporting_facts"))
            sample_id = ex.get("id") or ex.get("_id") or ex.get("sample_id")

            if not question or not answer or not sections or not gold_supporting_facts:
                continue

            support_titles = {title for title, _sent_idx in _hotpot_support_fact_keys(ex.get("supporting_facts"))}
            ordered_sections = [section for section in sections if section[0] in support_titles]
            ordered_sections.extend(section for section in sections if section[0] not in support_titles)

            section_blocks: List[str] = []
            for title, sent_list in ordered_sections:
                sentence_lines = "\n".join(f"- {sent.strip()}" for sent in sent_list if sent.strip())
                if sentence_lines:
                    section_blocks.append(f"[{title}]\n{sentence_lines}")

            evidence_text = "\n\n".join(section_blocks)
            if not evidence_text:
                continue
            evidence_text = _truncate_text_head_tail(
                evidence_text,
                max_chars=9000,
                head_chars=7000,
                tail_chars=1500,
            )

            key_fact_meta = select_hotpot_key_facts(gold_supporting_facts)

            self.examples.append(
                {
                    "sample_id": str(sample_id) if sample_id is not None else f"hotpotqa_reasoning_{len(self.examples)}",
                    "question": question,
                    "answer": answer,
                    "gold_supporting_facts": gold_supporting_facts,
                    **key_fact_meta,
                    "evidence_text": evidence_text,
                }
            )
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        prompt_prefix = _hotpot_reasoning_prompt_prefix(self.output_contract, prompt_variant=self.prompt_variant)
        response_cue = _hotpot_reasoning_response_cue(
            self.output_contract,
            response_cue_variant=self.response_cue_variant,
        )
        source_start = len(prompt_prefix)
        source_end = source_start + len(item["evidence_text"])
        prompt = (
            prompt_prefix
            + item["evidence_text"]
            + f"\nQuestion: {item['question']}\n"
            + response_cue
        )
        target = _hotpot_reasoning_target_continuation_text(
            list(item["gold_supporting_facts"]),
            item["answer"],
            output_contract=self.output_contract,
            response_cue_variant=self.response_cue_variant,
        )

        ex = build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=target,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.50,
            add_bos=True,
            add_eos=True,
        )
        task_format = "structured_text"
        if self.output_contract == "stepwise":
            task_format = "structured_text_stepwise"
        elif self.output_contract == "keyfacts":
            task_format = "structured_text_keyfacts"

        ex["meta"] = {
            "sample_id": item["sample_id"],
            "question_text": item["question"],
            "gold_answer": item["answer"],
            "gold_supporting_facts": list(item["gold_supporting_facts"]),
            "gold_key_fact_1": item.get("gold_key_fact_1", ""),
            "gold_key_fact_2": item.get("gold_key_fact_2", ""),
            "gold_key_facts_selected": list(item.get("gold_key_facts_selected") or []),
            "gold_key_fact_2_is_copy": bool(item.get("gold_key_fact_2_is_copy")),
            "task_format": task_format,
            "benchmark_role": "basin_probe",
            "reasoning_eval_mode": True,
            "allow_abstain": False,
            "abstain_mode": "none",
            "abstain_text": None,
            "abstain_option_letter": None,
            "supports_concrete_answer": True,
            "hotpotqa_reasoning_output_contract": self.output_contract,
            "hotpotqa_reasoning_prompt_variant": self.prompt_variant,
            "hotpotqa_reasoning_response_cue_variant": self.response_cue_variant,
            "hotpotqa_reasoning_response_cue_text": response_cue,
            "context_pull_source_char_span": [source_start, source_end],
            "context_pull_source_text": item["evidence_text"],
            "context_pull_source_kind": "evidence",
        }
        return ex


class TruthfulQAMCDataset(Dataset):
    """
    TruthfulQA multiple-choice framed as predicting a single option letter (A-D).

    Optionally appends a legacy abstain option (E) for calibration sweeps.
    This path is kept for backward compatibility; it is not the default v4
    abstention wording.
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "validation",
        max_length: int = 256,
        max_samples: int = 817,
        answer_loss_tokens: int = 4,
        add_abstain: bool = True,
        dataset_name: str = "EleutherAI/truthful_qa_mc",
        dataset_config_name: str = "multiple_choice",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens
        self.add_abstain = add_abstain

        ds = load_dataset(dataset_name, dataset_config_name, split=split)
        self.examples = []
        for ex in ds:
            q = (ex.get("question") or "").strip()
            choices = ex.get("choices") or ex.get("options") or []
            label = ex.get("label")

            if not q or not choices or label is None:
                continue

            if isinstance(choices, dict):
                if ("text" in choices) and isinstance(choices.get("text"), list):
                    choices = choices.get("text") or []
                else:
                    keys = sorted(list(choices.keys()))
                    choices = [choices[k] for k in keys]

            if not isinstance(choices, list) or len(choices) != 4:
                continue

            letters = ["A", "B", "C", "D"]
            opt_pairs = [(letters[i], str(choices[i]).strip()) for i in range(4)]
            if self.add_abstain:
                opt_pairs.append(("E", LEGACY_TRUTHFULQA_ABSTAIN_TEXT))

            answer_key = None
            if isinstance(label, int):
                if 0 <= label < 4:
                    answer_key = letters[label]
            else:
                label_s = str(label).strip()
                if label_s in letters:
                    answer_key = label_s

            if answer_key is None:
                continue

            self.examples.append((q, opt_pairs, answer_key))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        q, options, answer_key = self.examples[idx]
        options_str = "\n".join([f"{lbl}) {txt}" for lbl, txt in options])
        prompt = (
            f"Question: {q}\n"
            "Options:\n"
            f"{options_str}\n"
            + ANSWER_DELIM
        )
        ex = build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=answer_key,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.40,
            add_bos=True,
            add_eos=True,
        )
        ex["meta"] = {
            "sample_id": f"truthfulqa_{idx}",
            "gold_label": answer_key,
            "task_format": "mc_letter",
            "benchmark_role": "basin_probe",
            "reasoning_eval_mode": False,
            "allow_abstain": bool(self.add_abstain),
            "abstain_mode": "extra_option" if self.add_abstain else "none",
            "abstain_option_letter": "E" if self.add_abstain else None,
            "abstain_text": LEGACY_TRUTHFULQA_ABSTAIN_TEXT if self.add_abstain else None,
            "supports_concrete_answer": True,
        }
        return ex


class SummarizationDataset(Dataset):
    """
    XSum-style abstractive summarization framed for evaluation-only loss.
    Loss is focused on the first few tokens of the summary to keep it cheap.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "validation",
        max_length: int = 384,
        max_samples: int = 200,
        answer_loss_tokens: int = 32,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        ds = load_dataset("xsum", split=split)
        self.examples = []
        for ex in ds:
            doc = (ex.get("document") or "").strip()
            summ = (ex.get("summary") or "").strip()
            if not doc or not summ:
                continue
            self.examples.append((doc, summ))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        document, summary = self.examples[idx]
        # Keep tokenization cost bounded for long articles.
        document = _truncate_text_head_tail(document, max_chars=6000, head_chars=3500, tail_chars=2000)

        prompt = (
            "Summarize the article.\n"
            "ARTICLE:\n"
            f"{document}\n"
            + SUMMARY_DELIM
        )

        return build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=summary,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.30,
            add_bos=True,
            add_eos=True,
        )

class ExtractiveQADataset(Dataset):
    """
    Extractive QA (SQuAD-style) for cross-task evaluation.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "validation",
        max_length: int = 384,
        max_samples: int = 200,
        answer_loss_tokens: int = 16,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        ds = load_dataset("squad", split=split)
        self.examples = []
        for ex in ds:
            ctx = (ex.get("context") or "").strip()
            q = (ex.get("question") or "").strip()
            ans_list = ex.get("answers", {}).get("text") or []
            answer = ans_list[0].strip() if ans_list else ""
            if not ctx or not q or not answer:
                continue
            self.examples.append((ctx, q, answer))
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        context, question, answer = self.examples[idx]
        # Try to keep a window of context around the answer to make the eval meaningful.
        ctx_l = context.lower()
        ans_l = answer.lower()
        pos = ctx_l.find(ans_l)
        if pos != -1:
            start = max(0, pos - 600)
            end = min(len(context), pos + len(answer) + 600)
            context = context[start:end]
        else:
            context = _truncate_text_head_tail(context, max_chars=2400, head_chars=1600, tail_chars=800)

        return build_extractive_qa_example(
            tokenizer=self.tokenizer,
            context=context,
            question=question,
            answer=answer,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            add_bos=True,
            add_eos=True,
        )


class SQuAD2ReasoningDataset(Dataset):
    """
    SQuAD 2.0-style reasoning eval:
    - answer when supported by the provided context
    - otherwise emit a canonical context-grounded abstain string
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "validation",
        max_length: int = 384,
        max_samples: int = 200,
        answer_loss_tokens: int = 16,
        abstain_text: str = DEFAULT_CONTEXT_ABSTAIN_TEXT,
        dataset_name: str = "rajpurkar/squad_v2",
        build_insufficiency_negatives: bool = False,
        insufficiency_negative_max_candidates: int = 8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens
        self.abstain_text = abstain_text
        self.build_insufficiency_negatives = bool(build_insufficiency_negatives and "train" in str(split).lower())
        self.insufficiency_negative_max_candidates = max(0, int(insufficiency_negative_max_candidates))
        self.candidate_cleanup_rejected_count = 0
        self.candidate_cleanup_rejected_examples: List[str] = []

        ds = load_dataset(dataset_name, split=split)
        self.examples = []
        answer_bank: List[str] = []
        for ex in ds:
            context = (ex.get("context") or "").strip()
            question = (ex.get("question") or "").strip()
            sample_id = ex.get("id")
            answers = ex.get("answers") or {}
            gold_answers = []
            if isinstance(answers, dict):
                for answer in answers.get("text") or []:
                    answer_s = str(answer).strip()
                    if answer_s:
                        gold_answers.append(answer_s)
                        answer_bank.append(answer_s)
            if not context or not question:
                continue
            is_answerable = len(gold_answers) > 0
            self.examples.append(
                {
                    "sample_id": str(sample_id) if sample_id is not None else f"squad2_{len(self.examples)}",
                    "context": context,
                    "question": question,
                    "gold_answers": gold_answers,
                    "is_answerable": is_answerable,
                }
            )
            if len(self.examples) >= max_samples:
                break
        self.answer_bank = self._dedupe_candidates(answer_bank)
        if self.build_insufficiency_negatives and self.insufficiency_negative_max_candidates > 0:
            for idx, item in enumerate(self.examples):
                if bool(item.get("is_answerable")):
                    continue
                records = self._negative_candidate_records_for_unanswerable(idx)
                item["insufficiency_negative_candidate_texts"] = [record["text"] for record in records]
                item["insufficiency_negative_candidate_sources"] = [record["source"] for record in records]
                item["insufficiency_negative_candidate_source"] = (
                    "squad2_train_context_spans+question_overlap+answer_bank"
                )

    def __len__(self):
        return len(self.examples)

    def _dedupe_candidates(self, candidates: List[str]) -> List[str]:
        records = [{"text": candidate, "source": "answer_bank"} for candidate in candidates]
        cleaned = dedupe_candidate_records(records, abstain_text=self.abstain_text)
        self.candidate_cleanup_rejected_count += int(cleaned.get("rejected_count") or 0)
        for example in cleaned.get("rejected_examples") or []:
            if len(self.candidate_cleanup_rejected_examples) < 24:
                self.candidate_cleanup_rejected_examples.append(str(example))
        return [str(record["text"]) for record in cleaned["records"]]

    def _negative_candidates_for_unanswerable(self, idx: int) -> List[str]:
        return [record["text"] for record in self._negative_candidate_records_for_unanswerable(idx)]

    def _negative_candidate_records_for_unanswerable(self, idx: int) -> List[Dict[str, str]]:
        item = self.examples[idx]
        context_records = self._context_negative_candidate_records(
            str(item.get("context") or ""),
            str(item.get("question") or ""),
        )
        if not self.answer_bank and not context_records:
            return []
        records: List[Dict[str, str]] = list(context_records)
        if self.answer_bank:
            start = idx % len(self.answer_bank)
            rotated = self.answer_bank[start:] + self.answer_bank[:start]
            records.extend({"text": text, "source": "answer_bank"} for text in rotated)
        cleaned = dedupe_candidate_records(records, abstain_text=self.abstain_text)
        self.candidate_cleanup_rejected_count += int(cleaned.get("rejected_count") or 0)
        for example in cleaned.get("rejected_examples") or []:
            if len(self.candidate_cleanup_rejected_examples) < 24:
                self.candidate_cleanup_rejected_examples.append(str(example))
        return list(cleaned["records"])[: self.insufficiency_negative_max_candidates]

    def _context_negative_candidates(self, context: str) -> List[str]:
        return [record["text"] for record in self._context_negative_candidate_records(context, "")]

    def _context_negative_candidate_records(self, context: str, question: str) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        for match in re.finditer(
            r"\b[A-Z][A-Za-z0-9'’-]*(?:\s+(?:of|the|and|de|la|van|von|[A-Z][A-Za-z0-9'’-]*)){0,7}",
            context,
        ):
            text = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:()[]{}")
            if len(text) >= 3:
                records.append({"text": text, "source": "proper_noun_span"})
        for match in re.finditer(
            r"\b(?:1[5-9]\d{2}|20\d{2}|\d+(?:\.\d+)?|(?:\d{1,2}(?:st|nd|rd|th)\s+century))\b",
            context,
            flags=re.IGNORECASE,
        ):
            records.append({"text": match.group(0), "source": "date_or_number"})

        terms = set(question_terms(question))
        if terms:
            token_matches = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’-]*", context))
            tokens = [match.group(0) for match in token_matches]
            for idx, token in enumerate(tokens):
                if token.casefold() not in terms:
                    continue
                left = max(0, idx - 3)
                right = min(len(tokens), idx + 6)
                window = " ".join(tokens[left:right])
                records.append({"text": window, "source": "question_overlap_window"})
                for span_left in range(max(left, idx - 2), idx + 1):
                    span_right = min(right, span_left + 5)
                    span = " ".join(tokens[span_left:span_right])
                    records.append({"text": span, "source": "short_noun_like_span"})
        return list(dedupe_candidate_records(records, abstain_text=self.abstain_text)["records"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        context = item["context"]
        question = item["question"]
        gold_answers = list(item["gold_answers"])
        is_answerable = bool(item["is_answerable"])
        target = gold_answers[0] if is_answerable else self.abstain_text

        if is_answerable:
            context = _crop_context_around_answers(
                context,
                gold_answers,
                window_before=700,
                window_after=700,
                fallback_max_chars=2800,
                fallback_head_chars=1800,
                fallback_tail_chars=900,
            )
        else:
            context = _truncate_text_head_tail(context, max_chars=2800, head_chars=1800, tail_chars=900)

        prompt = (
            "Answer the question using only the provided context.\n"
            "If the answer cannot be inferred from the context, respond exactly with:\n"
            f"{self.abstain_text}\n"
            "Context:\n"
            f"{context}\n"
            f"Question: {question}\n"
            + ANSWER_DELIM
        )
        ex = build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=target,
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.35,
            add_bos=True,
            add_eos=True,
        )
        ex["meta"] = {
            "sample_id": item["sample_id"],
            "task_name": "squad2_reasoning",
            "is_answerable": is_answerable,
            "gold_answers": gold_answers,
            "answer_candidate_texts": gold_answers,
            "insufficiency_negative_candidate_texts": list(item.get("insufficiency_negative_candidate_texts") or []),
            "insufficiency_negative_candidate_sources": list(item.get("insufficiency_negative_candidate_sources") or []),
            "insufficiency_negative_candidate_source": str(item.get("insufficiency_negative_candidate_source") or ""),
            "candidate_cleanup_rejected_count": int(self.candidate_cleanup_rejected_count),
            "candidate_cleanup_rejected_examples": list(self.candidate_cleanup_rejected_examples[:12]),
            "abstain_target": not is_answerable,
            "task_format": "text_generation",
            "benchmark_role": "reasoning_measure",
            "reasoning_eval_mode": True,
            "allow_abstain": True,
            "abstain_mode": "text",
            "abstain_text": self.abstain_text,
            "abstain_option_letter": None,
            "supports_concrete_answer": is_answerable,
        }
        return ex


class FEVERReasoningDataset(Dataset):
    """
    FEVER-style claim/evidence reasoning eval with a native insufficient-evidence label.
    """

    LABEL_MAP = {
        "SUPPORTS": ("A", "supported", True),
        "REFUTES": ("B", "refuted", True),
        "NOT ENOUGH INFO": ("C", "not_enough_info", False),
    }

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "dev",
        max_length: int = 320,
        max_samples: int = 500,
        answer_loss_tokens: int = 4,
        dataset_name: str = "pietrolesci/nli_fever",
        dataset_config_name: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_loss_tokens = answer_loss_tokens

        if dataset_config_name:
            ds = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        self.examples = []
        for ex in ds:
            claim = (ex.get("hypothesis") or ex.get("claim") or "").strip()
            evidence = (ex.get("premise") or ex.get("evidence") or ex.get("context") or "").strip()
            raw_label = ex.get("fever_gold_label") or ex.get("label") or ex.get("gold_label")
            sample_id = ex.get("id") or ex.get("cid") or ex.get("sample_id")

            if not claim or not evidence or raw_label is None:
                continue

            label_key = str(raw_label).strip().upper().replace("_", " ")
            if label_key.isdigit():
                label_idx = int(label_key)
                idx_to_label = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
                if 0 <= label_idx < len(idx_to_label):
                    label_key = idx_to_label[label_idx]
            if label_key not in self.LABEL_MAP:
                continue

            answer_key, reasoning_label, supports_concrete_answer = self.LABEL_MAP[label_key]
            self.examples.append(
                {
                    "sample_id": str(sample_id) if sample_id is not None else f"fever_{len(self.examples)}",
                    "claim": claim,
                    "evidence": evidence,
                    "answer_key": answer_key,
                    "reasoning_label": reasoning_label,
                    "supports_concrete_answer": supports_concrete_answer,
                }
            )
            if len(self.examples) >= max_samples:
                break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        evidence = _truncate_text_head_tail(item["evidence"], max_chars=5000, head_chars=3200, tail_chars=1400)
        prompt = (
            "Decide whether the claim is supported, refuted, or cannot be determined from the provided evidence.\n"
            f"Claim: {item['claim']}\n"
            "Evidence:\n"
            f"{evidence}\n"
            "Options:\n"
            "A) Supported\n"
            "B) Refuted\n"
            f"C) {FEVER_REASONING_NEI_TEXT}\n"
            + ANSWER_DELIM
        )
        ex = build_sft_example(
            tokenizer=self.tokenizer,
            prompt=prompt,
            answer=item["answer_key"],
            max_length=self.max_length,
            answer_loss_tokens=self.answer_loss_tokens,
            prompt_head_ratio=0.40,
            add_bos=True,
            add_eos=True,
        )
        ex["meta"] = {
            "sample_id": item["sample_id"],
            "task_name": "fever_reasoning",
            "gold_label": item["answer_key"],
            "reasoning_label": item["reasoning_label"],
            "abstain_target": not item["supports_concrete_answer"],
            "task_format": "mc_letter",
            "benchmark_role": "reasoning_measure",
            "reasoning_eval_mode": True,
            "allow_abstain": True,
            "abstain_mode": "native_label",
            "abstain_text": FEVER_REASONING_NEI_TEXT,
            "abstain_option_letter": "C",
            "supports_concrete_answer": item["supports_concrete_answer"],
        }
        return ex

@dataclass
class Batch:
    """Padded batch for SFT-style supervision plus optional per-example metadata."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_len: torch.Tensor
    meta: Optional[List[Dict[str, Any]]] = None


def collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Batch:
    """Pad token fields and preserve optional example metadata across batching."""
    B = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    prompt_len = torch.tensor([x["prompt_len"] for x in batch], dtype=torch.long)
    meta = None

    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(ex["labels"], dtype=torch.long)
        attention_mask[i, :L] = 1
    if any(ex.get("meta") is not None for ex in batch):
        meta = []
        for ex in batch:
            item_meta = ex.get("meta")
            if isinstance(item_meta, dict):
                meta.append(dict(item_meta))
            else:
                meta.append({})

    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels, prompt_len=prompt_len, meta=meta)


# -------------------------
# 2) REAL head (thought space + intent shaping + dynamic basin + energy)
# -------------------------

class REALHead(nn.Module):
    """
    Input: pooled prompt embedding h in R^{d_model}
    Output:
      - final prefix embeddings P in R^{prefix_len x d_model}
      - predicted energies e_t per state along refinement trajectory
      - z trajectory (intent latent states) for stepwise energy supervision

    Upgrades implemented:
      (1) Stepwise *true* energy supervision is done in the wrapper using z_traj.
      (2) Update is "resonance-aware": update sees predicted energy and its delta.
      (3) Adaptive refinement policy: early stopping is available via refine_adaptive().
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 256,
        prefix_len: int = 16,
        num_steps: int = 4,
        num_particles: int = 4,     # thought space candidates
        init_noise: float = 0.25,
        step_scale: float = 0.35,
        step_embed_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.prefix_len = prefix_len
        self.num_steps = num_steps
        self.num_particles = num_particles
        self.init_noise = init_noise
        self.step_scale = step_scale

        self.step_embed = nn.Embedding(num_steps + 1, step_embed_dim)

        self.h_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
            nn.Tanh(),
        )

        # energy(z,h,step) : scalar
        self.energy_head = nn.Sequential(
            nn.LayerNorm(latent_dim + d_model + step_embed_dim),
            nn.Linear(latent_dim + d_model + step_embed_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 1),
        )

        # update(z,h, e, de, step) : delta_z
        upd_in_dim = latent_dim + d_model + step_embed_dim + 2  # + e + de
        self.update = nn.Sequential(
            nn.LayerNorm(upd_in_dim),
            nn.Linear(upd_in_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

        # z -> prefix embeddings
        self.prefix_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, prefix_len * d_model),
        )

    def _energy(self, z: torch.Tensor, h: torch.Tensor, step_idx: int) -> torch.Tensor:
        B = z.size(0)
        step = torch.full((B,), step_idx, device=z.device, dtype=torch.long)
        se = self.step_embed(step)
        x = torch.cat([z, h, se], dim=-1)
        raw = self.energy_head(x).squeeze(-1)
        return F.softplus(raw)  # [B], >= 0 for stable pressure control

    def _update(self, z: torch.Tensor, h: torch.Tensor, e: torch.Tensor, de: torch.Tensor, step_idx: int) -> torch.Tensor:
        B = z.size(0)
        step = torch.full((B,), step_idx, device=z.device, dtype=torch.long)
        se = self.step_embed(step)
        x = torch.cat([z, h, se, e.unsqueeze(-1), de.unsqueeze(-1)], dim=-1)
        return self.update(x)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: [B, d_model]

        Returns:
          prefix_final: [B, prefix_len, d_model]
          energies:     [B, num_steps+1]  energies for states z_0 ... z_S
          z_traj:       [B, num_steps+1, latent_dim]  z_0 ... z_S
        """
        B = h.size(0)

        # ---- thought space sampling (candidate z0) ----
        z0 = self.h_proj(h)  # [B, latent]
        if self.num_particles > 1:
            noise = torch.randn(B, self.num_particles, self.latent_dim, device=h.device) * self.init_noise
            zc = z0.unsqueeze(1) + noise  # [B, P, latent]
            hc = h.unsqueeze(1).expand(B, self.num_particles, self.d_model).reshape(B * self.num_particles, self.d_model)
            zc_flat = zc.reshape(B * self.num_particles, self.latent_dim)

            # energy at step 0 for selection (untrained early on is fine; it bootstraps)
            e0 = self._energy(zc_flat, hc, step_idx=0).view(B, self.num_particles)
            best = torch.argmin(e0, dim=1)
            z = zc[torch.arange(B, device=h.device), best]
        else:
            z = z0

        z_traj = [z]
        energies = []

        prev_e = None

        # ---- dynamic basin refinement ----
        for t in range(self.num_steps):
            e_t = self._energy(z, h, step_idx=t)  # energy of current state
            energies.append(e_t)

            if prev_e is None:
                de = torch.zeros_like(e_t)
            else:
                de = e_t - prev_e

            delta = self._update(z, h, e_t, de, step_idx=t)
            z = z + self.step_scale * torch.tanh(delta)

            z_traj.append(z)
            prev_e = e_t

        # final energy for z_S
        e_final = self._energy(z, h, step_idx=self.num_steps)
        energies.append(e_final)

        energies = torch.stack(energies, dim=1)     # [B, S+1]
        z_traj = torch.stack(z_traj, dim=1)         # [B, S+1, latent]
        prefix_final = self.prefix_proj(z).view(B, self.prefix_len, self.d_model)

        return prefix_final, energies, z_traj

    @torch.no_grad()
    def refine_adaptive(
        self,
        h: torch.Tensor,
        max_steps: Optional[int] = None,
        energy_delta_tol: float = 1e-3,
        patience: int = 1,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference-time adaptive refinement policy: accept/reject each proposed update based on predicted energy.

        Returns:
          prefix:      [B, prefix_len, d_model]
          energies:    [B, T] (variable T up to max_steps+1)
          z_traj:      [B, T, latent_dim]
          steps_used:  [B] number of update steps actually applied
        """
        if max_steps is None:
            max_steps = self.num_steps

        B = h.size(0)

        # initial z0 (no multi-particle selection here by default; keep deterministic-ish)
        z = self.h_proj(h)
        z_traj = [z]
        energies = []

        prev_e = None
        no_accept = torch.zeros((B,), device=h.device, dtype=torch.long)
        active = torch.ones((B,), device=h.device, dtype=torch.bool)
        steps_used = torch.zeros((B,), device=h.device, dtype=torch.long)
        attempts = torch.zeros((B,), device=h.device, dtype=torch.long)
        rejects = torch.zeros((B,), device=h.device, dtype=torch.long)

        for t in range(max_steps):
            step_idx = min(t, self.num_steps)

            # energy of current state
            e_before = self._energy(z, h, step_idx=step_idx)
            energies.append(e_before)

            # real Δe channel (match forward() semantics)
            de = torch.zeros_like(e_before) if prev_e is None else (e_before - prev_e)

            # propose update
            delta = self._update(z, h, e_before, de, step_idx=step_idx)
            z_prop = z + self.step_scale * torch.tanh(delta)

            # lookahead accept/reject (same step_idx to avoid "clock" drift)
            e_after = self._energy(z_prop, h, step_idx=step_idx)
            accept = ((e_before - e_after) > energy_delta_tol) & active

            attempts += active.long()
            rejects += (active & (~accept)).long()

            # commit only if accepted
            z = torch.where(accept.unsqueeze(-1), z_prop, z)
            z_traj.append(z)

            steps_used += accept.long()
            no_accept = torch.where(accept, torch.zeros_like(no_accept), no_accept + 1)
            active = active & (no_accept <= patience)

            prev_e = e_before
            if not active.any():
                break

        # final energy (step index aligned to how many steps we actually attempted)
        final_step_idx = min(len(energies), self.num_steps)
        e_final = self._energy(z, h, step_idx=final_step_idx)
        energies.append(e_final)

        energies_t = torch.stack(energies, dim=1)  # [B, T]
        z_t = torch.stack(z_traj, dim=1)          # [B, T, latent]
        prefix = self.prefix_proj(z).view(B, self.prefix_len, self.d_model)

        if return_stats:
            return prefix, energies_t, z_t, steps_used, attempts, rejects
        return prefix, energies_t, z_t, steps_used


# -------------------------
# 3) Wrapper: Frozen backbone + REAL head
# -------------------------

class REALWrapper(nn.Module):
    """
    Implements:
      - final-step LM loss (with gradients to the head via prefix conditioning)
      - stepwise TRUE energy computation (no_grad) for steps 0..S-1
      - energy head supervision across all states z_0..z_S
      - monotonicity penalty on predicted energy
    """

    def __init__(
        self,
        backbone: AutoModelForCausalLM,
        head: REALHead,
        energy_weight: float = 0.50,
        mono_weight: float = 0.02,
        tokenizer=None,
        insufficiency_contact_objective: bool = False,
        insufficiency_contact_weight: float = 0.0,
        insufficiency_contact_margin: float = 0.0,
        insufficiency_contact_steps: str = "final",
        insufficiency_contact_max_candidates: int = 2,
        insufficiency_contact_apply_to_supported: bool = True,
        insufficiency_contact_apply_to_insufficient: bool = True,
        insufficiency_contact_hard_negatives: bool = False,
        insufficiency_contact_hard_negative_pool_size: int = 8,
        insufficiency_contact_hard_negative_top_k: int = 1,
        insufficiency_contact_hard_negative_refresh_steps: int = 1,
        insufficiency_contact_hard_negative_selection_mode: str = "per_step",
        insufficiency_contact_negative_source: str = "train_split_bank",
        insufficiency_contact_loss_mode: str = "legacy",
        insufficiency_contact_tolerance: float = 0.25,
        insufficiency_contact_supported_guard_weight: float = 0.0,
        insufficiency_contact_supported_margin_floor: float = 0.25,
        insufficiency_contact_positive_nll_weight: float = 0.0,
        insufficiency_contact_path_allowed_drift: float = 0.0,
        insufficiency_contact_checkpoint_candidates: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.energy_weight = energy_weight
        self.mono_weight = mono_weight
        self.tokenizer = tokenizer
        self.insufficiency_contact_objective = bool(insufficiency_contact_objective)
        self.insufficiency_contact_weight = float(insufficiency_contact_weight)
        self.insufficiency_contact_margin = float(insufficiency_contact_margin)
        self.insufficiency_contact_steps = str(insufficiency_contact_steps or "final")
        self.insufficiency_contact_max_candidates = max(0, int(insufficiency_contact_max_candidates))
        self.insufficiency_contact_apply_to_supported = bool(insufficiency_contact_apply_to_supported)
        self.insufficiency_contact_apply_to_insufficient = bool(insufficiency_contact_apply_to_insufficient)
        self.insufficiency_contact_hard_negatives = bool(insufficiency_contact_hard_negatives)
        self.insufficiency_contact_hard_negative_pool_size = max(1, int(insufficiency_contact_hard_negative_pool_size))
        self.insufficiency_contact_hard_negative_top_k = max(1, int(insufficiency_contact_hard_negative_top_k))
        self.insufficiency_contact_hard_negative_refresh_steps = max(1, int(insufficiency_contact_hard_negative_refresh_steps))
        if self.insufficiency_contact_hard_negative_refresh_steps != 1:
            raise ValueError(
                "insufficiency_contact_hard_negative_refresh_steps > 1 is not implemented yet; use 1."
            )
        self.insufficiency_contact_hard_negative_selection_mode = str(
            insufficiency_contact_hard_negative_selection_mode or "per_step"
        )
        if self.insufficiency_contact_hard_negative_selection_mode not in {
            "per_step",
            "final_step",
            "entry_step",
            "max_over_steps_selected",
        }:
            raise ValueError(
                "insufficiency_contact_hard_negative_selection_mode must be one of: "
                "per_step, final_step, entry_step, max_over_steps_selected"
            )
        self.insufficiency_contact_negative_source = str(insufficiency_contact_negative_source or "train_split_bank")
        self.insufficiency_contact_loss_mode = str(insufficiency_contact_loss_mode or "legacy")
        self.insufficiency_contact_tolerance = float(insufficiency_contact_tolerance)
        self.insufficiency_contact_supported_guard_weight = float(insufficiency_contact_supported_guard_weight)
        self.insufficiency_contact_supported_margin_floor = float(insufficiency_contact_supported_margin_floor)
        self.insufficiency_contact_positive_nll_weight = float(insufficiency_contact_positive_nll_weight)
        self.insufficiency_contact_path_allowed_drift = float(insufficiency_contact_path_allowed_drift)
        self.insufficiency_contact_checkpoint_candidates = bool(insufficiency_contact_checkpoint_candidates)

    def train(self, mode: bool = True):
        """Override nn.Module.train() to keep the frozen backbone in eval mode.

        Even with requires_grad=False, enabling train mode on the backbone can re-enable dropout
        (and other training-time behaviors), which makes 'true energy' targets noisy and can
        destabilize comparisons.
        """
        super().train(mode)
        # Frozen basins must remain deterministic.
        self.backbone.eval()
        return self

    def _build_prefixed_inputs(
        self,
        prefix: torch.Tensor,             # [B, P, D]
        emb: torch.Tensor,                # [B, L, D]
        attention_mask: torch.Tensor,     # [B, L]
        labels: torch.Tensor,             # [B, L]
    ):
        inputs_embeds, attn, labels2 = build_inputs_embeds_with_prefix(
            emb=emb,
            attention_mask=attention_mask,
            labels=labels,
            prefix=prefix,
        )
        return inputs_embeds, attn, labels2

    def _contact_objective_active(self) -> bool:
        return bool(
            self.insufficiency_contact_objective
            and self.insufficiency_contact_weight > 0.0
            and self.tokenizer is not None
        )

    def _candidate_token_ids(self, text: str, *, device: torch.device) -> Optional[torch.Tensor]:
        text = str(text or "").strip()
        if not text:
            return None
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = getattr(encoded, "input_ids", None)
        if ids is None and isinstance(encoded, dict):
            ids = encoded.get("input_ids")
        if not ids:
            return None
        return torch.tensor([list(ids)], dtype=torch.long, device=device)

    def _contact_candidate_texts(self, meta: Dict[str, Any]) -> Optional[Tuple[str, List[str], bool]]:
        task_name = str(meta.get("task_name") or "")
        if not task_name:
            if "is_answerable" in meta:
                task_name = "squad2_reasoning"
            elif "gold_label" in meta:
                task_name = "fever_reasoning"

        insufficient = bool(meta.get("abstain_target"))
        if insufficient and not self.insufficiency_contact_apply_to_insufficient:
            return None
        if not insufficient and not self.insufficiency_contact_apply_to_supported:
            return None

        max_candidates = (
            self.insufficiency_contact_hard_negative_pool_size
            if self.insufficiency_contact_hard_negatives
            else self.insufficiency_contact_max_candidates
        )
        if task_name == "squad2_reasoning":
            abstain_text = str(meta.get("abstain_text") or DEFAULT_CONTEXT_ABSTAIN_TEXT)
            if insufficient:
                negative_source = (
                    meta.get("insufficiency_negative_candidate_texts")
                    or meta.get("answer_candidate_texts")
                    or []
                )
                negatives = [str(x).strip() for x in negative_source if str(x).strip()]
                return abstain_text, negatives[:max_candidates], insufficient

            gold_answers = [str(x).strip() for x in (meta.get("gold_answers") or []) if str(x).strip()]
            if not gold_answers:
                return None
            return gold_answers[0], [abstain_text][:max_candidates], insufficient

        if task_name == "fever_reasoning":
            gold_label = str(meta.get("gold_label") or "").strip().upper()
            if gold_label not in {"A", "B", "C"}:
                return None
            if gold_label == "C":
                fever_negatives = ["A", "B"] if self.insufficiency_contact_hard_negatives else ["A", "B"][:max_candidates]
                return "C", fever_negatives[:max_candidates], insufficient
            return gold_label, ["C"][:max_candidates], insufficient

        return None

    def _compute_insufficiency_contact_loss(
        self,
        *,
        input_ids: torch.Tensor,
        prompt_len: torch.Tensor,
        meta: Optional[List[Dict[str, Any]]],
        contact_prefixes: List[torch.Tensor],
        model_dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=device)
        stats = {
            "insufficiency_contact_loss": zero,
            "insufficiency_contact_eligible_rows": zero,
            "insufficiency_contact_insufficient_rows": zero,
            "insufficiency_contact_supported_rows": zero,
            "insufficiency_contact_missing_negative_rows": zero,
            "insufficiency_contact_margin_loss": zero,
            "insufficiency_contact_positive_nll": zero,
            "insufficiency_contact_threshold_loss": zero,
            "insufficiency_contact_supported_guard_loss": zero,
            "insufficiency_contact_path_preservation_loss": zero,
            "insufficiency_contact_candidate_pool_size": zero,
            "insufficiency_contact_selected_negative_token_len": zero,
            "insufficiency_contact_selected_negative_score": zero,
            "insufficiency_contact_hard_margin": zero,
            "insufficiency_contact_selected_negative_nonfirst_rows": zero,
            "insufficiency_contact_fallback_negative_rows": zero,
            "insufficiency_contact_fever_nei_supported_selected": zero,
            "insufficiency_contact_fever_nei_refuted_selected": zero,
        }
        if not self._contact_objective_active() or not meta or not contact_prefixes:
            return zero, stats

        row_losses: List[torch.Tensor] = []
        positive_nll_terms: List[torch.Tensor] = []
        margin_terms: List[torch.Tensor] = []
        threshold_terms: List[torch.Tensor] = []
        supported_guard_terms: List[torch.Tensor] = []
        path_terms: List[torch.Tensor] = []
        selected_negative_score_terms: List[torch.Tensor] = []
        hard_margin_terms: List[torch.Tensor] = []
        eligible_rows = 0
        insufficient_rows = 0
        supported_rows = 0
        missing_negative_rows = 0
        fallback_negative_rows = 0
        selected_negative_rows = 0
        selected_negative_token_len = 0
        selected_negative_nonfirst_rows = 0
        candidate_pool_size_total = 0
        fever_nei_supported_selected = 0
        fever_nei_refuted_selected = 0

        margin_target = torch.tensor(float(self.insufficiency_contact_margin), device=device)
        tolerance = torch.tensor(float(self.insufficiency_contact_tolerance), device=device)
        supported_margin_floor = torch.tensor(float(self.insufficiency_contact_supported_margin_floor), device=device)
        path_allowed_drift = torch.tensor(float(self.insufficiency_contact_path_allowed_drift), device=device)
        threshold_mode = self.insufficiency_contact_loss_mode == "threshold_contrastive"
        for row_idx, row_meta in enumerate(meta):
            row_meta = row_meta or {}
            candidate_pack = self._contact_candidate_texts(row_meta)
            if candidate_pack is None:
                continue
            positive_text, negative_texts, insufficient = candidate_pack
            positive_ids = self._candidate_token_ids(positive_text, device=device)
            if positive_ids is None:
                continue

            negative_items: List[Tuple[int, str, torch.Tensor]] = []
            for negative_idx, negative_text in enumerate(negative_texts):
                negative_ids = self._candidate_token_ids(negative_text, device=device)
                if negative_ids is not None:
                    negative_items.append((negative_idx, str(negative_text), negative_ids))

            eligible_rows += 1
            if insufficient:
                insufficient_rows += 1
            else:
                supported_rows += 1
            candidate_pool_size_total += len(negative_items)
            if not negative_items:
                missing_negative_rows += 1
            if self.insufficiency_contact_hard_negatives and len(negative_items) <= 1:
                fallback_negative_rows += 1

            prompt_i = input_ids[row_idx : row_idx + 1, : int(prompt_len[row_idx].item())]
            fixed_selected_items: Optional[List[Tuple[int, str, torch.Tensor]]] = None
            if (
                negative_items
                and self.insufficiency_contact_hard_negatives
                and self.insufficiency_contact_hard_negative_selection_mode != "per_step"
            ):
                positive_scores_for_selection: List[float] = []
                negative_scores_for_selection: List[List[float]] = [[] for _item in negative_items]
                with torch.no_grad():
                    for prefix in contact_prefixes:
                        prefix_i = prefix[row_idx : row_idx + 1].detach()
                        positive_score_detached = average_candidate_logprob(
                            backbone=self.backbone,
                            prompt_input_ids=prompt_i,
                            prefix=prefix_i,
                            candidate_ids=positive_ids,
                            model_dtype=model_dtype,
                            checkpoint_backbone=self.insufficiency_contact_checkpoint_candidates,
                        ).squeeze(0)
                        positive_scores_for_selection.append(float(positive_score_detached.float().item()))
                        for item_pos, (_negative_idx, _negative_text, negative_ids) in enumerate(negative_items):
                            negative_score_detached = average_candidate_logprob(
                                backbone=self.backbone,
                                prompt_input_ids=prompt_i,
                                prefix=prefix_i,
                                candidate_ids=negative_ids,
                                model_dtype=model_dtype,
                                checkpoint_backbone=self.insufficiency_contact_checkpoint_candidates,
                            ).squeeze(0)
                            negative_scores_for_selection[item_pos].append(
                                float(negative_score_detached.float().item())
                            )
                selection_info = select_hard_negative_by_scores(
                    positive_scores_by_step=positive_scores_for_selection,
                    negative_scores_by_candidate=negative_scores_for_selection,
                    selection_mode=self.insufficiency_contact_hard_negative_selection_mode,
                    top_k=self.insufficiency_contact_hard_negative_top_k,
                )
                fixed_positions = [
                    int(pos)
                    for pos in (selection_info.get("selected_indices") or [])
                    if 0 <= int(pos) < len(negative_items)
                ]
                if fixed_positions:
                    fixed_selected_items = [negative_items[pos] for pos in fixed_positions]
                    first_pos = fixed_positions[0]
                    first_negative_idx, first_negative_text, first_negative_ids = negative_items[first_pos]
                    selected_negative_rows += 1
                    if first_negative_idx != 0:
                        selected_negative_nonfirst_rows += 1
                    selected_negative_token_len += int(first_negative_ids.size(1))
                    selected_negative_scores = negative_scores_for_selection[first_pos]
                    selected_negative_score = max(selected_negative_scores) if selected_negative_scores else None
                    if selected_negative_score is not None:
                        selected_negative_score_terms.append(
                            torch.tensor(float(selected_negative_score), device=device)
                        )
                    if insufficient and str(row_meta.get("task_name") or "") == "fever_reasoning":
                        if first_negative_text == "A":
                            fever_nei_supported_selected += 1
                        elif first_negative_text == "B":
                            fever_nei_refuted_selected += 1

            row_selection_counted = fixed_selected_items is not None
            step_losses: List[torch.Tensor] = []
            row_hard_margins: List[torch.Tensor] = []
            for prefix in contact_prefixes:
                prefix_i = prefix[row_idx : row_idx + 1]
                positive_score = average_candidate_logprob(
                    backbone=self.backbone,
                    prompt_input_ids=prompt_i,
                    prefix=prefix_i,
                    candidate_ids=positive_ids,
                    model_dtype=model_dtype,
                    checkpoint_backbone=self.insufficiency_contact_checkpoint_candidates,
                ).squeeze(0)
                positive_nll = -positive_score
                if negative_items:
                    selected_items = fixed_selected_items or negative_items
                    selected_score_values: Dict[int, torch.Tensor] = {}
                    if self.insufficiency_contact_hard_negatives and fixed_selected_items is None:
                        scored_items = []
                        with torch.no_grad():
                            for negative_idx, negative_text, negative_ids in negative_items:
                                score = average_candidate_logprob(
                                    backbone=self.backbone,
                                    prompt_input_ids=prompt_i,
                                    prefix=prefix_i.detach(),
                                    candidate_ids=negative_ids,
                                    model_dtype=model_dtype,
                                    checkpoint_backbone=self.insufficiency_contact_checkpoint_candidates,
                                ).squeeze(0)
                                scored_items.append(
                                    (
                                        float(score.detach().float().item()),
                                        negative_idx,
                                        negative_text,
                                        negative_ids,
                                        score.detach(),
                                    )
                                )
                        scored_items.sort(key=lambda item: item[0], reverse=True)
                        selected_scored = scored_items[
                            : max(1, min(self.insufficiency_contact_hard_negative_top_k, len(scored_items)))
                        ]
                        selected_items = [(idx, text, ids) for _score, idx, text, ids, _score_tensor in selected_scored]
                        selected_score_values = {idx: score_tensor for _score, idx, _text, _ids, score_tensor in selected_scored}
                        if selected_scored and not row_selection_counted:
                            selected_negative_rows += 1
                            if selected_scored[0][1] != 0:
                                selected_negative_nonfirst_rows += 1
                            selected_negative_token_len += int(selected_scored[0][3].size(1))
                            selected_negative_score_terms.append(selected_scored[0][4])
                            if insufficient and str(row_meta.get("task_name") or "") == "fever_reasoning":
                                if selected_scored[0][2] == "A":
                                    fever_nei_supported_selected += 1
                                elif selected_scored[0][2] == "B":
                                    fever_nei_refuted_selected += 1
                            row_selection_counted = True

                    negative_scores = [
                        average_candidate_logprob(
                            backbone=self.backbone,
                            prompt_input_ids=prompt_i,
                            prefix=prefix_i,
                            candidate_ids=negative_ids,
                            model_dtype=model_dtype,
                            checkpoint_backbone=self.insufficiency_contact_checkpoint_candidates,
                        ).squeeze(0)
                        for _negative_idx, _negative_text, negative_ids in selected_items
                    ]
                    negative_score = torch.stack(negative_scores).max()
                    if not selected_score_values:
                        selected_negative_score_terms.append(negative_score.detach())

                    if threshold_mode:
                        if insufficient:
                            hard_margin = negative_score - positive_score
                            threshold_loss = F.relu(hard_margin - tolerance)
                            hard_margin_terms.append(hard_margin)
                            threshold_terms.append(threshold_loss)
                            margin_terms.append(threshold_loss)
                            row_hard_margins.append(hard_margin)
                            step_loss = (
                                threshold_loss
                                + self.insufficiency_contact_positive_nll_weight * positive_nll
                            )
                        else:
                            answer_margin = positive_score - negative_score
                            supported_guard = F.relu(supported_margin_floor - answer_margin)
                            supported_guard_terms.append(supported_guard)
                            margin_terms.append(supported_guard)
                            step_loss = (
                                self.insufficiency_contact_positive_nll_weight * positive_nll
                                + self.insufficiency_contact_supported_guard_weight * supported_guard
                            )
                    else:
                        margin_loss = F.relu(margin_target - (positive_score - negative_score))
                        margin_terms.append(margin_loss)
                        step_loss = positive_nll + margin_loss
                else:
                    step_loss = (
                        self.insufficiency_contact_positive_nll_weight * positive_nll
                        if threshold_mode
                        else positive_nll
                    )
                positive_nll_terms.append(positive_nll)
                step_losses.append(step_loss)

            if step_losses:
                row_loss = torch.stack(step_losses).mean()
                if threshold_mode and insufficient and len(row_hard_margins) > 1:
                    early_margin = row_hard_margins[0]
                    final_margin = row_hard_margins[-1]
                    if bool((early_margin.detach() <= tolerance).item()):
                        path_loss = F.relu(final_margin - tolerance)
                    else:
                        path_loss = F.relu(final_margin - early_margin - path_allowed_drift)
                    path_terms.append(path_loss)
                    row_loss = row_loss + path_loss
                row_losses.append(row_loss)

        if not row_losses:
            return zero, stats

        contact_loss = torch.stack(row_losses).mean()
        stats.update(
            {
                "insufficiency_contact_loss": contact_loss,
                "insufficiency_contact_eligible_rows": torch.tensor(float(eligible_rows), device=device),
                "insufficiency_contact_insufficient_rows": torch.tensor(float(insufficient_rows), device=device),
                "insufficiency_contact_supported_rows": torch.tensor(float(supported_rows), device=device),
                "insufficiency_contact_missing_negative_rows": torch.tensor(float(missing_negative_rows), device=device),
                "insufficiency_contact_margin_loss": torch.stack(margin_terms).mean() if margin_terms else zero,
                "insufficiency_contact_positive_nll": (
                    torch.stack(positive_nll_terms).mean() if positive_nll_terms else zero
                ),
                "insufficiency_contact_threshold_loss": torch.stack(threshold_terms).mean() if threshold_terms else zero,
                "insufficiency_contact_supported_guard_loss": (
                    torch.stack(supported_guard_terms).mean() if supported_guard_terms else zero
                ),
                "insufficiency_contact_path_preservation_loss": torch.stack(path_terms).mean() if path_terms else zero,
                "insufficiency_contact_candidate_pool_size": torch.tensor(
                    float(candidate_pool_size_total / max(1, eligible_rows)),
                    device=device,
                ),
                "insufficiency_contact_selected_negative_token_len": torch.tensor(
                    float(selected_negative_token_len / max(1, selected_negative_rows)),
                    device=device,
                ),
                "insufficiency_contact_selected_negative_score": (
                    torch.stack(selected_negative_score_terms).mean() if selected_negative_score_terms else zero
                ),
                "insufficiency_contact_hard_margin": torch.stack(hard_margin_terms).mean() if hard_margin_terms else zero,
                "insufficiency_contact_selected_negative_nonfirst_rows": torch.tensor(
                    float(selected_negative_nonfirst_rows),
                    device=device,
                ),
                "insufficiency_contact_fallback_negative_rows": torch.tensor(float(fallback_negative_rows), device=device),
                "insufficiency_contact_fever_nei_supported_selected": torch.tensor(
                    float(fever_nei_supported_selected),
                    device=device,
                ),
                "insufficiency_contact_fever_nei_refuted_selected": torch.tensor(
                    float(fever_nei_refuted_selected),
                    device=device,
                ),
            }
        )
        return contact_loss, stats

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        # Place all tensors on the embedding device (good default with device_map="auto")
        embed_weight = self.backbone.get_input_embeddings().weight
        dev = embed_weight.device
        model_dtype = getattr(self.backbone, "dtype", torch.float16)

        input_ids = batch.input_ids.to(dev)
        attention_mask = batch.attention_mask.to(dev)
        labels = batch.labels.to(dev)
        prompt_len = batch.prompt_len.to(dev)

        # Frozen token embeddings
        with torch.no_grad():
            emb = self.backbone.get_input_embeddings()(input_ids)  # [B, L, D]

        B, L, D = emb.shape
        pos = torch.arange(L, device=dev).unsqueeze(0)
        prompt_mask = (pos < prompt_len.unsqueeze(1)) & attention_mask.bool()
        h = masked_mean(emb, prompt_mask, dim=1)  # [B, D]

        # Head forward (grad through head params)
        prefix_final, energies_pred, z_traj = self.head(h)  # energies_pred: [B, S+1]
        # ---- Prefix safety: prevents fp16 overflow / NaNs ----
        max_prefix_norm = 5.0
        with torch.no_grad():
            pn_pre = prefix_final.detach().norm(dim=-1)  # [B, P]
            pn_pre_safe = torch.nan_to_num(pn_pre, nan=0.0, posinf=1e9, neginf=0.0)
            pn_pre_mean = pn_pre_safe.mean()
            pn_pre_max = pn_pre_safe.max()
            clamped = (~torch.isfinite(pn_pre)) | (pn_pre > max_prefix_norm)
            clamped_frac = clamped.float().mean()
        prefix_final = clamp_prefix_norm(prefix_final, max_prefix_norm, eps=1e-6)
        with torch.no_grad():
            pn_post = prefix_final.detach().norm(dim=-1)
            pn_post_safe = torch.nan_to_num(pn_post, nan=0.0, posinf=1e9, neginf=0.0)
            pn_post_mean = pn_post_safe.mean()
            pn_post_max = pn_post_safe.max()
        prefix_absmax = prefix_final.detach().abs().max()
        prefix_norm_mean = pn_post_mean
        prefix_final = prefix_final.to(model_dtype)

        # ---- Final-step LM loss (with grad to head via prefix) ----
        inputs_embeds, attn, labels2 = self._build_prefixed_inputs(
            prefix_final,
            emb.detach().to(model_dtype),
            attention_mask,
            labels,
        )

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )

        loss_per_sample_final = causal_ce_per_sample(out.logits, labels2)  # [B]
        lm_loss = loss_per_sample_final.mean()

        # ---- Stepwise TRUE energy targets (no_grad) for z_0..z_{S-1} ----
        # We compute targets for the first S states under no_grad, then append final loss as target for z_S.
        S = self.head.num_steps
        # z_traj: [B, S+1, latent] -> partial states [B, S, latent]
        z_partial = z_traj[:, :S, :].detach()

        with torch.no_grad():
            # prefix for each partial state: [B, S, P, D]
            z_flat = z_partial.reshape(B * S, self.head.latent_dim)
            prefix_partial = self.head.prefix_proj(z_flat).view(B * S, self.head.prefix_len, D)
            prefix_partial = clamp_prefix_norm(prefix_partial, max_prefix_norm, eps=1e-6).to(model_dtype)

            # replicate token embeddings and masks across steps
            emb_rep = emb.detach().to(model_dtype).unsqueeze(1).expand(B, S, L, D).reshape(B * S, L, D)
            attn_rep = attention_mask.unsqueeze(1).expand(B, S, L).reshape(B * S, L)
            labels_rep = labels.unsqueeze(1).expand(B, S, L).reshape(B * S, L)

            inputs_embeds_s, attn_s, labels2_s = self._build_prefixed_inputs(
                prefix_partial, emb_rep, attn_rep, labels_rep
            )

            out_s = self.backbone(
                inputs_embeds=inputs_embeds_s,
                attention_mask=attn_s,
                use_cache=False,
                return_dict=True,
            )

            loss_partial = causal_ce_per_sample(out_s.logits, labels2_s)  # [B*S]
            loss_partial = loss_partial.view(B, S)  # [B, S]

        # targets for all states z_0..z_S: [B, S+1]
        targets = torch.cat([loss_partial, loss_per_sample_final.detach().unsqueeze(1)], dim=1)

        # ---- Energy supervision (upgrade #1) ----
        # energies_pred shape is [B, S+1]
        energy_mse = F.mse_loss(energies_pred, targets)

        # ---- Monotonicity / basin deepening pressure ----
        # Encourage predicted energy to not increase along trajectory
        mono_pen = F.relu(energies_pred[:, 1:] - energies_pred[:, :-1]).mean()

        # ---- Candidate-contact objective ----
        # Keep this after the no-grad true-energy pass so candidate-scoring graphs
        # do not raise peak memory during the replicated stepwise measurement.
        contact_prefixes = [prefix_final]
        if self._contact_objective_active() and self.insufficiency_contact_steps in {
            "all",
            "early_final",
            "entry_final",
            "entry_mid_final",
        }:
            if self.insufficiency_contact_steps == "all":
                z_contact = z_traj
            elif self.insufficiency_contact_steps == "entry_mid_final":
                mid_idx = int(z_traj.size(1) // 2)
                z_contact = torch.stack([z_traj[:, 0, :], z_traj[:, mid_idx, :], z_traj[:, -1, :]], dim=1)
            else:
                z_contact = torch.stack([z_traj[:, 0, :], z_traj[:, -1, :]], dim=1)
            contact_step_count = int(z_contact.size(1))
            z_contact_flat = z_contact.reshape(B * contact_step_count, self.head.latent_dim)
            prefix_contact = self.head.prefix_proj(z_contact_flat).view(
                B,
                contact_step_count,
                self.head.prefix_len,
                D,
            )
            prefix_contact = clamp_prefix_norm(prefix_contact, max_prefix_norm, eps=1e-6).to(model_dtype)
            contact_prefixes = [prefix_contact[:, step_idx] for step_idx in range(contact_step_count)]

        contact_loss, contact_stats = self._compute_insufficiency_contact_loss(
            input_ids=input_ids,
            prompt_len=prompt_len,
            meta=batch.meta,
            contact_prefixes=contact_prefixes,
            model_dtype=model_dtype,
            device=dev,
        )

        total = (
            lm_loss
            + self.energy_weight * energy_mse
            + self.mono_weight * mono_pen
            + self.insufficiency_contact_weight * contact_loss
        )

        # ---- Logging helpers ----
        with torch.no_grad():
            corr = pearson_corr(energies_pred, targets)
            pred_mean = energies_pred.mean(dim=0)
            targ_mean = targets.mean(dim=0)

        return {
            "loss": total,
            "lm_loss": lm_loss.detach(),
            "energy_mse": energy_mse.detach(),
            "mono_pen": mono_pen.detach(),
            "corr_pred_true": torch.tensor(corr, device=dev),
            "pred_energy_mean": pred_mean.detach(),
            "true_energy_mean": targ_mean.detach(),
            "prefix_absmax": prefix_absmax.detach(),
            "prefix_norm": prefix_norm_mean.detach(),
            "prefix_norm_pre_mean": pn_pre_mean.detach(),
            "prefix_norm_pre_max": pn_pre_max.detach(),
            "prefix_norm_post_mean": pn_post_mean.detach(),
            "prefix_norm_post_max": pn_post_max.detach(),
            "prefix_clamped_frac": clamped_frac.detach(),
            "insufficiency_contact_loss": contact_stats["insufficiency_contact_loss"].detach(),
            "insufficiency_contact_eligible_rows": contact_stats[
                "insufficiency_contact_eligible_rows"
            ].detach(),
            "insufficiency_contact_insufficient_rows": contact_stats[
                "insufficiency_contact_insufficient_rows"
            ].detach(),
            "insufficiency_contact_supported_rows": contact_stats[
                "insufficiency_contact_supported_rows"
            ].detach(),
            "insufficiency_contact_missing_negative_rows": contact_stats[
                "insufficiency_contact_missing_negative_rows"
            ].detach(),
            "insufficiency_contact_margin_loss": contact_stats["insufficiency_contact_margin_loss"].detach(),
            "insufficiency_contact_positive_nll": contact_stats["insufficiency_contact_positive_nll"].detach(),
            "insufficiency_contact_threshold_loss": contact_stats["insufficiency_contact_threshold_loss"].detach(),
            "insufficiency_contact_supported_guard_loss": contact_stats[
                "insufficiency_contact_supported_guard_loss"
            ].detach(),
            "insufficiency_contact_path_preservation_loss": contact_stats[
                "insufficiency_contact_path_preservation_loss"
            ].detach(),
            "insufficiency_contact_candidate_pool_size": contact_stats[
                "insufficiency_contact_candidate_pool_size"
            ].detach(),
            "insufficiency_contact_selected_negative_token_len": contact_stats[
                "insufficiency_contact_selected_negative_token_len"
            ].detach(),
            "insufficiency_contact_selected_negative_score": contact_stats[
                "insufficiency_contact_selected_negative_score"
            ].detach(),
            "insufficiency_contact_hard_margin": contact_stats["insufficiency_contact_hard_margin"].detach(),
            "insufficiency_contact_selected_negative_nonfirst_rows": contact_stats[
                "insufficiency_contact_selected_negative_nonfirst_rows"
            ].detach(),
            "insufficiency_contact_fallback_negative_rows": contact_stats[
                "insufficiency_contact_fallback_negative_rows"
            ].detach(),
            "insufficiency_contact_fever_nei_supported_selected": contact_stats[
                "insufficiency_contact_fever_nei_supported_selected"
            ].detach(),
            "insufficiency_contact_fever_nei_refuted_selected": contact_stats[
                "insufficiency_contact_fever_nei_refuted_selected"
            ].detach(),
        }



# -------------------------
# 3.5) Evaluation helpers (stable metrics + baseline)
# -------------------------

@torch.no_grad()
def baseline_loss_mean(backbone: AutoModelForCausalLM, batch: Batch) -> torch.Tensor:
    """
    Baseline: frozen backbone with NO prefix steering.
    Returns mean CE loss over answer tokens (prompt tokens are -100 in labels).
    """
    emb_w = backbone.get_input_embeddings().weight
    dev = emb_w.device
    model_dtype = getattr(backbone, "dtype", torch.float16)

    input_ids = batch.input_ids.to(dev)
    attention_mask = batch.attention_mask.to(dev)
    labels = batch.labels.to(dev)

    emb = backbone.get_input_embeddings()(input_ids).to(model_dtype)

    out = backbone(
        inputs_embeds=emb,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    loss_per_sample = causal_ce_per_sample(out.logits, labels)
    return loss_per_sample.mean()


@torch.no_grad()
def _first_supervised_positions(labels: torch.Tensor) -> torch.Tensor:
    """Return [B] indices of the first supervised token per sample (labels != -100).

    If a sample has no supervised tokens, returns -1 for that sample.
    """
    B, T = labels.shape
    mask = labels != -100
    # argmax over a mask is not safe when all-false; handle explicitly
    idx = torch.full((B,), -1, device=labels.device, dtype=torch.long)
    any_sup = mask.any(dim=1)
    if any_sup.any():
        # For rows with any supervision, find first True
        first = mask.float().cumsum(dim=1).eq(1).float().argmax(dim=1)
        idx = torch.where(any_sup, first, idx)
    return idx


@torch.no_grad()
def _multichoice_acc_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    candidate_token_ids: Optional[List[int]] = None,
) -> float:
    """Compute multichoice accuracy on the FIRST supervised token.

    We restrict predictions to the candidate tokens (A/B/C/D...) to avoid
    counting non-letter tokens as "answers". If candidate_token_ids is None,
    we infer candidates from the gold tokens present in the batch.
    """
    # logits: [B,T,V], labels: [B,T]
    B, T, V = logits.shape
    dev = logits.device
    first_pos = _first_supervised_positions(labels)  # [B]
    valid = first_pos > 0  # need pos-1 for causal logits
    if not valid.any():
        return float('nan')

    gold = labels[torch.arange(B, device=dev), first_pos.clamp(min=0)]
    gold = gold[valid]

    # predicted token comes from logits at (first_pos - 1)
    pred_pos = (first_pos - 1)[valid]
    sel_logits = logits[torch.arange(B, device=dev)[valid], pred_pos]  # [Bv,V]

    if candidate_token_ids is None:
        candidate_token_ids = sorted({int(t) for t in gold.tolist()})
    if not candidate_token_ids:
        return float('nan')

    cand = torch.tensor(candidate_token_ids, device=dev, dtype=torch.long)
    cand_logits = sel_logits.index_select(dim=-1, index=cand)
    pred_idx = cand_logits.argmax(dim=-1)
    pred_tok = cand[pred_idx]

    acc = (pred_tok == gold).float().mean().item()
    return float(acc)


@torch.no_grad()
def _supervised_token_acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Token accuracy over supervised positions only (labels != -100).

    Uses causal shifting: logits[:, :-1] predict labels[:, 1:].
    """
    if logits.ndim != 3 or labels.ndim != 2:
        return float("nan")
    if logits.size(0) != labels.size(0) or logits.size(1) != labels.size(1):
        return float("nan")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    denom = mask.float().sum()
    if denom.item() <= 0:
        return float("nan")

    pred = shift_logits.argmax(dim=-1)
    correct = (pred == shift_labels) & mask
    return float(correct.float().sum().item() / denom.item())


def _single_token_id(tokenizer, text: str) -> Optional[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        return None
    return int(ids[0])


def _candidate_token_id_map(tokenizer, letters: List[str]) -> Optional[Dict[str, int]]:
    out: Dict[str, int] = {}
    for s in letters:
        tid = _single_token_id(tokenizer, s)
        if tid is None:
            return None
        out[s] = tid
    if len(set(out.values())) != len(out):
        return None
    return out


def _first_int_from_text(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", text.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _multichoice_labels_for_prefix(prefix: str) -> List[str]:
    """Return the canonical option labels for a multichoice-style eval prefix."""
    if prefix == "mmlu_pro":
        return [chr(ord("A") + i) for i in range(10)]
    if prefix == "truthfulqa":
        return ["A", "B", "C", "D", "E"]
    if prefix == "fever_reasoning":
        return ["A", "B", "C"]
    return ["A", "B", "C", "D"]


MULTICHOICE_PREFIXES = {"multichoice", "mmlu_pro", "gpqa_diamond", "truthfulqa", "fever_reasoning"}


@torch.no_grad()
def _decode_supervised_predictions(tokenizer, logits: torch.Tensor, labels: torch.Tensor) -> List[Optional[str]]:
    """Decode greedy predictions over the supervised answer window for each sample."""
    B, _T, _V = logits.shape
    dev = logits.device
    sup = labels != -100
    out: List[Optional[str]] = []
    for i in range(B):
        pos = sup[i].nonzero(as_tuple=False).flatten()
        if pos.numel() == 0 or int(pos.min().item()) <= 0:
            out.append(None)
            continue
        pred_pos = (pos - 1).to(dev)
        pred_ids = logits[i, pred_pos].argmax(dim=-1).tolist()
        out.append(tokenizer.decode(pred_ids, skip_special_tokens=True).strip())
    return out


@torch.no_grad()
def _multichoice_predictions_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_id_map: Dict[str, int],
    candidate_labels: List[str],
) -> List[Optional[str]]:
    """Return predicted option labels on the first supervised position for each sample."""
    if any(lbl not in token_id_map for lbl in candidate_labels):
        return [None] * labels.size(0)

    B, _T, _V = logits.shape
    dev = logits.device
    first_pos = _first_supervised_positions(labels)
    valid = first_pos > 0
    out: List[Optional[str]] = [None] * B
    if not valid.any():
        return out

    rows = torch.arange(B, device=dev)[valid]
    pred_pos = (first_pos - 1)[valid]
    sel_logits = logits[rows, pred_pos]

    cand_ids = torch.tensor([token_id_map[lbl] for lbl in candidate_labels], device=dev, dtype=torch.long)
    cand_logits = sel_logits.index_select(dim=-1, index=cand_ids)
    pred_idx = cand_logits.argmax(dim=-1).tolist()
    row_ids = rows.tolist()
    for row_idx, choice_idx in zip(row_ids, pred_idx):
        out[row_idx] = candidate_labels[int(choice_idx)]
    return out


def _safe_rate(num: float, den: float) -> float:
    """Guard division for sparse eval slices; returns NaN when the denominator is empty."""
    if den <= 0:
        return float("nan")
    return float(num / den)


def _empty_reasoning_stats() -> Dict[str, float]:
    """Accumulator state for cheap teacher-forced reasoning metrics."""
    return {
        "total": 0.0,
        "insufficient_total": 0.0,
        "insufficient_abstain_correct": 0.0,
        "abstain_pred_total": 0.0,
        "abstain_pred_correct": 0.0,
        "false_collapse": 0.0,
        "unnecessary_abstain": 0.0,
        "concrete_pred_total": 0.0,
        "answerable_total": 0.0,
        "answerable_correct": 0.0,
        "overall_correct": 0.0,
        "supported_total": 0.0,
        "supported_correct": 0.0,
        "refuted_total": 0.0,
        "refuted_correct": 0.0,
        "not_enough_info_total": 0.0,
        "not_enough_info_correct": 0.0,
    }


def _update_squad2_reasoning_stats(
    stats: Dict[str, float],
    meta: List[Dict[str, Any]],
    pred_texts: List[Optional[str]],
) -> None:
    """Update answer-vs-abstain counts for SQuAD 2.0-style text outputs."""
    for item_meta, pred_text in zip(meta, pred_texts):
        if not isinstance(item_meta, dict):
            continue
        is_answerable = bool(item_meta.get("is_answerable"))
        abstain_text = str(item_meta.get("abstain_text") or DEFAULT_CONTEXT_ABSTAIN_TEXT)
        pred_norm = _normalize_eval_text(pred_text or "")
        abstain_norm = _normalize_eval_text(abstain_text)
        pred_is_abstain = bool(pred_norm) and pred_norm == abstain_norm
        gold_answers = [
            _normalize_eval_text(str(ans))
            for ans in (item_meta.get("gold_answers") or [])
            if isinstance(ans, str) and str(ans).strip()
        ]

        stats["total"] += 1.0
        if pred_is_abstain:
            stats["abstain_pred_total"] += 1.0
        else:
            stats["concrete_pred_total"] += 1.0

        if is_answerable:
            stats["answerable_total"] += 1.0
            answer_correct = bool(pred_norm) and pred_norm in gold_answers
            if answer_correct:
                stats["answerable_correct"] += 1.0
                stats["overall_correct"] += 1.0
            if pred_is_abstain:
                stats["unnecessary_abstain"] += 1.0
        else:
            stats["insufficient_total"] += 1.0
            if pred_is_abstain:
                stats["insufficient_abstain_correct"] += 1.0
                stats["abstain_pred_correct"] += 1.0
                stats["overall_correct"] += 1.0
            else:
                stats["false_collapse"] += 1.0


def _update_fever_reasoning_stats(
    stats: Dict[str, float],
    meta: List[Dict[str, Any]],
    pred_labels: List[Optional[str]],
) -> None:
    """Update supported/refuted/NEI decision counts for FEVER-style MC outputs."""
    for item_meta, pred_label in zip(meta, pred_labels):
        if not isinstance(item_meta, dict):
            continue
        gold_label = item_meta.get("gold_label")
        reasoning_label = item_meta.get("reasoning_label")
        abstain_label = str(item_meta.get("abstain_option_letter") or "C")
        if not gold_label or not reasoning_label:
            continue

        pred_is_abstain = pred_label == abstain_label
        is_insufficient = reasoning_label == "not_enough_info"

        stats["total"] += 1.0
        if pred_is_abstain:
            stats["abstain_pred_total"] += 1.0
        else:
            stats["concrete_pred_total"] += 1.0

        if is_insufficient:
            stats["insufficient_total"] += 1.0
            if pred_is_abstain:
                stats["insufficient_abstain_correct"] += 1.0
                stats["abstain_pred_correct"] += 1.0
            else:
                stats["false_collapse"] += 1.0
        else:
            stats["answerable_total"] += 1.0
            if pred_is_abstain:
                stats["unnecessary_abstain"] += 1.0

        label_total_key = f"{reasoning_label}_total"
        label_correct_key = f"{reasoning_label}_correct"
        if label_total_key in stats:
            stats[label_total_key] += 1.0
            if pred_label == gold_label:
                stats[label_correct_key] += 1.0

        if pred_label == gold_label:
            stats["overall_correct"] += 1.0


def _finalize_reasoning_metrics(prefix: str, stats: Dict[str, float], *, system: str) -> Dict[str, float]:
    """Project accumulated reasoning counts into logged metric keys."""
    metrics = {
        f"{prefix}_uncertainty_recall_{system}": _safe_rate(
            stats["insufficient_abstain_correct"], stats["insufficient_total"]
        ),
        f"{prefix}_uncertainty_precision_{system}": _safe_rate(
            stats["abstain_pred_correct"], stats["abstain_pred_total"]
        ),
        f"{prefix}_false_collapse_rate_{system}": _safe_rate(stats["false_collapse"], stats["insufficient_total"]),
        f"{prefix}_unnecessary_abstain_rate_{system}": _safe_rate(
            stats["unnecessary_abstain"], stats["answerable_total"]
        ),
        f"{prefix}_coverage_{system}": _safe_rate(stats["concrete_pred_total"], stats["total"]),
    }
    if prefix == "squad2_reasoning":
        metrics[f"{prefix}_answerable_accuracy_{system}"] = _safe_rate(
            stats["answerable_correct"], stats["answerable_total"]
        )
        metrics[f"{prefix}_overall_accuracy_{system}"] = _safe_rate(stats["overall_correct"], stats["total"])
    elif prefix == "fever_reasoning":
        metrics[f"{prefix}_overall_label_accuracy_{system}"] = _safe_rate(stats["overall_correct"], stats["total"])
        metrics[f"{prefix}_supported_accuracy_{system}"] = _safe_rate(
            stats["supported_correct"], stats["supported_total"]
        )
        metrics[f"{prefix}_refuted_accuracy_{system}"] = _safe_rate(stats["refuted_correct"], stats["refuted_total"])
        metrics[f"{prefix}_not_enough_info_accuracy_{system}"] = _safe_rate(
            stats["not_enough_info_correct"], stats["not_enough_info_total"]
        )
    return metrics


@torch.no_grad()
def _int_em_from_logits(tokenizer, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Integer exact-match over the supervised answer window (teacher-forced)."""
    B, T, _V = logits.shape
    dev = logits.device
    correct = 0
    total = 0

    sup = labels != -100  # [B,T]
    for i in range(B):
        pos = sup[i].nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        if int(pos.min().item()) <= 0:
            continue
        gold_ids = labels[i, pos].tolist()
        pred_pos = (pos - 1).to(dev)
        pred_ids = logits[i, pred_pos].argmax(dim=-1).tolist()

        gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True).strip()
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        gi = _first_int_from_text(gold_text)
        pi = _first_int_from_text(pred_text)
        if gi is None or pi is None:
            total += 1
            continue
        correct += int(gi == pi)
        total += 1

    if total == 0:
        return float("nan")
    return float(correct / total)


@torch.no_grad()
def _multichoice_abstention_data_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_id_map: Dict[str, int],
    concrete_labels: List[str],
    abstain_label: str,
) -> Tuple[List[float], List[int]]:
    """Return (p_abstain, correct_nonabstain) for MC tasks with an abstain option."""
    required = list(concrete_labels) + [abstain_label]
    if any(k not in token_id_map for k in required):
        return [], []

    B, _T, _V = logits.shape
    dev = logits.device

    first_pos = _first_supervised_positions(labels)  # [B]
    valid = first_pos > 0
    if not valid.any():
        return [], []

    rows = torch.arange(B, device=dev)[valid]
    pred_pos = (first_pos - 1)[valid]
    sel_logits = logits[rows, pred_pos]  # [Bv,V]
    gold = labels[rows, first_pos[valid]]  # [Bv]

    # Restrict gold to the concrete non-abstain labels only.
    gold_ok = torch.zeros_like(gold, dtype=torch.bool)
    for k in concrete_labels:
        gold_ok |= (gold == int(token_id_map[k]))
    if not gold_ok.any():
        return [], []

    sel_logits = sel_logits[gold_ok]
    gold = gold[gold_ok]

    cand_ids = torch.tensor([token_id_map[k] for k in required], device=dev, dtype=torch.long)
    lp = F.log_softmax(sel_logits, dim=-1)
    cand_lp = lp.index_select(dim=-1, index=cand_ids)
    cand_p = cand_lp.exp()  # [N,C+1]

    p_abstain = cand_p[:, -1]
    non_abstain = cand_p[:, : len(concrete_labels)]
    pred_idx = non_abstain.argmax(dim=-1)
    pred_tok = cand_ids[: len(concrete_labels)][pred_idx]
    correct_nonabstain = (pred_tok == gold).long()

    return p_abstain.detach().float().cpu().tolist(), correct_nonabstain.detach().cpu().tolist()


@torch.no_grad()
def _truthfulqa_abstention_data_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    token_id_map: Dict[str, int],
) -> Tuple[List[float], List[int]]:
    """Backward-compatible wrapper for TruthfulQA-style A/B/C/D(+E abstain)."""
    return _multichoice_abstention_data_from_logits(
        logits,
        labels,
        token_id_map=token_id_map,
        concrete_labels=["A", "B", "C", "D"],
        abstain_label="E",
    )


def _task_eval_metadata(prefix: str, config: "TrainingConfig") -> Dict[str, Any]:
    """Return additive role/abstention metadata for a named eval task."""
    # Current in-repo tasks remain basin-facing by default. Future uncertainty-aware
    # datasets can override these semantics explicitly without changing old tasks.
    meta: Dict[str, Any] = {
        "benchmark_role": "basin_probe",
        "reasoning_eval_mode": False,
        "allow_abstain": False,
        "abstain_mode": "none",
        "abstain_text": None,
        "abstain_option_letter": None,
        "task_format": "mc_letter" if prefix in MULTICHOICE_PREFIXES else "text_generation",
    }
    if prefix == "truthfulqa":
        allow_abstain = bool(config.truthfulqa_add_abstain)
        meta.update(
            {
                "allow_abstain": allow_abstain,
                "abstain_mode": "extra_option" if allow_abstain else "none",
                "abstain_text": LEGACY_TRUTHFULQA_ABSTAIN_TEXT if allow_abstain else None,
                "abstain_option_letter": (config.abstain_option_letter or "E") if allow_abstain else None,
            }
        )
    elif prefix == "squad2_reasoning":
        meta.update(
            {
                "benchmark_role": "reasoning_measure",
                "reasoning_eval_mode": True,
                "allow_abstain": True,
                "abstain_mode": "text",
                "abstain_text": config.abstain_text,
                "abstain_option_letter": None,
                "task_format": "text_generation",
            }
        )
    elif prefix == "fever_reasoning":
        meta.update(
            {
                "benchmark_role": "reasoning_measure",
                "reasoning_eval_mode": True,
                "allow_abstain": True,
                "abstain_mode": "native_label",
                "abstain_text": FEVER_REASONING_NEI_TEXT,
                "abstain_option_letter": "C",
                "task_format": "mc_letter",
            }
        )
    return meta


def _attach_task_eval_metadata(metrics: Dict[str, Any], *, prefix: str, config: "TrainingConfig") -> None:
    """Attach per-task role/abstention metadata to a metrics dict in-place."""
    for key, value in _task_eval_metadata(prefix, config).items():
        metrics[f"{prefix}_{key}"] = value


@torch.no_grad()
def _forward_base_logits(backbone: AutoModelForCausalLM, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels) for the base model (no prefix)."""
    emb_w = backbone.get_input_embeddings().weight
    dev = emb_w.device
    model_dtype = getattr(backbone, 'dtype', torch.float16)

    input_ids = batch.input_ids.to(dev)
    attention_mask = batch.attention_mask.to(dev)
    labels = batch.labels.to(dev)

    emb = backbone.get_input_embeddings()(input_ids).to(model_dtype)
    out = backbone(inputs_embeds=emb, attention_mask=attention_mask, use_cache=False, return_dict=True)
    return out.logits, labels


@torch.no_grad()
def _forward_static_logits(
    static_model: 'StaticPrefixWrapper',
    batch: Batch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels2) for static prefix model."""
    backbone = static_model.backbone
    emb_w = backbone.get_input_embeddings().weight
    dev = emb_w.device
    model_dtype = getattr(backbone, 'dtype', torch.float16)

    input_ids = batch.input_ids.to(dev)
    attention_mask = batch.attention_mask.to(dev)
    labels = batch.labels.to(dev)

    emb = backbone.get_input_embeddings()(input_ids).to(model_dtype)
    B, _, _ = emb.shape
    prefix = static_model.static_prefix(B).to(model_dtype)
    inputs_embeds, attn, labels2 = static_model._build_prefixed_inputs(prefix, emb, attention_mask, labels)
    out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False, return_dict=True)
    return out.logits, labels2


@torch.no_grad()
def _forward_real_logits(
    model: 'REALWrapper',
    batch: Batch,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels2) for REAL model using the fixed-step forward() prefix."""
    backbone = model.backbone
    head = model.head

    emb_w = backbone.get_input_embeddings().weight
    dev = emb_w.device
    model_dtype = getattr(backbone, 'dtype', torch.float16)

    input_ids = batch.input_ids.to(dev)
    attention_mask = batch.attention_mask.to(dev)
    labels = batch.labels.to(dev)
    prompt_len = batch.prompt_len.to(dev)

    emb = backbone.get_input_embeddings()(input_ids)  # [B,L,D]
    B, L, _ = emb.shape

    pos = torch.arange(L, device=dev).unsqueeze(0)
    prompt_mask = (pos < prompt_len.unsqueeze(1)) & attention_mask.bool()
    h = masked_mean(emb, prompt_mask, dim=1)

    prefix_final, _, _ = head(h)
    prefix_final = prefix_final.to(model_dtype)
    emb2 = emb.to(model_dtype)

    inputs_embeds, attn, labels2 = build_inputs_embeds_with_prefix(
        emb=emb2,
        attention_mask=attention_mask,
        labels=labels,
        prefix=prefix_final,
    )

    out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False, return_dict=True)
    return out.logits, labels2


@torch.no_grad()
def eval_real(
    model: "REALWrapper",
    backbone: AutoModelForCausalLM,
    val_loader: DataLoader,
    static_model: Optional["StaticPrefixWrapper"] = None,
    num_batches: int = 10,
    prefix: str = "val",
    tokenizer=None,
) -> Dict[str, float]:
    """
    Stable evaluation:
      - REAL loss vs baseline loss (no prefix)
      - Optional: static-prefix loss (prompt tuning control)
      - avg delta_true = true_energy[0] - true_energy[-1]
      - avg corr over multiple batches (less noisy)
      - Optional probe metrics (by `prefix`): multichoice letter acc, BoolQ yes/no acc, extractive-QA token acc, AIME EM, TruthfulQA abstention AUC
      - First-pass reasoning metrics for `squad2_reasoning` / `fever_reasoning`
        using teacher-forced predictions from the supervised answer window

    Returns metrics with keys prefixed by `prefix`, e.g. `val_base`.
    """
    model.eval()
    if static_model is not None:
        static_model.eval()

    real_sum = 0.0
    base_sum = 0.0
    static_sum = 0.0

    gain_real_sum = 0.0
    gain_static_sum = 0.0
    gain_cond_sum = 0.0

    emse_sum = 0.0
    mono_sum = 0.0
    corr_sum = 0.0
    delta_true_sum = 0.0
    acc_base_sum = 0.0
    acc_static_sum = 0.0
    acc_real_sum = 0.0
    tokacc_base_sum = 0.0
    tokacc_static_sum = 0.0
    tokacc_real_sum = 0.0
    em_base_sum = 0.0
    em_static_sum = 0.0
    em_real_sum = 0.0
    sup_tokens_sum = 0.0
    zero_sup_frac_sum = 0.0

    multichoice_candidate_token_ids = None
    multichoice_token_id_map = None
    if tokenizer is not None and prefix in MULTICHOICE_PREFIXES:
        letters = _multichoice_labels_for_prefix(prefix)
        token_map = _candidate_token_id_map(tokenizer, letters)
        if token_map is not None:
            multichoice_candidate_token_ids = list(token_map.values())
            multichoice_token_id_map = token_map

    boolq_candidate_token_ids = None
    if tokenizer is not None and prefix == "boolq":
        tok_yes = _single_token_id(tokenizer, "yes")
        tok_no = _single_token_id(tokenizer, "no")
        if tok_yes is not None and tok_no is not None and tok_yes != tok_no:
            boolq_candidate_token_ids = [int(tok_yes), int(tok_no)]

    truthfulqa_pE_base: List[float] = []
    truthfulqa_pE_static: List[float] = []
    truthfulqa_pE_real: List[float] = []
    truthfulqa_correct_base: List[int] = []
    truthfulqa_correct_static: List[int] = []
    truthfulqa_correct_real: List[int] = []
    reasoning_stats = None
    if prefix in {"squad2_reasoning", "fever_reasoning"}:
        reasoning_stats = {
            "base": _empty_reasoning_stats(),
            "real": _empty_reasoning_stats(),
        }
        if static_model is not None:
            reasoning_stats["static"] = _empty_reasoning_stats()

    n = 0
    for batch in val_loader:
        # Supervision sanity: how many tokens are actually supervised in this batch?
        sup = (batch.labels != -100).sum(dim=1)
        sup_tokens_sum += float(sup.float().mean().item())
        zero_sup_frac_sum += float((sup == 0).float().mean().item())
        out = model(batch)
        b = baseline_loss_mean(backbone, batch)

        real_i = float(out["lm_loss"].item())
        base_i = float(b.item())

        real_sum += real_i
        base_sum += base_i
        gain_real_sum += (base_i - real_i)

        if static_model is not None:
            s = static_model(batch)
            static_i = float(s.item())
            static_sum += static_i
            gain_static_sum += (base_i - static_i)
            gain_cond_sum += (static_i - real_i)

        emse_sum += float(out["energy_mse"].item())
        mono_sum += float(out["mono_pen"].item())
        corr_sum += float(out["corr_pred_true"].item())

        te = out["true_energy_mean"]  # [S+1], mean over batch
        delta_true_sum += float((te[0] - te[-1]).item())

        # Optional: probe-specific metrics (extra instrumentation).
        if prefix in MULTICHOICE_PREFIXES or prefix in {"aime", "boolq", "extractive_qa", "squad2_reasoning"}:
            try:
                base_logits, base_labels = _forward_base_logits(backbone, batch)
                real_logits, real_labels = _forward_real_logits(model, batch)
                static_logits, static_labels = None, None
                if static_model is not None:
                    static_logits, static_labels = _forward_static_logits(static_model, batch)

                if prefix in MULTICHOICE_PREFIXES:
                    acc_base_sum += _multichoice_acc_from_logits(
                        base_logits, base_labels, candidate_token_ids=multichoice_candidate_token_ids
                    )
                    acc_real_sum += _multichoice_acc_from_logits(
                        real_logits, real_labels, candidate_token_ids=multichoice_candidate_token_ids
                    )
                    if static_model is not None and static_logits is not None and static_labels is not None:
                        acc_static_sum += _multichoice_acc_from_logits(
                            static_logits, static_labels, candidate_token_ids=multichoice_candidate_token_ids
                        )

                    # TruthfulQA abstention calibration: AUC over accuracy-coverage curve.
                    if prefix == "truthfulqa" and tokenizer is not None and multichoice_token_id_map is not None:
                        pE, corr = _truthfulqa_abstention_data_from_logits(
                            base_logits, base_labels, token_id_map=multichoice_token_id_map
                        )
                        truthfulqa_pE_base.extend(pE)
                        truthfulqa_correct_base.extend(corr)

                        pE, corr = _truthfulqa_abstention_data_from_logits(
                            real_logits, real_labels, token_id_map=multichoice_token_id_map
                        )
                        truthfulqa_pE_real.extend(pE)
                        truthfulqa_correct_real.extend(corr)

                        if static_model is not None and static_logits is not None and static_labels is not None:
                            pE, corr = _truthfulqa_abstention_data_from_logits(
                                static_logits, static_labels, token_id_map=multichoice_token_id_map
                            )
                            truthfulqa_pE_static.extend(pE)
                            truthfulqa_correct_static.extend(corr)

                if prefix == "boolq" and tokenizer is not None:
                    acc_base_sum += _multichoice_acc_from_logits(
                        base_logits, base_labels, candidate_token_ids=boolq_candidate_token_ids
                    )
                    acc_real_sum += _multichoice_acc_from_logits(
                        real_logits, real_labels, candidate_token_ids=boolq_candidate_token_ids
                    )
                    if static_model is not None and static_logits is not None and static_labels is not None:
                        acc_static_sum += _multichoice_acc_from_logits(
                            static_logits, static_labels, candidate_token_ids=boolq_candidate_token_ids
                        )

                if prefix == "extractive_qa":
                    tokacc_base_sum += _supervised_token_acc_from_logits(base_logits, base_labels)
                    tokacc_real_sum += _supervised_token_acc_from_logits(real_logits, real_labels)
                    if static_model is not None and static_logits is not None and static_labels is not None:
                        tokacc_static_sum += _supervised_token_acc_from_logits(static_logits, static_labels)

                if prefix == "aime" and tokenizer is not None:
                    em_base_sum += _int_em_from_logits(tokenizer, base_logits, base_labels)
                    em_real_sum += _int_em_from_logits(tokenizer, real_logits, real_labels)
                    if static_model is not None and static_logits is not None and static_labels is not None:
                        em_static_sum += _int_em_from_logits(tokenizer, static_logits, static_labels)

                if reasoning_stats is not None and tokenizer is not None and batch.meta is not None:
                    if prefix == "squad2_reasoning":
                        _update_squad2_reasoning_stats(
                            reasoning_stats["base"],
                            batch.meta,
                            _decode_supervised_predictions(tokenizer, base_logits, base_labels),
                        )
                        _update_squad2_reasoning_stats(
                            reasoning_stats["real"],
                            batch.meta,
                            _decode_supervised_predictions(tokenizer, real_logits, real_labels),
                        )
                        if static_model is not None and static_logits is not None and static_labels is not None:
                            _update_squad2_reasoning_stats(
                                reasoning_stats["static"],
                                batch.meta,
                                _decode_supervised_predictions(tokenizer, static_logits, static_labels),
                            )
                    elif prefix == "fever_reasoning" and multichoice_token_id_map is not None:
                        candidate_labels = _multichoice_labels_for_prefix(prefix)
                        _update_fever_reasoning_stats(
                            reasoning_stats["base"],
                            batch.meta,
                            _multichoice_predictions_from_logits(
                                base_logits,
                                base_labels,
                                token_id_map=multichoice_token_id_map,
                                candidate_labels=candidate_labels,
                            ),
                        )
                        _update_fever_reasoning_stats(
                            reasoning_stats["real"],
                            batch.meta,
                            _multichoice_predictions_from_logits(
                                real_logits,
                                real_labels,
                                token_id_map=multichoice_token_id_map,
                                candidate_labels=candidate_labels,
                            ),
                        )
                        if static_model is not None and static_logits is not None and static_labels is not None:
                            _update_fever_reasoning_stats(
                                reasoning_stats["static"],
                                batch.meta,
                                _multichoice_predictions_from_logits(
                                    static_logits,
                                    static_labels,
                                    token_id_map=multichoice_token_id_map,
                                    candidate_labels=candidate_labels,
                                ),
                            )
            except Exception:
                # Keep eval robust; accuracy is extra instrumentation.
                pass

        n += 1
        if n >= num_batches:
            break

    # restore train mode for caller
    model.train()

    n = max(n, 1)
    base_key = f"{prefix}_base"
    real_key = f"{prefix}_real"
    metrics = {
        base_key: base_sum / n,
        real_key: real_sum / n,
        f"{prefix}_gain_real": gain_real_sum / n,
        f"{prefix}_e_mse": emse_sum / n,
        f"{prefix}_mono": mono_sum / n,
        f"{prefix}_corr": corr_sum / n,
        f"{prefix}_delta_true": delta_true_sum / n,
        f"{prefix}_sup_tokens": sup_tokens_sum / n,
        f"{prefix}_zero_sup_frac": zero_sup_frac_sum / n,
    }

    if static_model is not None:
        metrics.update({
            f"{prefix}_static": static_sum / n,
            f"{prefix}_gain_static": gain_static_sum / n,
            f"{prefix}_gain_conditional": gain_cond_sum / n,
        })

    # Multichoice-style accuracy metrics (first supervised token, restricted candidates).
    if prefix in MULTICHOICE_PREFIXES or (prefix == "boolq" and tokenizer is not None):
        metrics.update({
            f"{prefix}_acc_base": acc_base_sum / n,
            f"{prefix}_acc_real": acc_real_sum / n,
        })
        if static_model is not None:
            metrics[f"{prefix}_acc_static"] = acc_static_sum / n

    # Generic supervised-token accuracy (cheap/stable; useful for loss-only probes).
    if prefix == "extractive_qa":
        metrics.update({
            f"{prefix}_tokacc_base": tokacc_base_sum / n,
            f"{prefix}_tokacc_real": tokacc_real_sum / n,
        })
        if static_model is not None:
            metrics[f"{prefix}_tokacc_static"] = tokacc_static_sum / n

    # AIME integer exact-match (teacher-forced).
    if prefix == "aime" and tokenizer is not None:
        metrics.update({
            f"{prefix}_em_base": em_base_sum / n,
            f"{prefix}_em_real": em_real_sum / n,
        })
        if static_model is not None:
            metrics[f"{prefix}_em_static"] = em_static_sum / n

    # TruthfulQA abstention AUC (accuracy-coverage).
    if prefix == "truthfulqa" and tokenizer is not None and multichoice_token_id_map is not None:
        def _abstain_auc(pE: List[float], corr: List[int]) -> float:
            if not pE or not corr or len(pE) != len(corr):
                return float("nan")
            pairs = sorted(zip(pE, corr), key=lambda x: x[0])
            cum = 0
            auc_sum = 0.0
            for i, (_p, c) in enumerate(pairs, start=1):
                cum += int(c)
                auc_sum += cum / i
            return auc_sum / len(pairs)

        metrics[f"{prefix}_abstain_auc_base"] = _abstain_auc(truthfulqa_pE_base, truthfulqa_correct_base)
        metrics[f"{prefix}_abstain_auc_real"] = _abstain_auc(truthfulqa_pE_real, truthfulqa_correct_real)
        if static_model is not None:
            metrics[f"{prefix}_abstain_auc_static"] = _abstain_auc(truthfulqa_pE_static, truthfulqa_correct_static)

    if reasoning_stats is not None:
        metrics.update(_finalize_reasoning_metrics(prefix, reasoning_stats["base"], system="base"))
        metrics.update(_finalize_reasoning_metrics(prefix, reasoning_stats["real"], system="real"))
        if static_model is not None and "static" in reasoning_stats:
            metrics.update(_finalize_reasoning_metrics(prefix, reasoning_stats["static"], system="static"))

    return metrics


@torch.no_grad()
def eval_real_accept_reject(
    model: "REALWrapper",
    backbone: AutoModelForCausalLM,
    val_loader: DataLoader,
    *,
    num_batches: int,
    refine_max_steps: Optional[int],
    energy_delta_tol: float,
    patience: int,
    prefix: str = "val",
) -> Dict[str, float]:
    """
    Optional eval path for surfacing train/eval policy mismatch:
    - REAL loss using refine_adaptive() (accept/reject) prefix selection
    - mean steps used
    - mean reject rate over active steps
    """
    model.eval()

    real_sum = 0.0
    steps_sum = 0.0
    reject_rate_sum = 0.0

    n = 0
    for batch in val_loader:
        embed_weight = backbone.get_input_embeddings().weight
        dev = embed_weight.device
        model_dtype = getattr(backbone, "dtype", torch.float16)

        input_ids = batch.input_ids.to(dev)
        attention_mask = batch.attention_mask.to(dev)
        labels = batch.labels.to(dev)
        prompt_len = batch.prompt_len.to(dev)

        emb = backbone.get_input_embeddings()(input_ids)  # [B,L,D]
        B, L, _D = emb.shape

        pos = torch.arange(L, device=dev).unsqueeze(0)
        prompt_mask = (pos < prompt_len.unsqueeze(1)) & attention_mask.bool()
        h = masked_mean(emb, prompt_mask, dim=1)  # [B,D]

        prefix_final, _energies_t, _z_t, steps_used, attempts, rejects = model.head.refine_adaptive(
            h,
            max_steps=refine_max_steps,
            energy_delta_tol=energy_delta_tol,
            patience=patience,
            return_stats=True,
        )

        prefix_final = prefix_final.to(model_dtype)
        emb2 = emb.to(model_dtype)
        inputs_embeds, attn, labels2 = build_inputs_embeds_with_prefix(
            emb=emb2,
            attention_mask=attention_mask,
            labels=labels,
            prefix=prefix_final,
        )

        out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False, return_dict=True)
        loss_per_sample = causal_ce_per_sample(out.logits, labels2)  # [B]

        real_sum += float(loss_per_sample.mean().item())
        steps_sum += float(steps_used.float().mean().item())
        reject_rate_sum += float((rejects.float() / attempts.clamp(min=1).float()).mean().item())

        n += 1
        if n >= num_batches:
            break

    # restore train mode for caller
    model.train()

    n = max(n, 1)
    return {
        f"{prefix}_real_accept_reject_loss": real_sum / n,
        f"{prefix}_real_accept_reject_steps": steps_sum / n,
        f"{prefix}_real_accept_reject_reject_rate": reject_rate_sum / n,
    }


# -------------------------
# 3.6) Static-prefix baseline (prompt tuning control)
# -------------------------

class StaticPrefix(nn.Module):
    """
    A trainable prefix that does NOT depend on the input (h).
    This is the "soft prompt tuning" control baseline.

    If REAL only learns a global bias, StaticPrefix will match it.
    If REAL truly conditions on the internal vector field, it should beat StaticPrefix.
    """
    def __init__(self, prefix_len: int, d_model: int, init_std: float = 0.02):
        super().__init__()
        self.prefix = nn.Parameter(torch.zeros(prefix_len, d_model))
        nn.init.normal_(self.prefix, mean=0.0, std=init_std)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.prefix.unsqueeze(0).expand(batch_size, -1, -1)


class StaticPrefixWrapper(nn.Module):
    """
    Frozen backbone + static prefix (trainable).
    Computes CE loss (teacher-forced) with the static prefix prepended.
    """
    def __init__(self, backbone: AutoModelForCausalLM, static_prefix: StaticPrefix):
        super().__init__()
        self.backbone = backbone
        self.static_prefix = static_prefix

    def train(self, mode: bool = True):
        """Override nn.Module.train() to keep the frozen backbone in eval mode.

        StaticPrefix can be trained while the backbone remains deterministic.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def _build_prefixed_inputs(
        self,
        prefix: torch.Tensor,             # [B, P, D]
        emb: torch.Tensor,                # [B, L, D]
        attention_mask: torch.Tensor,     # [B, L]
        labels: torch.Tensor,             # [B, L]
    ):
        inputs_embeds, attn, labels2 = build_inputs_embeds_with_prefix(
            emb=emb,
            attention_mask=attention_mask,
            labels=labels,
            prefix=prefix,
        )
        return inputs_embeds, attn, labels2

    def forward(self, batch: Batch) -> torch.Tensor:
        emb_w = self.backbone.get_input_embeddings().weight
        dev = emb_w.device
        model_dtype = getattr(self.backbone, "dtype", torch.float16)

        input_ids = batch.input_ids.to(dev)
        attention_mask = batch.attention_mask.to(dev)
        labels = batch.labels.to(dev)

        # Token embeddings do not need gradients
        with torch.no_grad():
            emb = self.backbone.get_input_embeddings()(input_ids).to(model_dtype)

        B, L, D = emb.shape
        prefix = self.static_prefix(B).to(model_dtype)

        inputs_embeds, attn, labels2 = self._build_prefixed_inputs(prefix, emb, attention_mask, labels)

        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )

        loss_per_sample = causal_ce_per_sample(out.logits, labels2)
        return loss_per_sample.mean()


def train_static_prefix(
    static_model: StaticPrefixWrapper,
    static_prefix: StaticPrefix,
    train_loader: DataLoader,
    steps: int,
    lr: float,
    grad_accum: int,
    log_every: int = 100,
):
    """
    Trains a static prefix baseline for a fixed number of microsteps.
    Backbone remains frozen.
    """
    static_model.train()
    opt = torch.optim.AdamW(static_prefix.parameters(), lr=lr, weight_decay=0.0)
    opt.zero_grad(set_to_none=True)

    t = 0
    while t < steps:
        for batch in train_loader:
            loss = static_model(batch) / grad_accum
            loss.backward()

            if (t + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(static_prefix.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            if t % log_every == 0:
                print(f"[StaticPrefix] step={t:05d} lm={loss.item() * grad_accum:.3f}")

            t += 1
            if t >= steps:
                break

    static_model.eval()
    for p in static_prefix.parameters():
        p.requires_grad = False


@dataclass
class TrainingConfig:
    """CLI/config snapshot for training, probe eval, and inference-policy eval."""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_length: int = 256
    max_samples: int = 12000
    val_max_samples: int = 2000
    train_dataset_name: str = "wikitext"
    train_dataset_config_name: str = "wikitext-2-raw-v1"
    val_dataset_name: str = "wikitext"
    val_dataset_config_name: str = "wikitext-2-raw-v1"
    p_drop_word: float = 0.18
    p_swap_adj: float = 0.06
    val_p_drop_word: float = 0.18
    val_p_swap_adj: float = 0.06
    answer_loss_tokens: int = 64
    short_answer_loss_tokens: int = 8
    batch_size: int = 2
    val_batch_size: int = 8
    grad_accum: int = 8
    train_steps: int = 800
    eval_every: int = 200
    eval_batches: int = 10
    eval_inference_policy: str = "none"
    eval_boolq: bool = False
    eval_boolq_max_samples: int = 500
    eval_mc: bool = False
    mc_dataset_name: str = "ai2_arc"
    mc_dataset_config_name: str = "ARC-Challenge"
    eval_mc_max_samples: int = 500
    eval_summarization: bool = False
    summarization_max_samples: int = 200
    summarization_loss_tokens: int = 32
    eval_extractive_qa: bool = False
    extractive_qa_max_samples: int = 200
    extractive_qa_loss_tokens: int = 16
    eval_mmlu_pro: bool = False
    eval_mmlu_pro_max_samples: int = 500
    mmlu_pro_eval_split: str = "test"
    mmlu_pro_dataset_name: str = "TIGER-Lab/MMLU-Pro"
    eval_gpqa_diamond: bool = False
    eval_gpqa_diamond_max_samples: int = 200
    gpqa_diamond_eval_split: str = "train"
    gpqa_diamond_dataset_name: str = "jinulee-v/gpqa-diamond"
    gpqa_diamond_dataset_config_name: Optional[str] = None
    eval_aime: bool = False
    eval_aime_max_samples: int = 30
    aime_eval_split: str = "train"
    aime_dataset_name: str = "HuggingFaceH4/aime_2024"
    aime_dataset_config_name: Optional[str] = None
    eval_hotpotqa: bool = False
    eval_hotpotqa_max_samples: int = 200
    hotpotqa_eval_split: str = "validation"
    hotpotqa_dataset_name: str = "hotpotqa/hotpot_qa"
    hotpotqa_dataset_config_name: Optional[str] = "distractor"
    hotpotqa_loss_tokens: int = 16
    eval_truthfulqa: bool = False
    eval_truthfulqa_max_samples: int = 500
    truthfulqa_eval_split: str = "validation"
    truthfulqa_add_abstain: bool = True
    truthfulqa_dataset_name: str = "EleutherAI/truthful_qa_mc"
    truthfulqa_dataset_config_name: str = "multiple_choice"
    eval_squad2_reasoning: bool = False
    squad2_reasoning_max_samples: int = 200
    squad2_reasoning_max_length: int = 384
    squad2_reasoning_loss_tokens: int = 16
    squad2_reasoning_split: str = "validation"
    squad2_reasoning_dataset_name: str = "rajpurkar/squad_v2"
    eval_fever_reasoning: bool = False
    fever_reasoning_max_samples: int = 500
    fever_reasoning_max_length: int = 320
    fever_reasoning_loss_tokens: int = 4
    fever_reasoning_split: str = "dev"
    fever_reasoning_dataset_name: str = "pietrolesci/nli_fever"
    fever_reasoning_dataset_config_name: Optional[str] = None
    hotpotqa_reasoning_max_samples: int = 128
    hotpotqa_reasoning_max_length: int = 512
    hotpotqa_reasoning_loss_tokens: int = 96
    hotpotqa_reasoning_split: str = "validation"
    hotpotqa_reasoning_dataset_name: str = "hotpotqa/hotpot_qa"
    hotpotqa_reasoning_dataset_config_name: Optional[str] = "distractor"
    hotpotqa_reasoning_output_contract: str = "default"
    reasoning_eval_mode: bool = False
    allow_abstain: bool = False
    abstain_text: str = DEFAULT_CONTEXT_ABSTAIN_TEXT
    abstain_option_letter: str = "E"
    squad2_reasoning_train_ratio: float = 0.0
    squad2_reasoning_train_max_samples: int = 2000
    squad2_reasoning_train_loss_tokens: int = 16
    squad2_reasoning_train_split: str = "train"
    fever_reasoning_train_ratio: float = 0.0
    fever_reasoning_train_max_samples: int = 2000
    fever_reasoning_train_loss_tokens: int = 4
    fever_reasoning_train_split: str = "train"
    insufficiency_contact_objective: bool = False
    insufficiency_contact_weight: float = 0.0
    insufficiency_contact_margin: float = 0.0
    insufficiency_contact_steps: str = "final"
    insufficiency_contact_max_candidates: int = 2
    insufficiency_contact_apply_to_supported: bool = True
    insufficiency_contact_apply_to_insufficient: bool = True
    insufficiency_contact_hard_negatives: bool = False
    insufficiency_contact_hard_negative_pool_size: int = 8
    insufficiency_contact_hard_negative_top_k: int = 1
    insufficiency_contact_hard_negative_refresh_steps: int = 1
    insufficiency_contact_hard_negative_selection_mode: str = "per_step"
    insufficiency_contact_negative_source: str = "train_split_bank"
    insufficiency_contact_loss_mode: str = "legacy"
    insufficiency_contact_tolerance: float = 0.25
    insufficiency_contact_supported_guard_weight: float = 0.0
    insufficiency_contact_supported_margin_floor: float = 0.25
    insufficiency_contact_positive_nll_weight: float = 0.0
    insufficiency_contact_path_allowed_drift: float = 0.0
    insufficiency_contact_checkpoint_candidates: bool = False
    squad_train_ratio: float = 0.0
    squad_train_max_samples: int = 2000
    squad_train_max_length: int = 384
    squad_train_answer_loss_tokens: int = 16
    squad_train_split: str = "train"
    boolq_train_ratio: float = 0.0
    boolq_train_max_samples: int = 2000
    boolq_train_max_length: int = 160
    boolq_train_answer_loss_tokens: int = 4
    boolq_train_split: str = "train"
    mc_train_ratio: float = 0.0
    mc_train_max_samples: int = 2000
    mc_train_max_length: int = 256
    mc_train_answer_loss_tokens: int = 4
    mc_train_split: str = "train"
    mmlu_pro_train_ratio: float = 0.0
    mmlu_pro_train_max_samples: int = 2000
    mmlu_pro_train_max_length: int = 256
    mmlu_pro_train_answer_loss_tokens: int = 4
    mmlu_pro_train_split: str = "train"
    gpqa_diamond_train_ratio: float = 0.0
    gpqa_diamond_train_max_samples: int = 2000
    gpqa_diamond_train_max_length: int = 256
    gpqa_diamond_train_answer_loss_tokens: int = 4
    gpqa_diamond_train_split: str = "train"
    aime_train_ratio: float = 0.0
    aime_train_max_samples: int = 30
    aime_train_max_length: int = 256
    aime_train_answer_loss_tokens: int = 8
    aime_train_split: str = "train"
    hotpotqa_train_ratio: float = 0.0
    hotpotqa_train_max_samples: int = 2000
    hotpotqa_train_max_length: int = 320
    hotpotqa_train_answer_loss_tokens: int = 16
    hotpotqa_train_split: str = "train"
    truthfulqa_train_ratio: float = 0.0
    truthfulqa_train_max_samples: int = 2000
    truthfulqa_train_max_length: int = 256
    truthfulqa_train_answer_loss_tokens: int = 4
    truthfulqa_train_split: str = "validation"
    train_static_prefix: bool = True
    static_prefix_ckpt: Optional[str] = None
    static_train_steps: int = 800
    static_lr: float = 2e-4
    prefix_len: int = 16
    num_steps: int = 4
    num_particles: int = 4
    latent_dim: int = 256
    lr: float = 2e-4
    seed: int = 0
    val_seed: int = 1
    output_dir: str = "outputs"
    run_id: Optional[str] = None
    save_every: Optional[int] = None
    resume_from: Optional[str] = None
    baseline_path: Optional[str] = None
    write_base_baseline: bool = False
    inference_refine_policy: str = "accept_reject_v1"
    inference_refine_max_steps: Optional[int] = None
    inference_refine_energy_delta_tol: float = 1e-3
    inference_refine_patience: int = 1
    sample_generations: int = 0


def _timestamp() -> str:
    return datetime.utcnow().isoformat()


def save_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def save_head_checkpoint(head: REALHead, opt: Optional[torch.optim.Optimizer], step: int, path: Path, config: TrainingConfig, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "head": head.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "step": step,
        "seed": seed,
        "config": asdict(config),
    }
    torch.save(state, path)


def load_head_checkpoint(head: REALHead, opt: Optional[torch.optim.Optimizer], path: Path):
    state = torch.load(path, map_location="cpu")
    head.load_state_dict(state["head"])
    if opt is not None and state.get("optimizer") is not None:
        opt.load_state_dict(state["optimizer"])
    return state.get("step", 0), state.get("seed")


def save_static_checkpoint(static_prefix: StaticPrefix, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"static_prefix": static_prefix.state_dict()}, path)


def load_static_checkpoint(static_prefix: StaticPrefix, path: Path):
    state = torch.load(path, map_location="cpu")
    static_prefix.load_state_dict(state["static_prefix"])


def ensure_backbone_frozen(backbone: AutoModelForCausalLM):
    for p in backbone.parameters():
        p.requires_grad = False
    assert not any(p.requires_grad for p in backbone.parameters()), "Backbone parameters must remain frozen"


def compute_baseline_only(backbone: AutoModelForCausalLM, val_loader: DataLoader, num_batches: int) -> float:
    total = 0.0
    n = 0
    for batch in val_loader:
        total += float(baseline_loss_mean(backbone, batch).item())
        n += 1
        if n >= num_batches:
            break
    return total / max(n, 1)


def baseline_signature(config: TrainingConfig, val_ds: DenoiseToOriginalDataset) -> Dict:
    return {
        "model_name": config.model_name,
        "val_seed": config.val_seed,
        "max_length": config.max_length,
        "val_max_samples": config.val_max_samples,
        "val_dataset_name": config.val_dataset_name,
        "val_dataset_config_name": config.val_dataset_config_name,
        "val_p_drop_word": config.val_p_drop_word,
        "val_p_swap_adj": config.val_p_swap_adj,
        "answer_loss_tokens": config.answer_loss_tokens,
        "eval_batches": config.eval_batches,
        "deterministic_corruption": getattr(val_ds, "deterministic_corruption", False),
    }


def verify_baseline_compatibility(baseline_ref: Dict, config: TrainingConfig, val_ds: DenoiseToOriginalDataset) -> bool:
    sig = baseline_signature(config, val_ds)
    for k, expected in sig.items():
        if baseline_ref.get(k) != expected:
            print(
                f"[Baseline] Incompatible baseline for {k}: expected {expected}, got {baseline_ref.get(k)}. "
                "Skipping baseline deltas."
            )
            return False
    return True


def sample_generations(
    tokenizer: AutoTokenizer,
    backbone: AutoModelForCausalLM,
    model: REALWrapper,
    static_model: Optional[StaticPrefixWrapper],
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    num_samples: int = 0,
    max_new_tokens: int = 64,
):
    if num_samples <= 0:
        return

    print("\n[Samples] Generations (base/static/real)")
    model.eval()
    if static_model is not None:
        static_model.eval()

    pad = tokenizer.pad_token_id
    it = iter(val_loader)
    emb_w = backbone.get_input_embeddings().weight
    embed_device = emb_w.device
    for _ in range(num_samples):
        try:
            batch = next(it)
        except StopIteration:
            break

        input_ids = batch.input_ids[:1]
        attention_mask = batch.attention_mask[:1]
        labels = batch.labels[:1]
        prompt_len = batch.prompt_len[:1]

        prompt_ids = input_ids[0, : prompt_len.item()]
        target_ids = labels[0]
        target_ids = target_ids[target_ids != -100]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

        prompt_ids_only = input_ids[:, : prompt_len.item()]
        prompt_attn_only = attention_mask[:, : prompt_len.item()]

        base_gen = "<base generation failed>"
        try:
            with torch.no_grad():
                base_out = backbone.generate(
                    input_ids=prompt_ids_only.to(embed_device),
                    attention_mask=prompt_attn_only.to(embed_device),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad,
                )
                base_gen = tokenizer.decode(base_out[0][prompt_len.item():], skip_special_tokens=True)
        except Exception as e:
            print(f"[Samples] Base generation error: {e}")

        static_gen = None
        if static_model is not None:
            try:
                with torch.no_grad():
                    model_dtype = getattr(backbone, "dtype", torch.float16)
                    emb = backbone.get_input_embeddings()(prompt_ids_only.to(embed_device)).to(model_dtype)
                    prefix = static_model.static_prefix(1).to(model_dtype)
                    attn_mask = prompt_attn_only.to(embed_device)
                    lbl = labels[:, : prompt_len.item()].to(embed_device)
                    inputs_embeds, attn, _ = static_model._build_prefixed_inputs(prefix, emb, attn_mask, lbl)
                    out_ids = backbone.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=pad,
                    )
                    # drop prefix tokens
                    gen_part = out_ids[0, prefix.size(1) + prompt_len.item() :]
                    static_gen = tokenizer.decode(gen_part, skip_special_tokens=True)
            except Exception as e:
                print(f"[Samples] Static generation error: {e}")

        real_gen = "<real generation failed>"
        try:
            with torch.no_grad():
                model_dtype = getattr(backbone, "dtype", torch.float16)
                emb = backbone.get_input_embeddings()(prompt_ids_only.to(embed_device)).to(model_dtype)
                pos = torch.arange(emb.size(1), device=emb.device).unsqueeze(0)
                prompt_mask = (pos < prompt_len.unsqueeze(1).to(emb.device)) & prompt_attn_only.to(emb.device).bool()
                h = masked_mean(emb, prompt_mask, dim=1)
                refine_policy = (getattr(config, "inference_refine_policy", "accept_reject_v1") or "accept_reject_v1")
                if refine_policy in {"forward", "fixed", "fixed_steps"}:
                    prefix_final, _, _ = model.head(h)
                else:
                    prefix_final, _, _, _ = model.head.refine_adaptive(
                        h,
                        max_steps=getattr(config, "inference_refine_max_steps", None),
                        energy_delta_tol=getattr(config, "inference_refine_energy_delta_tol", 1e-3),
                        patience=getattr(config, "inference_refine_patience", 1),
                    )
                prefix_final = prefix_final.to(model_dtype)
                inputs_embeds, attn, _ = model._build_prefixed_inputs(
                    prefix_final,
                    emb,
                    prompt_attn_only.to(emb.device),
                    labels[:, : prompt_len.item()].to(emb.device),
                )
                out_ids = backbone.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad,
                )
                gen_part = out_ids[0, prefix_final.size(1) + prompt_len.item() :]
                real_gen = tokenizer.decode(gen_part, skip_special_tokens=True)
        except Exception as e:
            print(f"[Samples] REAL generation error: {e}")

        print("-----")
        print(f"PROMPT:\n{prompt_text}")
        print(f"TARGET:\n{target_text}")
        print(f"BASE:\n{base_gen}")
        if static_gen is not None:
            print(f"STATIC:\n{static_gen}")
        print(f"REAL:\n{real_gen}\n")

    model.train()
    if static_model is not None:
        static_model.train()


def run_eval_and_log(
    model: REALWrapper,
    backbone: AutoModelForCausalLM,
    val_loader: DataLoader,
    metrics_path: Path,
    config: TrainingConfig,
    step: int,
    phase: str,
    static_model: Optional[StaticPrefixWrapper] = None,
    baseline_ref: Optional[Dict] = None,
    extra_eval_loaders: Optional[List[Tuple[str, DataLoader]]] = None,
    tokenizer=None,
):
    with temp_seed(config.val_seed):
        metrics = eval_real(
            model,
            backbone,
            val_loader,
            static_model=static_model,
            num_batches=config.eval_batches,
            prefix="val",
            tokenizer=tokenizer,
        )
    metrics.update(
        {
            "run_id": config.run_id,
            "step": step,
            "split": "val",
            "phase": phase,
            "timestamp": _timestamp(),
            "model_name": config.model_name,
            "seed": config.seed,
        }
    )
    _attach_task_eval_metadata(metrics, prefix="val", config=config)

    if extra_eval_loaders:
        for name, loader in extra_eval_loaders:
            with temp_seed(config.val_seed):
                task_metrics = eval_real(
                    model,
                    backbone,
                    loader,
                    static_model=static_model,
                    num_batches=config.eval_batches,
                    prefix=name,
                    tokenizer=tokenizer,
                )
            _attach_task_eval_metadata(task_metrics, prefix=name, config=config)
            metrics.update(task_metrics)

    eval_inf_policy = (getattr(config, "eval_inference_policy", "none") or "none").strip().lower()
    if eval_inf_policy not in {"none", "accept_reject_v1"}:
        raise ValueError(f"Unknown eval_inference_policy={eval_inf_policy!r} (expected none|accept_reject_v1)")
    if eval_inf_policy == "accept_reject_v1":
        with temp_seed(config.val_seed):
            ar_metrics = eval_real_accept_reject(
                model,
                backbone,
                val_loader,
                num_batches=config.eval_batches,
                refine_max_steps=getattr(config, "inference_refine_max_steps", None),
                energy_delta_tol=getattr(config, "inference_refine_energy_delta_tol", 1e-3),
                patience=getattr(config, "inference_refine_patience", 1),
                prefix="val",
            )
        metrics.update(ar_metrics)

    if baseline_ref is not None and "val_base" in baseline_ref:
        base_ref = baseline_ref["val_base"]
        metrics["delta_real_vs_saved_base"] = base_ref - metrics.get("val_real", float("nan"))
        if "val_static" in metrics:
            metrics["delta_static_vs_saved_base"] = base_ref - metrics.get("val_static", float("nan"))

    append_jsonl(metrics_path, metrics)

    base_msg = (
        f"[VAL@{step:05d}][{phase}] base={metrics['val_base']:.3f} "
        f"static={metrics.get('val_static', float('nan')):.3f} real={metrics['val_real']:.3f} "
        f"gainS={metrics.get('val_gain_static', float('nan')):.3f} gainR={metrics['val_gain_real']:.3f} "
        f"cond={metrics.get('val_gain_conditional', float('nan')):.3f} "
        f"delta_true={metrics['val_delta_true']:.4f} corr={metrics['val_corr']:.3f} e_mse={metrics['val_e_mse']:.3f} "
        f"sup={metrics.get('val_sup_tokens', float('nan')):.1f} z0={metrics.get('val_zero_sup_frac', float('nan')):.2f}"
    )
    if "val_real_accept_reject_loss" in metrics:
        base_msg = (
            base_msg
            + " "
            + f"ar={metrics.get('val_real_accept_reject_loss', float('nan')):.3f}"
            + f" ar_steps={metrics.get('val_real_accept_reject_steps', float('nan')):.2f}"
            + f" ar_rej={metrics.get('val_real_accept_reject_reject_rate', float('nan')):.2f}"
        )

    extra_msgs = []
    if extra_eval_loaders:
        for name, _ in extra_eval_loaders:
            base_k = f"{name}_base"
            static_k = f"{name}_static"
            real_k = f"{name}_real"
            cond_k = f"{name}_gain_conditional"
            msg = (
                f"[{name}] base={metrics.get(base_k, float('nan')):.3f} "
                f"static={metrics.get(static_k, float('nan')):.3f} "
                f"real={metrics.get(real_k, float('nan')):.3f} "
                f"cond={metrics.get(cond_k, float('nan')):.3f} "
                f"sup={metrics.get(f'{name}_sup_tokens', float('nan')):.1f} z0={metrics.get(f'{name}_zero_sup_frac', float('nan')):.2f}"
            )
            if name in MULTICHOICE_PREFIXES:
                acc = metrics.get(f"{name}_acc_real", float("nan"))
                try:
                    msg = msg + f" acc={float(acc):.3f}"
                except Exception:
                    pass
                if name == "truthfulqa" and f"{name}_abstain_auc_real" in metrics:
                    auc = metrics.get(f"{name}_abstain_auc_real", float("nan"))
                    try:
                        msg = msg + f" auc={float(auc):.3f}"
                    except Exception:
                        pass
            if name == "aime" and f"{name}_em_real" in metrics:
                em = metrics.get(f"{name}_em_real", float("nan"))
                try:
                    msg = msg + f" em={float(em):.3f}"
                except Exception:
                    pass
            if name == "squad2_reasoning":
                ans_acc = metrics.get(f"{name}_answerable_accuracy_real", float("nan"))
                urec = metrics.get(f"{name}_uncertainty_recall_real", float("nan"))
                fcr = metrics.get(f"{name}_false_collapse_rate_real", float("nan"))
                try:
                    msg = msg + f" ans_acc={float(ans_acc):.3f} urec={float(urec):.3f} fcr={float(fcr):.3f}"
                except Exception:
                    pass
            if name == "fever_reasoning":
                acc = metrics.get(f"{name}_overall_label_accuracy_real", float("nan"))
                urec = metrics.get(f"{name}_uncertainty_recall_real", float("nan"))
                fcr = metrics.get(f"{name}_false_collapse_rate_real", float("nan"))
                try:
                    msg = msg + f" acc={float(acc):.3f} urec={float(urec):.3f} fcr={float(fcr):.3f}"
                except Exception:
                    pass
            extra_msgs.append(msg)

    print(" ".join([base_msg] + extra_msgs))


# -------------------------
# 4) Train
# -------------------------

def main(config: TrainingConfig):
    set_seed(config.seed)
    if config.run_id is None:
        config.run_id = time.strftime("%Y%m%d_%H%M%S")
    if config.insufficiency_contact_steps not in {"final", "all", "early_final", "entry_final", "entry_mid_final"}:
        raise ValueError("insufficiency_contact_steps must be one of: final, all, early_final, entry_final, entry_mid_final")
    if config.insufficiency_contact_max_candidates < 0:
        raise ValueError("insufficiency_contact_max_candidates must be non-negative")
    if config.insufficiency_contact_loss_mode not in {"legacy", "threshold_contrastive"}:
        raise ValueError("insufficiency_contact_loss_mode must be one of: legacy, threshold_contrastive")
    if config.insufficiency_contact_hard_negative_selection_mode not in {
        "per_step",
        "final_step",
        "entry_step",
        "max_over_steps_selected",
    }:
        raise ValueError(
            "insufficiency_contact_hard_negative_selection_mode must be one of: "
            "per_step, final_step, entry_step, max_over_steps_selected"
        )
    if int(config.insufficiency_contact_hard_negative_refresh_steps) != 1:
        raise ValueError("insufficiency_contact_hard_negative_refresh_steps > 1 is not implemented yet; use 1.")
    if config.insufficiency_contact_objective and config.insufficiency_contact_weight <= 0.0:
        print("[ContactObjective] Objective flag is true but weight <= 0.0; the contact loss is inert.")
    if config.insufficiency_contact_checkpoint_candidates:
        print("[ContactObjective] Candidate scoring activation checkpointing is enabled.")

    run_dir = Path(config.output_dir) / config.run_id
    checkpoints_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    base_baseline_path = run_dir / "base_baseline.json"

    print(f"Run directory: {run_dir}")

    # 4-bit quantization for Colab-friendliness
    compute_dtype = auto_dtype()
    print(f"[dtype] compute_dtype={compute_dtype}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=compute_dtype,
    )
    backbone.config.use_cache = False
    backbone.eval()
    ensure_backbone_frozen(backbone)

    # Head must live on the same device as token embeddings
    emb_dev = backbone.get_input_embeddings().weight.device
    d_model = backbone.get_input_embeddings().embedding_dim

    head = REALHead(
        d_model=d_model,
        latent_dim=config.latent_dim,
        prefix_len=config.prefix_len,
        num_steps=config.num_steps,
        num_particles=config.num_particles,
        init_noise=0.25,
        step_scale=0.35,
        step_embed_dim=16,
    ).to(emb_dev)

    model = REALWrapper(
        backbone,
        head,
        energy_weight=0.50,
        mono_weight=0.02,
        tokenizer=tokenizer,
        insufficiency_contact_objective=config.insufficiency_contact_objective,
        insufficiency_contact_weight=config.insufficiency_contact_weight,
        insufficiency_contact_margin=config.insufficiency_contact_margin,
        insufficiency_contact_steps=config.insufficiency_contact_steps,
        insufficiency_contact_max_candidates=config.insufficiency_contact_max_candidates,
        insufficiency_contact_apply_to_supported=config.insufficiency_contact_apply_to_supported,
        insufficiency_contact_apply_to_insufficient=config.insufficiency_contact_apply_to_insufficient,
        insufficiency_contact_hard_negatives=config.insufficiency_contact_hard_negatives,
        insufficiency_contact_hard_negative_pool_size=config.insufficiency_contact_hard_negative_pool_size,
        insufficiency_contact_hard_negative_top_k=config.insufficiency_contact_hard_negative_top_k,
        insufficiency_contact_hard_negative_refresh_steps=config.insufficiency_contact_hard_negative_refresh_steps,
        insufficiency_contact_hard_negative_selection_mode=config.insufficiency_contact_hard_negative_selection_mode,
        insufficiency_contact_negative_source=config.insufficiency_contact_negative_source,
        insufficiency_contact_loss_mode=config.insufficiency_contact_loss_mode,
        insufficiency_contact_tolerance=config.insufficiency_contact_tolerance,
        insufficiency_contact_supported_guard_weight=config.insufficiency_contact_supported_guard_weight,
        insufficiency_contact_supported_margin_floor=config.insufficiency_contact_supported_margin_floor,
        insufficiency_contact_positive_nll_weight=config.insufficiency_contact_positive_nll_weight,
        insufficiency_contact_path_allowed_drift=config.insufficiency_contact_path_allowed_drift,
        insufficiency_contact_checkpoint_candidates=config.insufficiency_contact_checkpoint_candidates,
    ).to(emb_dev)

    static_prefix = None
    static_model = None
    if config.train_static_prefix:
        static_prefix = StaticPrefix(prefix_len=config.prefix_len, d_model=d_model).to(emb_dev)
        static_model = StaticPrefixWrapper(backbone, static_prefix).to(emb_dev)
    elif config.static_prefix_ckpt:
        static_ckpt = Path(config.static_prefix_ckpt)
        if static_ckpt.is_file():
            static_prefix = StaticPrefix(prefix_len=config.prefix_len, d_model=d_model).to(emb_dev)
            load_static_checkpoint(static_prefix, static_ckpt)
            for p in static_prefix.parameters():
                p.requires_grad = False
            static_model = StaticPrefixWrapper(backbone, static_prefix).to(emb_dev)
            static_model.eval()
            print(f"[StaticPrefix] Loaded checkpoint from {static_ckpt}")
        else:
            print(f"[StaticPrefix] checkpoint not found at {static_ckpt}; static baseline disabled")

    train_ds = DenoiseToOriginalDataset(
        tokenizer=tokenizer,
        split="train",
        max_length=config.max_length,
        p_drop_word=config.p_drop_word,
        p_swap_adj=config.p_swap_adj,
        seed=config.seed,
        max_samples=config.max_samples,
        answer_loss_tokens=config.answer_loss_tokens,
        deterministic_corruption=False,
        dataset_name=config.train_dataset_name,
        dataset_config_name=config.train_dataset_config_name,
    )

    def _collate(examples):
        return collate_batch(examples, pad_id=tokenizer.pad_token_id)

    denoise_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate)

    ratio_fields = [
        ("squad2_reasoning_train_ratio", config.squad2_reasoning_train_ratio),
        ("fever_reasoning_train_ratio", config.fever_reasoning_train_ratio),
        ("squad_train_ratio", config.squad_train_ratio),
        ("boolq_train_ratio", config.boolq_train_ratio),
        ("mc_train_ratio", config.mc_train_ratio),
        ("mmlu_pro_train_ratio", config.mmlu_pro_train_ratio),
        ("gpqa_diamond_train_ratio", config.gpqa_diamond_train_ratio),
        ("aime_train_ratio", config.aime_train_ratio),
        ("hotpotqa_train_ratio", config.hotpotqa_train_ratio),
        ("truthfulqa_train_ratio", config.truthfulqa_train_ratio),
    ]
    if any(v < 0 for _k, v in ratio_fields):
        raise ValueError("train ratios must be non-negative")

    sum_ratios = sum(v for _k, v in ratio_fields)
    denoise_ratio = 1.0 - sum_ratios
    if denoise_ratio < 0:
        keys = ", ".join([k for k, _v in ratio_fields])
        raise ValueError(f"{keys} must sum to <= 1.0")

    squad_loader = None
    if config.squad_train_ratio > 0:
        squad_ds = ExtractiveQADataset(
            tokenizer=tokenizer,
            split=config.squad_train_split,
            max_length=config.squad_train_max_length,
            max_samples=config.squad_train_max_samples,
            answer_loss_tokens=config.squad_train_answer_loss_tokens,
        )
        if len(squad_ds) == 0:
            raise ValueError("squad_train_ratio > 0 but SQuAD training set is empty")
        squad_loader = DataLoader(squad_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate)

    squad2_reasoning_train_loader = None
    if config.squad2_reasoning_train_ratio > 0:
        squad2_reasoning_train_ds = SQuAD2ReasoningDataset(
            tokenizer=tokenizer,
            split=config.squad2_reasoning_train_split,
            max_length=config.squad2_reasoning_max_length,
            max_samples=config.squad2_reasoning_train_max_samples,
            answer_loss_tokens=config.squad2_reasoning_train_loss_tokens,
            abstain_text=config.abstain_text,
            dataset_name=config.squad2_reasoning_dataset_name,
            build_insufficiency_negatives=True,
            insufficiency_negative_max_candidates=max(
                2,
                config.insufficiency_contact_max_candidates,
                config.insufficiency_contact_hard_negative_pool_size
                if config.insufficiency_contact_hard_negatives
                else 0,
            ),
        )
        if len(squad2_reasoning_train_ds) == 0:
            raise ValueError("squad2_reasoning_train_ratio > 0 but SQuAD2 reasoning training set is empty")
        squad2_reasoning_train_loader = DataLoader(
            squad2_reasoning_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        )

    boolq_train_loader = None
    if config.boolq_train_ratio > 0:
        boolq_train_ds = BoolQShortAnswerDataset(
            tokenizer=tokenizer,
            split=config.boolq_train_split,
            max_length=config.boolq_train_max_length,
            max_samples=config.boolq_train_max_samples,
            answer_loss_tokens=config.boolq_train_answer_loss_tokens,
        )
        if len(boolq_train_ds) == 0:
            raise ValueError("boolq_train_ratio > 0 but BoolQ training set is empty")
        boolq_train_loader = DataLoader(boolq_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate)

    fever_reasoning_train_loader = None
    if config.fever_reasoning_train_ratio > 0:
        fever_reasoning_train_ds = FEVERReasoningDataset(
            tokenizer=tokenizer,
            split=config.fever_reasoning_train_split,
            max_length=config.fever_reasoning_max_length,
            max_samples=config.fever_reasoning_train_max_samples,
            answer_loss_tokens=config.fever_reasoning_train_loss_tokens,
            dataset_name=config.fever_reasoning_dataset_name,
            dataset_config_name=config.fever_reasoning_dataset_config_name,
        )
        if len(fever_reasoning_train_ds) == 0:
            raise ValueError("fever_reasoning_train_ratio > 0 but FEVER reasoning training set is empty")
        fever_reasoning_train_loader = DataLoader(
            fever_reasoning_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        )


    mc_train_loader = None
    if config.mc_train_ratio > 0:
        mc_train_ds = MultipleChoiceLetterDataset(
            tokenizer=tokenizer,
            split=config.mc_train_split,
            max_length=config.mc_train_max_length,
            max_samples=config.mc_train_max_samples,
            dataset_name=config.mc_dataset_name,
            dataset_config_name=config.mc_dataset_config_name,
            answer_loss_tokens=config.mc_train_answer_loss_tokens,
        )
        if len(mc_train_ds) == 0:
            raise ValueError("mc_train_ratio > 0 but ARC training set is empty")
        mc_train_loader = DataLoader(mc_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate)

    mmlu_pro_train_loader = None
    if config.mmlu_pro_train_ratio > 0:
        mmlu_pro_train_ds = MMLUProLetterDataset(
            tokenizer=tokenizer,
            split=config.mmlu_pro_train_split,
            max_length=config.mmlu_pro_train_max_length,
            max_samples=config.mmlu_pro_train_max_samples,
            answer_loss_tokens=config.mmlu_pro_train_answer_loss_tokens,
            dataset_name=config.mmlu_pro_dataset_name,
        )
        if len(mmlu_pro_train_ds) == 0:
            raise ValueError("mmlu_pro_train_ratio > 0 but MMLU-Pro training set is empty")
        mmlu_pro_train_loader = DataLoader(
            mmlu_pro_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        )

    gpqa_diamond_train_loader = None
    if config.gpqa_diamond_train_ratio > 0:
        gpqa_diamond_train_ds = GPQADiamondLetterDataset(
            tokenizer=tokenizer,
            split=config.gpqa_diamond_train_split,
            max_length=config.gpqa_diamond_train_max_length,
            max_samples=config.gpqa_diamond_train_max_samples,
            answer_loss_tokens=config.gpqa_diamond_train_answer_loss_tokens,
            dataset_name=config.gpqa_diamond_dataset_name,
            dataset_config_name=config.gpqa_diamond_dataset_config_name,
        )
        if len(gpqa_diamond_train_ds) == 0:
            raise ValueError("gpqa_diamond_train_ratio > 0 but GPQA-diamond training set is empty")
        gpqa_diamond_train_loader = DataLoader(
            gpqa_diamond_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        )

    aime_train_loader = None
    if config.aime_train_ratio > 0:
        aime_train_ds = AIMEShortAnswerDataset(
            tokenizer=tokenizer,
            split=config.aime_train_split,
            max_length=config.aime_train_max_length,
            max_samples=config.aime_train_max_samples,
            answer_loss_tokens=config.aime_train_answer_loss_tokens,
            dataset_name=config.aime_dataset_name,
            dataset_config_name=config.aime_dataset_config_name,
        )
        if len(aime_train_ds) == 0:
            raise ValueError("aime_train_ratio > 0 but AIME training set is empty")
        aime_train_loader = DataLoader(aime_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate)

    hotpotqa_train_loader = None
    if config.hotpotqa_train_ratio > 0:
        hotpotqa_train_ds = HotpotQADataset(
            tokenizer=tokenizer,
            split=config.hotpotqa_train_split,
            max_length=config.hotpotqa_train_max_length,
            max_samples=config.hotpotqa_train_max_samples,
            answer_loss_tokens=config.hotpotqa_train_answer_loss_tokens,
            dataset_name=config.hotpotqa_dataset_name,
            dataset_config_name=config.hotpotqa_dataset_config_name,
        )
        if len(hotpotqa_train_ds) == 0:
            raise ValueError("hotpotqa_train_ratio > 0 but HotpotQA training set is empty")
        hotpotqa_train_loader = DataLoader(
            hotpotqa_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        )

    truthfulqa_train_loader = None
    if config.truthfulqa_train_ratio > 0:
        truthfulqa_train_ds = TruthfulQAMCDataset(
            tokenizer=tokenizer,
            split=config.truthfulqa_train_split,
            max_length=config.truthfulqa_train_max_length,
            max_samples=config.truthfulqa_train_max_samples,
            answer_loss_tokens=config.truthfulqa_train_answer_loss_tokens,
            add_abstain=config.truthfulqa_add_abstain,
            dataset_name=config.truthfulqa_dataset_name,
            dataset_config_name=config.truthfulqa_dataset_config_name,
        )
        if len(truthfulqa_train_ds) == 0:
            raise ValueError("truthfulqa_train_ratio > 0 but TruthfulQA training set is empty")
        truthfulqa_train_loader = DataLoader(
            truthfulqa_train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate
        )
    def _infinite(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    train_iters: List[Tuple[str, object, float]] = []
    if denoise_ratio > 0:
        train_iters.append(("denoise", _infinite(denoise_loader), denoise_ratio))
    if config.squad2_reasoning_train_ratio > 0 and squad2_reasoning_train_loader is not None:
        train_iters.append(
            ("squad2_reasoning", _infinite(squad2_reasoning_train_loader), config.squad2_reasoning_train_ratio)
        )
    if config.fever_reasoning_train_ratio > 0 and fever_reasoning_train_loader is not None:
        train_iters.append(
            ("fever_reasoning", _infinite(fever_reasoning_train_loader), config.fever_reasoning_train_ratio)
        )
    if config.squad_train_ratio > 0 and squad_loader is not None:
        train_iters.append(("squad", _infinite(squad_loader), config.squad_train_ratio))
    if config.boolq_train_ratio > 0 and boolq_train_loader is not None:
        train_iters.append(("boolq", _infinite(boolq_train_loader), config.boolq_train_ratio))
    if config.mc_train_ratio > 0 and mc_train_loader is not None:
        train_iters.append(("multichoice", _infinite(mc_train_loader), config.mc_train_ratio))
    if config.mmlu_pro_train_ratio > 0 and mmlu_pro_train_loader is not None:
        train_iters.append(("mmlu_pro", _infinite(mmlu_pro_train_loader), config.mmlu_pro_train_ratio))
    if config.gpqa_diamond_train_ratio > 0 and gpqa_diamond_train_loader is not None:
        train_iters.append(("gpqa_diamond", _infinite(gpqa_diamond_train_loader), config.gpqa_diamond_train_ratio))
    if config.aime_train_ratio > 0 and aime_train_loader is not None:
        train_iters.append(("aime", _infinite(aime_train_loader), config.aime_train_ratio))
    if config.hotpotqa_train_ratio > 0 and hotpotqa_train_loader is not None:
        train_iters.append(("hotpotqa", _infinite(hotpotqa_train_loader), config.hotpotqa_train_ratio))
    if config.truthfulqa_train_ratio > 0 and truthfulqa_train_loader is not None:
        train_iters.append(("truthfulqa", _infinite(truthfulqa_train_loader), config.truthfulqa_train_ratio))

    if not train_iters:
        raise ValueError("no training data configured; check train ratios")

    if len(train_iters) > 1:
        parts = [f"denoise={denoise_ratio:.2f}"]
        for k, v in ratio_fields:
            if v > 0:
                parts.append(f"{k.replace('_train_ratio','')}={v:.2f}")
        print("[TrainMix] " + " ".join(parts))

    train_cum = []
    total = 0.0
    for _name, _it, weight in train_iters:
        total += weight
        train_cum.append(total)

    def _next_train_batch():
        if len(train_iters) == 1:
            name = train_iters[0][0]
            return name, next(train_iters[0][1])
        r = random.random() * total
        for idx, c in enumerate(train_cum):
            if r < c:
                name = train_iters[idx][0]
                return name, next(train_iters[idx][1])
        name = train_iters[-1][0]
        return name, next(train_iters[-1][1])

    train_counts = {name: 0 for name, _it, _w in train_iters}

    val_ds = DenoiseToOriginalDataset(
        tokenizer=tokenizer,
        split="validation",
        max_length=config.max_length,
        p_drop_word=config.val_p_drop_word,
        p_swap_adj=config.val_p_swap_adj,
        seed=config.val_seed,
        max_samples=config.val_max_samples,
        answer_loss_tokens=config.answer_loss_tokens,
        deterministic_corruption=True,
        dataset_name=config.val_dataset_name,
        dataset_config_name=config.val_dataset_config_name,
    )
    val_loader = DataLoader(val_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate)

    extra_eval_loaders: List[Tuple[str, DataLoader]] = []
    if config.eval_boolq:
        boolq_ds = BoolQShortAnswerDataset(
            tokenizer=tokenizer,
            split="validation",
            max_length=min(config.max_length, 192),
            max_samples=config.eval_boolq_max_samples,
            answer_loss_tokens=config.short_answer_loss_tokens,
        )
        extra_eval_loaders.append(
            (
                "boolq",
                DataLoader(boolq_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_mc:
        mc_ds = MultipleChoiceLetterDataset(
            tokenizer=tokenizer,
            split="validation",
            max_length=min(config.max_length, 224),
            max_samples=config.eval_mc_max_samples,
            dataset_name=config.mc_dataset_name,
            dataset_config_name=config.mc_dataset_config_name,
            answer_loss_tokens=config.short_answer_loss_tokens,
        )
        extra_eval_loaders.append(
            (
                "multichoice",
                DataLoader(mc_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_summarization:
        summarization_ds = SummarizationDataset(
            tokenizer=tokenizer,
            split="validation",
            max_length=max(config.max_length, 320),
            max_samples=config.summarization_max_samples,
            answer_loss_tokens=config.summarization_loss_tokens,
        )
        extra_eval_loaders.append(
            (
                "summarization",
                DataLoader(
                    summarization_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate
                ),
            )
        )

    if config.eval_extractive_qa:
        qa_ds = ExtractiveQADataset(
            tokenizer=tokenizer,
            split="validation",
            max_length=max(config.max_length, 320),
            max_samples=config.extractive_qa_max_samples,
            answer_loss_tokens=config.extractive_qa_loss_tokens,
        )
        extra_eval_loaders.append(
            (
                "extractive_qa",
                DataLoader(qa_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_squad2_reasoning:
        squad2_ds = SQuAD2ReasoningDataset(
            tokenizer=tokenizer,
            split=config.squad2_reasoning_split,
            max_length=config.squad2_reasoning_max_length,
            max_samples=config.squad2_reasoning_max_samples,
            answer_loss_tokens=config.squad2_reasoning_loss_tokens,
            abstain_text=config.abstain_text,
            dataset_name=config.squad2_reasoning_dataset_name,
        )
        extra_eval_loaders.append(
            (
                "squad2_reasoning",
                DataLoader(squad2_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_fever_reasoning:
        fever_ds = FEVERReasoningDataset(
            tokenizer=tokenizer,
            split=config.fever_reasoning_split,
            max_length=config.fever_reasoning_max_length,
            max_samples=config.fever_reasoning_max_samples,
            answer_loss_tokens=config.fever_reasoning_loss_tokens,
            dataset_name=config.fever_reasoning_dataset_name,
            dataset_config_name=config.fever_reasoning_dataset_config_name,
        )
        extra_eval_loaders.append(
            (
                "fever_reasoning",
                DataLoader(fever_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_mmlu_pro:
        mmlu_ds = MMLUProLetterDataset(
            tokenizer=tokenizer,
            split=config.mmlu_pro_eval_split,
            max_length=min(config.max_length, 256),
            max_samples=config.eval_mmlu_pro_max_samples,
            answer_loss_tokens=config.short_answer_loss_tokens,
            dataset_name=config.mmlu_pro_dataset_name,
        )
        extra_eval_loaders.append(
            (
                "mmlu_pro",
                DataLoader(mmlu_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_gpqa_diamond:
        gpqa_ds = GPQADiamondLetterDataset(
            tokenizer=tokenizer,
            split=config.gpqa_diamond_eval_split,
            max_length=min(config.max_length, 256),
            max_samples=config.eval_gpqa_diamond_max_samples,
            answer_loss_tokens=config.short_answer_loss_tokens,
            dataset_name=config.gpqa_diamond_dataset_name,
            dataset_config_name=config.gpqa_diamond_dataset_config_name,
        )
        extra_eval_loaders.append(
            (
                "gpqa_diamond",
                DataLoader(gpqa_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_aime:
        aime_ds = AIMEShortAnswerDataset(
            tokenizer=tokenizer,
            split=config.aime_eval_split,
            max_length=min(config.max_length, 256),
            max_samples=config.eval_aime_max_samples,
            answer_loss_tokens=config.short_answer_loss_tokens,
            dataset_name=config.aime_dataset_name,
            dataset_config_name=config.aime_dataset_config_name,
        )
        extra_eval_loaders.append(
            (
                "aime",
                DataLoader(aime_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_hotpotqa:
        hotpot_ds = HotpotQADataset(
            tokenizer=tokenizer,
            split=config.hotpotqa_eval_split,
            max_length=max(config.max_length, 320),
            max_samples=config.eval_hotpotqa_max_samples,
            answer_loss_tokens=config.hotpotqa_loss_tokens,
            dataset_name=config.hotpotqa_dataset_name,
            dataset_config_name=config.hotpotqa_dataset_config_name,
        )
        extra_eval_loaders.append(
            (
                "hotpotqa",
                DataLoader(hotpot_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    if config.eval_truthfulqa:
        truthful_ds = TruthfulQAMCDataset(
            tokenizer=tokenizer,
            split=config.truthfulqa_eval_split,
            max_length=min(config.max_length, 256),
            max_samples=config.eval_truthfulqa_max_samples,
            answer_loss_tokens=config.short_answer_loss_tokens,
            add_abstain=config.truthfulqa_add_abstain,
            dataset_name=config.truthfulqa_dataset_name,
            dataset_config_name=config.truthfulqa_dataset_config_name,
        )
        extra_eval_loaders.append(
            (
                "truthfulqa",
                DataLoader(truthful_ds, batch_size=config.val_batch_size, shuffle=False, collate_fn=_collate),
            )
        )

    # Save config snapshot
    save_json(run_dir / "config.json", asdict(config))

    # Optionally resume
    opt = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=0.01)
    start_step = 0
    if config.resume_from is not None:
        ckpt_path = Path(config.resume_from)
        if ckpt_path.is_file():
            start_step, ckpt_seed = load_head_checkpoint(head, opt, ckpt_path)
            # Ensure CLI --lr always wins on resume (optimizer state loads old LR).
            for pg in opt.param_groups:
                pg["lr"] = config.lr
            if ckpt_seed is not None:
                print(f"[Resume] loaded checkpoint from {ckpt_path} (step={start_step}, seed={ckpt_seed})")
            print(f"[Resume] optimizer lr set to {config.lr}")
        else:
            print(f"[Resume] checkpoint not found at {ckpt_path}, starting fresh")

    # Baseline reference file
    baseline_ref = None
    if config.baseline_path is not None and os.path.exists(config.baseline_path):
        with open(config.baseline_path) as f:
            candidate = json.load(f)
        if verify_baseline_compatibility(candidate, config, val_ds):
            baseline_ref = candidate
        else:
            print(f"[Baseline] Incompatible baseline at {config.baseline_path}; deltas will be skipped.")

    # Base-only benchmark
    base_only = compute_baseline_only(backbone, val_loader, num_batches=config.eval_batches)
    baseline_record = {
        **baseline_signature(config, val_ds),
        "run_id": config.run_id,
        "val_base": base_only,
        "timestamp": _timestamp(),
        "val_max_samples": config.val_max_samples,
        "max_length": config.max_length,
        "answer_loss_tokens": config.answer_loss_tokens,
    }
    save_json(base_baseline_path, baseline_record)
    if config.write_base_baseline:
        print(f"[BASE] val_base={base_only:.3f} -> saved to {base_baseline_path}")
        return

    # Initial eval
    run_eval_and_log(
        model,
        backbone,
        val_loader,
        metrics_path,
        config,
        step=start_step,
        phase="init",
        static_model=static_model,
        baseline_ref=baseline_ref,
        extra_eval_loaders=extra_eval_loaders,
        tokenizer=tokenizer,
    )

    if config.train_static_prefix and static_model is not None and static_prefix is not None:
        print("[StaticPrefix] Training baseline...")
        train_static_prefix(
            static_model=static_model,
            static_prefix=static_prefix,
            train_loader=denoise_loader,
            steps=config.static_train_steps,
            lr=config.static_lr,
            grad_accum=config.grad_accum,
            log_every=100,
        )
        save_static_checkpoint(static_prefix, checkpoints_dir / "static_prefix.pt")
        run_eval_and_log(
            model,
            backbone,
            val_loader,
            metrics_path,
            config,
            step=start_step,
            phase="after_static",
            static_model=static_model,
            baseline_ref=baseline_ref,
            extra_eval_loaders=extra_eval_loaders,
            tokenizer=tokenizer,
        )

    steps = start_step
    opt.zero_grad(set_to_none=True)

    while steps < config.train_steps:
        task_name, batch = _next_train_batch()
        train_counts[task_name] = train_counts.get(task_name, 0) + 1
        out = model(batch)
        loss = out["loss"] / config.grad_accum
        if not torch.isfinite(loss):
            print(f"[FATAL] Non-finite loss at step={steps} task={task_name}. Aborting.")
            break
        loss.backward()

        if (steps + 1) % config.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if steps % 50 == 0:
            pred = out["pred_energy_mean"].tolist()
            tru = out["true_energy_mean"].tolist()
            pred_str = " ".join([f"{v:.3f}" for v in pred])
            tru_str = " ".join([f"{v:.3f}" for v in tru])
            contact_str = ""
            if config.insufficiency_contact_objective and config.insufficiency_contact_weight > 0.0:
                contact_str = (
                    f" ic={out['insufficiency_contact_loss'].item():.3f}"
                    f" ic_pos={out['insufficiency_contact_positive_nll'].item():.3f}"
                    f" ic_margin={out['insufficiency_contact_margin_loss'].item():.3f}"
                    f" ic_thr={out['insufficiency_contact_threshold_loss'].item():.3f}"
                    f" ic_guard={out['insufficiency_contact_supported_guard_loss'].item():.3f}"
                    f" ic_path={out['insufficiency_contact_path_preservation_loss'].item():.3f}"
                    f" ic_rows={out['insufficiency_contact_eligible_rows'].item():.0f}"
                    f"/{out['insufficiency_contact_insufficient_rows'].item():.0f}"
                    f"/{out['insufficiency_contact_supported_rows'].item():.0f}"
                    f" ic_missing_neg={out['insufficiency_contact_missing_negative_rows'].item():.0f}"
                    f" ic_pool={out['insufficiency_contact_candidate_pool_size'].item():.1f}"
                    f" ic_hard={out['insufficiency_contact_hard_margin'].item():.3f}"
                    f" ic_nonfirst={out['insufficiency_contact_selected_negative_nonfirst_rows'].item():.0f}"
                    f" ic_fb={out['insufficiency_contact_fallback_negative_rows'].item():.0f}"
                    f" ic_fever={out['insufficiency_contact_fever_nei_supported_selected'].item():.0f}"
                    f"/{out['insufficiency_contact_fever_nei_refuted_selected'].item():.0f}"
                )
            print(
                f"step={steps:05d} task={task_name} "
                f"lm={out['lm_loss'].item():.3f} "
                f"e_mse={out['energy_mse'].item():.3f} "
                f"mono={out['mono_pen'].item():.3f} "
                f"corr={out['corr_pred_true'].item():.3f} "
                f"p_pre={out['prefix_norm_pre_mean'].item():.2f}/{out['prefix_norm_pre_max'].item():.2f} "
                f"p_post={out['prefix_norm_post_mean'].item():.2f}/{out['prefix_norm_post_max'].item():.2f} "
                f"p_clamp={out['prefix_clamped_frac'].item():.2f}"
                f"{contact_str}\n"
                f"  pred_energy=[{pred_str}]\n"
                f"  true_energy=[{tru_str}]"
            )

        if config.save_every is not None and config.save_every > 0 and steps > 0 and steps % config.save_every == 0:
            save_head_checkpoint(head, opt, steps, checkpoints_dir / f"head_step{steps:05d}.pt", config, config.seed)

        if (steps > 0) and (steps % config.eval_every == 0):
            if len(train_counts) > 1:
                total_seen = sum(train_counts.values())
                if total_seen > 0:
                    parts = [
                        f"{name}={train_counts.get(name, 0)} ({train_counts.get(name, 0) / total_seen:.2f})"
                        for name in train_counts
                    ]
                    print("[TrainMix] interval " + " ".join(parts))
                for name in train_counts:
                    train_counts[name] = 0
            run_eval_and_log(
                model,
                backbone,
                val_loader,
                metrics_path,
                config,
                step=steps,
                phase="train",
                static_model=static_model,
                baseline_ref=baseline_ref,
                extra_eval_loaders=extra_eval_loaders,
                tokenizer=tokenizer,
            )

        steps += 1

    # Final checkpoint and eval
    save_head_checkpoint(head, opt, steps, checkpoints_dir / f"head_step{steps:05d}.pt", config, config.seed)
    run_eval_and_log(
        model,
        backbone,
        val_loader,
        metrics_path,
        config,
        step=steps,
        phase="final",
        static_model=static_model,
        baseline_ref=baseline_ref,
        extra_eval_loaders=extra_eval_loaders,
        tokenizer=tokenizer,
    )

    sample_generations(
        tokenizer=tokenizer,
        backbone=backbone,
        model=model,
        static_model=static_model,
        val_loader=val_loader,
        config=config,
        num_samples=config.sample_generations,
    )

    print("Done. (Head trained; backbone frozen.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REAL head with optional static prefix baseline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=12000)
    parser.add_argument("--val_max_samples", type=int, default=2000)
    parser.add_argument("--train_dataset_name", type=str, default="wikitext")
    parser.add_argument("--train_dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--val_dataset_name", type=str, default="wikitext")
    parser.add_argument("--val_dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--p_drop_word", type=float, default=0.18)
    parser.add_argument("--p_swap_adj", type=float, default=0.06)
    parser.add_argument("--val_p_drop_word", type=float, default=0.18)
    parser.add_argument("--val_p_swap_adj", type=float, default=0.06)
    parser.add_argument("--answer_loss_tokens", type=int, default=64)
    parser.add_argument("--short_answer_loss_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=800)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--eval_inference_policy", type=str, choices=["none", "accept_reject_v1"], default="none")
    parser.add_argument("--eval_boolq", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--eval_boolq_max_samples", type=int, default=500)
    parser.add_argument("--eval_mc", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--mc_dataset_name", type=str, default="ai2_arc")
    parser.add_argument("--mc_dataset_config_name", type=str, default="ARC-Challenge")
    parser.add_argument("--eval_mc_max_samples", type=int, default=500)
    parser.add_argument("--eval_summarization", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--summarization_max_samples", type=int, default=200)
    parser.add_argument("--summarization_loss_tokens", type=int, default=32)
    parser.add_argument("--eval_extractive_qa", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--extractive_qa_max_samples", type=int, default=200)
    parser.add_argument("--extractive_qa_loss_tokens", type=int, default=16)
    parser.add_argument("--eval_mmlu_pro", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--eval_mmlu_pro_max_samples", type=int, default=500)
    parser.add_argument("--mmlu_pro_eval_split", type=str, default="test")
    parser.add_argument("--mmlu_pro_dataset_name", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--eval_gpqa_diamond", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--eval_gpqa_diamond_max_samples", type=int, default=200)
    parser.add_argument("--gpqa_diamond_eval_split", type=str, default="train")
    parser.add_argument("--gpqa_diamond_dataset_name", type=str, default="jinulee-v/gpqa-diamond")
    parser.add_argument("--gpqa_diamond_dataset_config_name", type=str, default=None)
    parser.add_argument("--eval_aime", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--eval_aime_max_samples", type=int, default=30)
    parser.add_argument("--aime_eval_split", type=str, default="train")
    parser.add_argument("--aime_dataset_name", type=str, default="HuggingFaceH4/aime_2024")
    parser.add_argument("--aime_dataset_config_name", type=str, default=None)
    parser.add_argument("--eval_hotpotqa", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--eval_hotpotqa_max_samples", type=int, default=200)
    parser.add_argument("--hotpotqa_eval_split", type=str, default="validation")
    parser.add_argument("--hotpotqa_dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument("--hotpotqa_dataset_config_name", type=str, default="distractor")
    parser.add_argument("--hotpotqa_loss_tokens", type=int, default=16)
    parser.add_argument("--eval_truthfulqa", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--eval_truthfulqa_max_samples", type=int, default=500)
    parser.add_argument("--truthfulqa_eval_split", type=str, default="validation")
    parser.add_argument("--truthfulqa_add_abstain", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--truthfulqa_dataset_name", type=str, default="EleutherAI/truthful_qa_mc")
    parser.add_argument("--truthfulqa_dataset_config_name", type=str, default="multiple_choice")
    parser.add_argument("--eval_squad2_reasoning", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--squad2_reasoning_max_samples", type=int, default=200)
    parser.add_argument("--squad2_reasoning_max_length", type=int, default=384)
    parser.add_argument("--squad2_reasoning_loss_tokens", type=int, default=16)
    parser.add_argument("--squad2_reasoning_split", type=str, default="validation")
    parser.add_argument("--squad2_reasoning_dataset_name", type=str, default="rajpurkar/squad_v2")
    parser.add_argument("--eval_fever_reasoning", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--fever_reasoning_max_samples", type=int, default=500)
    parser.add_argument("--fever_reasoning_max_length", type=int, default=320)
    parser.add_argument("--fever_reasoning_loss_tokens", type=int, default=4)
    parser.add_argument("--fever_reasoning_split", type=str, default="dev")
    parser.add_argument("--fever_reasoning_dataset_name", type=str, default="pietrolesci/nli_fever")
    parser.add_argument("--fever_reasoning_dataset_config_name", type=str, default=None)
    parser.add_argument("--hotpotqa_reasoning_max_samples", type=int, default=128)
    parser.add_argument("--hotpotqa_reasoning_max_length", type=int, default=512)
    parser.add_argument("--hotpotqa_reasoning_loss_tokens", type=int, default=96)
    parser.add_argument("--hotpotqa_reasoning_split", type=str, default="validation")
    parser.add_argument("--hotpotqa_reasoning_dataset_name", type=str, default="hotpotqa/hotpot_qa")
    parser.add_argument("--hotpotqa_reasoning_dataset_config_name", type=str, default="distractor")
    parser.add_argument(
        "--hotpotqa_reasoning_output_contract",
        type=str,
        default="default",
        choices=list(HOTPOT_REASONING_OUTPUT_CONTRACTS),
    )
    parser.add_argument("--reasoning_eval_mode", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--allow_abstain", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--abstain_text", type=str, default=DEFAULT_CONTEXT_ABSTAIN_TEXT)
    parser.add_argument("--abstain_option_letter", type=str, default="E")
    parser.add_argument("--squad2_reasoning_train_ratio", type=float, default=0.0)
    parser.add_argument("--squad2_reasoning_train_max_samples", type=int, default=2000)
    parser.add_argument("--squad2_reasoning_train_loss_tokens", type=int, default=16)
    parser.add_argument("--squad2_reasoning_train_split", type=str, default="train")
    parser.add_argument("--fever_reasoning_train_ratio", type=float, default=0.0)
    parser.add_argument("--fever_reasoning_train_max_samples", type=int, default=2000)
    parser.add_argument("--fever_reasoning_train_loss_tokens", type=int, default=4)
    parser.add_argument("--fever_reasoning_train_split", type=str, default="train")
    parser.add_argument("--insufficiency_contact_objective", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--insufficiency_contact_weight", type=float, default=0.0)
    parser.add_argument("--insufficiency_contact_margin", type=float, default=0.0)
    parser.add_argument(
        "--insufficiency_contact_steps",
        type=str,
        default="final",
        choices=["final", "all", "early_final", "entry_final", "entry_mid_final"],
    )
    parser.add_argument("--insufficiency_contact_max_candidates", type=int, default=2)
    parser.add_argument(
        "--insufficiency_contact_apply_to_supported",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument("--insufficiency_contact_hard_negatives", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--insufficiency_contact_hard_negative_pool_size", type=int, default=8)
    parser.add_argument("--insufficiency_contact_hard_negative_top_k", type=int, default=1)
    parser.add_argument("--insufficiency_contact_hard_negative_refresh_steps", type=int, default=1)
    parser.add_argument(
        "--insufficiency_contact_hard_negative_selection_mode",
        type=str,
        default="per_step",
        choices=["per_step", "final_step", "entry_step", "max_over_steps_selected"],
    )
    parser.add_argument("--insufficiency_contact_negative_source", type=str, default="train_split_bank")
    parser.add_argument(
        "--insufficiency_contact_loss_mode",
        type=str,
        default="legacy",
        choices=["legacy", "threshold_contrastive"],
    )
    parser.add_argument("--insufficiency_contact_tolerance", type=float, default=0.25)
    parser.add_argument("--insufficiency_contact_supported_guard_weight", type=float, default=0.0)
    parser.add_argument("--insufficiency_contact_supported_margin_floor", type=float, default=0.25)
    parser.add_argument("--insufficiency_contact_positive_nll_weight", type=float, default=0.0)
    parser.add_argument("--insufficiency_contact_path_allowed_drift", type=float, default=0.0)
    parser.add_argument(
        "--insufficiency_contact_checkpoint_candidates",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Activation-checkpoint differentiable insufficiency-contact candidate scoring to reduce VRAM.",
    )
    parser.add_argument(
        "--insufficiency_contact_apply_to_insufficient",
        type=lambda x: str(x).lower() == "true",
        default=True,
    )
    parser.add_argument("--squad_train_ratio", type=float, default=0.0)
    parser.add_argument("--squad_train_max_samples", type=int, default=2000)
    parser.add_argument("--squad_train_max_length", type=int, default=384)
    parser.add_argument("--squad_train_answer_loss_tokens", type=int, default=16)
    parser.add_argument("--squad_train_split", type=str, default="train")
    parser.add_argument("--boolq_train_ratio", type=float, default=0.0)
    parser.add_argument("--boolq_train_max_samples", type=int, default=2000)
    parser.add_argument("--boolq_train_max_length", type=int, default=160)
    parser.add_argument("--boolq_train_answer_loss_tokens", type=int, default=4)
    parser.add_argument("--boolq_train_split", type=str, default="train")
    parser.add_argument("--mc_train_ratio", type=float, default=0.0)
    parser.add_argument("--mc_train_max_samples", type=int, default=2000)
    parser.add_argument("--mc_train_max_length", type=int, default=256)
    parser.add_argument("--mc_train_answer_loss_tokens", type=int, default=4)
    parser.add_argument("--mc_train_split", type=str, default="train")
    parser.add_argument("--mmlu_pro_train_ratio", type=float, default=0.0)
    parser.add_argument("--mmlu_pro_train_max_samples", type=int, default=2000)
    parser.add_argument("--mmlu_pro_train_max_length", type=int, default=256)
    parser.add_argument("--mmlu_pro_train_answer_loss_tokens", type=int, default=4)
    parser.add_argument("--mmlu_pro_train_split", type=str, default="train")
    parser.add_argument("--gpqa_diamond_train_ratio", type=float, default=0.0)
    parser.add_argument("--gpqa_diamond_train_max_samples", type=int, default=2000)
    parser.add_argument("--gpqa_diamond_train_max_length", type=int, default=256)
    parser.add_argument("--gpqa_diamond_train_answer_loss_tokens", type=int, default=4)
    parser.add_argument("--gpqa_diamond_train_split", type=str, default="train")
    parser.add_argument("--aime_train_ratio", type=float, default=0.0)
    parser.add_argument("--aime_train_max_samples", type=int, default=30)
    parser.add_argument("--aime_train_max_length", type=int, default=256)
    parser.add_argument("--aime_train_answer_loss_tokens", type=int, default=8)
    parser.add_argument("--aime_train_split", type=str, default="train")
    parser.add_argument("--hotpotqa_train_ratio", type=float, default=0.0)
    parser.add_argument("--hotpotqa_train_max_samples", type=int, default=2000)
    parser.add_argument("--hotpotqa_train_max_length", type=int, default=320)
    parser.add_argument("--hotpotqa_train_answer_loss_tokens", type=int, default=16)
    parser.add_argument("--hotpotqa_train_split", type=str, default="train")
    parser.add_argument("--truthfulqa_train_ratio", type=float, default=0.0)
    parser.add_argument("--truthfulqa_train_max_samples", type=int, default=2000)
    parser.add_argument("--truthfulqa_train_max_length", type=int, default=256)
    parser.add_argument("--truthfulqa_train_answer_loss_tokens", type=int, default=4)
    parser.add_argument("--truthfulqa_train_split", type=str, default="validation")
    parser.add_argument("--train_static_prefix", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--static_prefix_ckpt", type=str, default=None)
    parser.add_argument("--static_train_steps", type=int, default=800)
    parser.add_argument("--batch_size_static", type=int, default=None, help="(deprecated)")
    parser.add_argument("--static_lr", type=float, default=2e-4)
    parser.add_argument("--prefix_len", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--num_particles", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--baseline_path", type=str, default=None)
    parser.add_argument("--write_base_baseline", action="store_true")
    parser.add_argument("--inference_refine_policy", type=str, default="accept_reject_v1")
    parser.add_argument("--inference_refine_max_steps", type=int, default=None)
    parser.add_argument("--inference_refine_energy_delta_tol", type=float, default=1e-3)
    parser.add_argument("--inference_refine_patience", type=int, default=1)
    parser.add_argument("--sample_generations", type=int, default=0)

    args = parser.parse_args()
    cfg = TrainingConfig(**{k: v for k, v in vars(args).items() if k != "batch_size_static"})
    main(cfg)
