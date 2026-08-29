"""Observation-only probe for residual direct prompt pull during generation.

This probe is eval-only and does not change generation, training, scoring, or
refine-policy behavior. It replays an already-generated continuation twice on the
frozen backbone:

1. full prompt
2. source-occluded prompt

For REAL, this is a measure of residual live prompt pull, not total grounding:
the REAL prefix may already encode some source information before replay.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from utils.replay_probe import (
    compute_masked_replay_probe,
    make_token_mask,
    token_span_from_char_span,
)


SOURCE_SPAN_MARKERS = {
    "squad2_reasoning": ("Context:\n", "\nQuestion:"),
    "fever_reasoning": ("Evidence:\n", "\nOptions:"),
    "hotpotqa_reasoning": ("Evidence:\n", "\nQuestion:"),
}

def _source_char_span_from_prompt(task_name: str, prompt_text: str) -> Optional[Tuple[int, int]]:
    """Locate the source/evidence substring inside a formatted reasoning prompt.

    Temporary contract: current reasoning tasks own source spans implicitly via
    prompt markers. Future tasks should pass source ownership via dataset
    metadata instead of re-parsing rendered prompt strings.
    """
    markers = SOURCE_SPAN_MARKERS.get(task_name)
    if not markers:
        return None
    start_marker, end_marker = markers
    start_idx = prompt_text.find(start_marker)
    if start_idx < 0:
        return None
    start_idx += len(start_marker)
    end_idx = prompt_text.find(end_marker, start_idx)
    if end_idx < 0 or end_idx <= start_idx:
        return None
    return start_idx, end_idx


def _source_char_span_from_meta(prompt_text: str, meta: Optional[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Prefer dataset-owned source spans/text when available."""
    if not isinstance(meta, dict):
        return None

    source_text = str(meta.get("context_pull_source_text") or "")
    if source_text:
        exact_idx = prompt_text.find(source_text)
        if exact_idx >= 0:
            return exact_idx, exact_idx + len(source_text)

        # Head+tail prompt truncation may keep only a prefix and suffix of the
        # original evidence span. When both anchors survive in the rendered prompt,
        # recover the surviving span between them.
        start_snippet = source_text[: min(128, len(source_text))].strip()
        end_snippet = source_text[-min(128, len(source_text)) :].strip()
        if start_snippet and end_snippet:
            start_idx = prompt_text.find(start_snippet)
            end_idx = prompt_text.rfind(end_snippet)
            if start_idx >= 0 and end_idx >= start_idx:
                end_idx += len(end_snippet)
                if end_idx > start_idx:
                    return start_idx, end_idx

    char_span = meta.get("context_pull_source_char_span")
    if isinstance(char_span, (list, tuple)) and len(char_span) == 2:
        try:
            start_idx = int(char_span[0])
            end_idx = int(char_span[1])
        except (TypeError, ValueError):
            start_idx = -1
            end_idx = -1
        if 0 <= start_idx < end_idx <= len(prompt_text):
            return start_idx, end_idx
    return None

def _resolve_source_token_mask(
    *,
    task_name: str,
    tokenizer,
    prompt_text: str,
    prompt_input_ids: torch.Tensor,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]], Optional[str]]:
    """Resolve the prompt token span that owns the source evidence/context."""
    char_span = _source_char_span_from_meta(prompt_text, meta)
    if char_span is None:
        char_span = _source_char_span_from_prompt(task_name, prompt_text)
    if char_span is None:
        return None, None, "no_source_span"

    tok_span = token_span_from_char_span(tokenizer, prompt_text, char_span, prompt_input_ids)
    if tok_span is None:
        return None, None, "span_to_tokens_failed"

    start_tok, end_tok = tok_span
    mask = make_token_mask(int(prompt_input_ids.size(1)), [(start_tok, end_tok)])
    if int(mask.sum().item()) <= 0:
        return None, None, "span_to_tokens_failed"
    return mask, tok_span, None


@torch.no_grad()
def compute_context_pull_probe(
    *,
    task_name: str,
    tokenizer,
    backbone,
    prompt_text: str,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
    prefix_final: Optional[torch.Tensor],
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, object]]:
    """Replay a generated trajectory with and without direct source access."""
    if generated_ids.ndim == 1:
        generated_ids = generated_ids.unsqueeze(0)
    if generated_ids.numel() == 0:
        return {"missing_reason": "empty_generation"}

    source_mask, source_span, missing_reason = _resolve_source_token_mask(
        task_name=task_name,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        prompt_input_ids=prompt_input_ids,
        meta=meta,
    )
    if source_mask is None or source_span is None:
        return {"missing_reason": str(missing_reason or "no_source_span")}

    prompt_len = int(prompt_input_ids.size(1))
    gen_len = int(generated_ids.size(1))
    source_kind = str((meta or {}).get("context_pull_source_kind") or {
        "squad2_reasoning": "context",
        "fever_reasoning": "evidence",
    }.get(task_name, "source"))

    prompt_ids = prompt_input_ids.detach().cpu()
    prompt_attn = prompt_attention_mask.detach().cpu()
    gen_ids = generated_ids.detach().cpu().long()

    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    gen_attn = torch.ones_like(gen_ids, dtype=prompt_attn.dtype)
    full_attn = torch.cat([prompt_attn, gen_attn], dim=1)
    occlude_mask = torch.cat([source_mask.unsqueeze(0), torch.zeros_like(gen_ids, dtype=torch.bool)], dim=1)

    probe = compute_masked_replay_probe(
        backbone=backbone,
        input_ids=full_ids,
        attention_mask=full_attn,
        prefix_final=prefix_final,
        score_token_span=(prompt_len, prompt_len + gen_len),
        occlude_token_mask=occlude_mask,
        trace_prefix="context_pull",
        target_token_ids=gen_ids,
    )
    if "summary" not in probe:
        return {"missing_reason": str(probe.get("missing_reason") or "replay_failed")}

    start_tok, end_tok = source_span
    summary = dict(probe["summary"])
    summary.setdefault("context_pull_source_tokens_masked", float(int(source_mask.sum().item())))

    return {
        "summary": summary,
        "source_kind": source_kind,
        "source_token_span": [int(start_tok), int(end_tok)],
        "source_tokens_masked": int(source_mask.sum().item()),
        "kl_full_vs_occluded": list(probe.get("kl") or []),
        "js_full_vs_occluded": list(probe.get("js") or []),
        "token_logprob_drop": list(probe.get("token_logprob_drop") or []),
        "gap_drop": list(probe.get("gap_drop") or []),
        "top1_flip": list(probe.get("top1_flip") or []),
    }
