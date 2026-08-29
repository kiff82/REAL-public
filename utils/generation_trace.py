"""Compact entry-step and generation-trace payload helpers.

These helpers are observation-only. They read existing `generate()` scores and
decoded token ids, but they do not alter generation behavior or add model
passes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from utils.pressure_probe import compute_pressure_trace


TRACE_VERSION = 1


def _decode_single_token_text(tokenizer, token_id: int) -> str:
    """Decode one token id without stripping special pieces."""
    try:
        return str(tokenizer.decode([int(token_id)], skip_special_tokens=False))
    except Exception:
        return ""


def _stack_step_scores(step_scores: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert generate() scores into a single `[T, V]` tensor."""
    if not step_scores:
        return None
    return torch.stack([score[0].detach().float().cpu() for score in step_scores], dim=0)


def _compact_topk_entries(
    tokenizer,
    log_probs: torch.Tensor,
    probs: torch.Tensor,
    *,
    top_k: int,
) -> list[Dict[str, Any]]:
    """Return a compact top-k list with decoded token text."""
    k = max(1, min(int(top_k), int(probs.numel())))
    top_vals, top_ids = torch.topk(probs, k=k, dim=-1)
    entries = []
    for idx, (prob, token_id) in enumerate(zip(top_vals, top_ids), start=1):
        token_id_int = int(token_id.item())
        entries.append(
            {
                "rank": int(idx),
                "token_id": token_id_int,
                "token_text": _decode_single_token_text(tokenizer, token_id_int),
                "logprob": float(log_probs[token_id_int].item()),
                "prob": float(prob.item()),
            }
        )
    return entries


def _trace_value(trace: Optional[Dict[str, torch.Tensor]], key: str, step_idx: int) -> Optional[float]:
    """Extract one scalar from a pressure-trace tensor dict."""
    if not isinstance(trace, dict):
        return None
    tensor = trace.get(key)
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.ndim != 1 or step_idx < 0 or step_idx >= int(tensor.numel()):
        return None
    return float(tensor[step_idx].detach().cpu().item())


def compute_entry_transition_probe_from_scores(
    tokenizer,
    step_scores: Sequence[torch.Tensor],
    generated_ids: torch.Tensor,
    *,
    prompt_token_count: int,
    top_k: int,
    pressure_trace: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Build the entry-step payload for the first generated token."""
    generated_steps = int(len(tuple(step_scores or ())))
    base_payload: Dict[str, Any] = {
        "trace_version": TRACE_VERSION,
        "generated_steps": generated_steps,
        "prompt_token_count": int(prompt_token_count),
    }
    if not step_scores:
        return {**base_payload, "missing_reason": "missing_step_scores"}
    if generated_ids.ndim == 2:
        generated_ids = generated_ids[0]
    if generated_ids.numel() <= 0:
        return {**base_payload, "missing_reason": "empty_generation"}

    stacked = _stack_step_scores(step_scores)
    trace = pressure_trace if pressure_trace is not None else compute_pressure_trace(stacked)

    entry_logits = step_scores[0][0].detach().float().cpu()
    log_probs = torch.log_softmax(entry_logits, dim=-1)
    probs = log_probs.exp()
    topk_entries = _compact_topk_entries(tokenizer, log_probs, probs, top_k=int(top_k))

    realized_token_id = int(generated_ids[0].item())
    realized_token_text = _decode_single_token_text(tokenizer, realized_token_id)
    realized_logprob = float(log_probs[realized_token_id].item())
    realized_prob = float(probs[realized_token_id].item())
    realized_rank = int((probs > probs[realized_token_id]).sum().item()) + 1
    first_top1 = topk_entries[0]
    first_top2_prob = float(topk_entries[1]["prob"]) if len(topk_entries) > 1 else 0.0
    first_step_gap = _trace_value(trace, "gap", 0)
    if first_step_gap is None:
        first_step_gap = float(first_top1["prob"] - first_top2_prob)

    payload = {
        **base_payload,
        "first_token_id": realized_token_id,
        "first_token_text": realized_token_text,
        "first_token_logprob": realized_logprob,
        "first_token_prob": realized_prob,
        "first_token_rank": int(realized_rank),
        "emitted_matches_top1": bool(realized_token_id == int(first_top1["token_id"])),
        "first_step_entropy": _trace_value(trace, "entropy", 0),
        "first_step_entropy_norm": _trace_value(trace, "entropy_norm", 0),
        "first_step_gap": first_step_gap,
        "first_step_pressure": _trace_value(trace, "pressure", 0),
        "first_step_pressure_norm": _trace_value(trace, "pressure_norm", 0),
        "first_step_top1_id": int(first_top1["token_id"]),
        "first_step_top1_text": str(first_top1["token_text"]),
        "first_step_top1_logprob": float(first_top1["logprob"]),
        "first_step_top1_prob": float(first_top1["prob"]),
        "topk": topk_entries,
    }

    # Backward-compatible aliases for the pre-existing repo surface.
    payload.update(
        {
            "entry_topk_tokens": [
                {
                    "rank": int(item["rank"]),
                    "token_id": int(item["token_id"]),
                    "text": str(item["token_text"]),
                    "prob": float(item["prob"]),
                }
                for item in topk_entries
            ],
            "entry_entropy": payload["first_step_entropy"],
            "entry_top1_prob": payload["first_step_top1_prob"],
            "entry_top2_prob": first_top2_prob,
            "entry_top_gap": payload["first_step_gap"],
            "entry_top1_token": {
                "token_id": int(first_top1["token_id"]),
                "text": str(first_top1["token_text"]),
            },
            "realized_first_token": {
                "token_id": realized_token_id,
                "text": realized_token_text,
            },
            "realized_first_token_rank_at_entry": int(realized_rank),
            "realized_first_token_prob_at_entry": realized_prob,
            "realized_first_token_top1_match": bool(payload["emitted_matches_top1"]),
        }
    )
    return payload


def compute_generation_trace_from_scores(
    tokenizer,
    step_scores: Sequence[torch.Tensor],
    generated_ids: torch.Tensor,
    *,
    prompt_token_count: int,
    top_k: int,
    pressure_trace: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Build a compact token-by-token generation trace."""
    generated_steps = int(len(tuple(step_scores or ())))
    base_payload: Dict[str, Any] = {
        "trace_version": TRACE_VERSION,
        "generated_steps": generated_steps,
        "prompt_token_count": int(prompt_token_count),
        "topk_size": int(top_k),
        "steps": [],
    }
    if not step_scores:
        return {**base_payload, "missing_reason": "missing_step_scores"}
    if generated_ids.ndim == 2:
        generated_ids = generated_ids[0]
    if generated_ids.numel() <= 0:
        return {**base_payload, "missing_reason": "empty_generation"}

    stacked = _stack_step_scores(step_scores)
    trace = pressure_trace if pressure_trace is not None else compute_pressure_trace(stacked)

    steps = []
    prev_top1_id: Optional[int] = None
    for step_idx, (score_t, token_id_t) in enumerate(zip(step_scores, generated_ids.tolist()), start=1):
        logits = score_t[0].detach().float().cpu()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        topk_entries = _compact_topk_entries(tokenizer, log_probs, probs, top_k=int(top_k))
        emitted_token_id = int(token_id_t)
        emitted_token_text = _decode_single_token_text(tokenizer, emitted_token_id)
        top1 = topk_entries[0]
        top1_id = int(top1["token_id"])
        steps.append(
            {
                "step": int(step_idx),
                "token_id": emitted_token_id,
                "token_text": emitted_token_text,
                "token_logprob": float(log_probs[emitted_token_id].item()),
                "token_prob": float(probs[emitted_token_id].item()),
                "emitted_matches_top1": bool(emitted_token_id == top1_id),
                "top1_id": top1_id,
                "top1_text": str(top1["token_text"]),
                "top1_logprob": float(top1["logprob"]),
                "top1_prob": float(top1["prob"]),
                "entropy": _trace_value(trace, "entropy", step_idx - 1),
                "entropy_norm": _trace_value(trace, "entropy_norm", step_idx - 1),
                "gap": _trace_value(trace, "gap", step_idx - 1),
                "pressure": _trace_value(trace, "pressure", step_idx - 1),
                "pressure_norm": _trace_value(trace, "pressure_norm", step_idx - 1),
                "top1_flip": bool(prev_top1_id is not None and top1_id != prev_top1_id),
                "topk": topk_entries,
            }
        )
        prev_top1_id = top1_id

    return {**base_payload, "steps": steps}
