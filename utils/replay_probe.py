"""Shared masked-replay utilities for observation-only reasoning probes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from real.core.prefix_ops import build_inputs_embeds_with_prefix


def _tokenizer_input_ids(encoded: Any) -> List[int]:
    if isinstance(encoded, dict):
        return list(encoded.get("input_ids") or [])
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        return []
    return list(input_ids)


def trace_summary_fields(trace_values: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Compute short-trace summary features for one 1D metric."""
    if trace_values.ndim != 1:
        raise ValueError(
            "trace_summary_fields expects a single-example trace with shape [T]; "
            f"got {tuple(trace_values.shape)}"
        )

    def _f(x: torch.Tensor) -> float:
        return float(x.detach().cpu().item())

    peak_idx = int(trace_values.argmax().detach().cpu().item())
    first = trace_values[0]
    final = trace_values[-1]
    peak = trace_values[peak_idx]
    delta_1_2 = trace_values[1] - trace_values[0] if trace_values.numel() > 1 else torch.zeros_like(first)
    return {
        f"{prefix}_first": _f(first),
        f"{prefix}_delta_1_2": _f(delta_1_2),
        f"{prefix}_mean": _f(trace_values.mean()),
        f"{prefix}_peak": _f(peak),
        f"{prefix}_peak_step_idx": float(peak_idx + 1),
        f"{prefix}_peak_minus_final": _f(peak - final),
        f"{prefix}_final": _f(final),
        f"{prefix}_auc": _f(trace_values.sum()),
    }


def stable_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build explicit position ids so replay geometry stays fixed across replays."""
    pos = attention_mask.long().cumsum(dim=1) - 1
    return pos.clamp_min(0)


def token_span_from_char_span(
    tokenizer,
    text: str,
    char_span: Tuple[int, int],
    input_ids: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    """Map a character span in rendered text back to token indices."""
    char_start, char_end = int(char_span[0]), int(char_span[1])
    if char_end <= char_start:
        return None

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    no_special_ids = _tokenizer_input_ids(encoded)

    flat_ids = input_ids[0].detach().cpu().tolist() if input_ids.ndim == 2 else input_ids.detach().cpu().tolist()
    bos_shift = 0
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and len(flat_ids) == len(no_special_ids) + 1 and flat_ids[0] == bos_id:
        bos_shift = 1

    if offsets:
        selected: List[int] = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:
                continue
            if tok_end <= char_start or tok_start >= char_end:
                continue
            selected.append(idx)
        if selected:
            start_tok = selected[0] + bos_shift
            end_tok = min(selected[-1] + 1 + bos_shift, len(flat_ids))
            if end_tok > start_tok:
                return start_tok, end_tok

    prefix_ids = _tokenizer_input_ids(tokenizer(text[:char_start], add_special_tokens=False))
    through_ids = _tokenizer_input_ids(tokenizer(text[:char_end], add_special_tokens=False))
    start_tok = len(prefix_ids) + bos_shift
    end_tok = min(len(through_ids) + bos_shift, len(flat_ids))
    if end_tok <= start_tok:
        return None
    return start_tok, end_tok


def make_token_mask(length: int, spans: Iterable[Tuple[int, int]]) -> torch.Tensor:
    """Create a flat boolean token mask from one or more [start, end) spans."""
    mask = torch.zeros(int(length), dtype=torch.bool)
    for start, end in spans:
        start_i = max(0, int(start))
        end_i = min(int(length), int(end))
        if end_i > start_i:
            mask[start_i:end_i] = True
    return mask


def masked_replay_logits(
    backbone,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    prefix_final: Optional[torch.Tensor],
    occlude_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one replay pass and return logits with stable token positions."""
    emb_weight = backbone.get_input_embeddings().weight
    emb_dev = emb_weight.device
    model_dtype = getattr(backbone, "dtype", emb_weight.dtype)

    ids = input_ids.to(emb_dev)
    attn = attention_mask.to(emb_dev)
    emb = backbone.get_input_embeddings()(ids).to(model_dtype)

    if occlude_token_mask is not None:
        occlude = occlude_token_mask.to(device=emb.device, dtype=torch.bool)
        if occlude.ndim == 1:
            occlude = occlude.unsqueeze(0)
        if occlude.shape != ids.shape:
            raise ValueError(
                f"occlude_token_mask shape {tuple(occlude.shape)} does not match input_ids {tuple(ids.shape)}"
            )
        # Embedding-level occlusion preserves sequence length, positions, and
        # causal indexing; the only intended intervention is content removal.
        emb = emb.masked_fill(occlude.unsqueeze(-1), 0.0)

    if prefix_final is None:
        inputs_embeds = emb
        attn2 = attn
    else:
        prefix = prefix_final.to(device=emb.device, dtype=model_dtype)
        if prefix.size(0) != emb.size(0):
            prefix = prefix.expand(emb.size(0), -1, -1)
        inputs_embeds, attn2, _labels_unused = build_inputs_embeds_with_prefix(
            emb=emb,
            attention_mask=attn,
            labels=None,
            prefix=prefix,
        )

    position_ids = stable_position_ids(attn2)
    try:
        out = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn2,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
    except TypeError:
        out = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn2,
            use_cache=False,
            return_dict=True,
        )
    return out.logits.detach().float()


def js_divergence(logp_a: torch.Tensor, logp_b: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two log-prob tensors."""
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    m = 0.5 * (p_a + p_b)
    log_m = torch.log(m.clamp_min(1e-12))
    kl_a = (p_a * (logp_a - log_m)).sum(dim=-1)
    kl_b = (p_b * (logp_b - log_m)).sum(dim=-1)
    return 0.5 * (kl_a + kl_b)


def top_gap(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 probability gap for one or more steps."""
    probs = F.softmax(logits, dim=-1)
    top_vals = torch.topk(probs, k=min(2, int(probs.size(-1))), dim=-1).values
    if top_vals.size(-1) == 1:
        return top_vals[..., 0]
    return top_vals[..., 0] - top_vals[..., 1]


def summarize_replay_difference(
    *,
    trace_prefix: str,
    score_steps: int,
    tokens_masked: int,
    kl: torch.Tensor,
    js: torch.Tensor,
    token_logprob_drop: torch.Tensor,
    gap_drop: torch.Tensor,
    top1_flip: torch.Tensor,
    intact_ce: Optional[torch.Tensor] = None,
    masked_ce: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Build compact summary features for one scored replay region."""
    summary: Dict[str, float] = {
        f"{trace_prefix}_steps": float(score_steps),
        f"{trace_prefix}_tokens_masked": float(tokens_masked),
        f"{trace_prefix}_top1_flip_rate": float(top1_flip.mean().item()),
    }
    summary.update(trace_summary_fields(kl, prefix=f"{trace_prefix}_kl"))
    summary.update(trace_summary_fields(js, prefix=f"{trace_prefix}_js"))
    summary.update(trace_summary_fields(token_logprob_drop, prefix=f"{trace_prefix}_token_logprob_drop"))
    summary.update(trace_summary_fields(gap_drop, prefix=f"{trace_prefix}_gap_drop"))
    if intact_ce is not None:
        summary[f"{trace_prefix}_ce_intact"] = float(intact_ce.detach().cpu().item())
    if masked_ce is not None:
        summary[f"{trace_prefix}_ce_masked"] = float(masked_ce.detach().cpu().item())
    if intact_ce is not None and masked_ce is not None:
        summary[f"{trace_prefix}_ce_delta"] = float((masked_ce - intact_ce).detach().cpu().item())
    return summary


@torch.no_grad()
def compute_masked_replay_probe(
    *,
    backbone,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_final: Optional[torch.Tensor],
    score_token_span: Tuple[int, int],
    occlude_token_mask: Optional[torch.Tensor],
    trace_prefix: str,
    target_token_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Replay one sequence intact vs masked and score a target token subspan."""
    score_start, score_end = int(score_token_span[0]), int(score_token_span[1])
    seq_len = int(input_ids.size(1))
    if score_start <= 0 or score_end <= score_start or score_end > seq_len:
        return {"missing_reason": "invalid_score_span"}

    logits_full = masked_replay_logits(
        backbone,
        input_ids,
        attention_mask,
        prefix_final=prefix_final,
        occlude_token_mask=None,
    )
    logits_masked = masked_replay_logits(
        backbone,
        input_ids,
        attention_mask,
        prefix_final=prefix_final,
        occlude_token_mask=occlude_token_mask,
    )

    prefix_shift = int(prefix_final.size(1)) if prefix_final is not None else 0
    start_pos = prefix_shift + score_start - 1
    end_pos = prefix_shift + score_end - 1
    step_logits_full = logits_full[:, start_pos:end_pos, :]
    step_logits_masked = logits_masked[:, start_pos:end_pos, :]
    score_steps = score_end - score_start
    if int(step_logits_full.size(1)) != score_steps or int(step_logits_masked.size(1)) != score_steps:
        return {"missing_reason": "replay_shape_mismatch"}

    if target_token_ids is None:
        target_token_ids = input_ids[:, score_start:score_end]
    if target_token_ids.ndim == 1:
        target_token_ids = target_token_ids.unsqueeze(0)
    target_token_ids = target_token_ids.to(step_logits_full.device).long()
    if int(target_token_ids.size(1)) != score_steps:
        return {"missing_reason": "target_shape_mismatch"}

    logp_full = F.log_softmax(step_logits_full, dim=-1)
    logp_masked = F.log_softmax(step_logits_masked, dim=-1)
    p_full = logp_full.exp()

    kl_full_vs_masked = (p_full * (logp_full - logp_masked)).sum(dim=-1).squeeze(0)
    js_full_vs_masked = js_divergence(logp_full, logp_masked).squeeze(0)
    token_logprob_full = logp_full.gather(-1, target_token_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
    token_logprob_masked = logp_masked.gather(-1, target_token_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
    token_logprob_drop = token_logprob_full - token_logprob_masked
    gap_drop = (top_gap(step_logits_full) - top_gap(step_logits_masked)).squeeze(0)
    top1_flip = (step_logits_full.argmax(dim=-1) != step_logits_masked.argmax(dim=-1)).float().squeeze(0)

    intact_ce = (-token_logprob_full).mean()
    masked_ce = (-token_logprob_masked).mean()
    tokens_masked = int(occlude_token_mask.sum().item()) if occlude_token_mask is not None else 0

    return {
        "summary": summarize_replay_difference(
            trace_prefix=trace_prefix,
            score_steps=score_steps,
            tokens_masked=tokens_masked,
            kl=kl_full_vs_masked,
            js=js_full_vs_masked,
            token_logprob_drop=token_logprob_drop,
            gap_drop=gap_drop,
            top1_flip=top1_flip,
            intact_ce=intact_ce,
            masked_ce=masked_ce,
        ),
        "score_token_span": [score_start, score_end],
        "tokens_masked": tokens_masked,
        "kl": [float(x) for x in kl_full_vs_masked.detach().cpu().tolist()],
        "js": [float(x) for x in js_full_vs_masked.detach().cpu().tolist()],
        "token_logprob_drop": [float(x) for x in token_logprob_drop.detach().cpu().tolist()],
        "gap_drop": [float(x) for x in gap_drop.detach().cpu().tolist()],
        "top1_flip": [float(x) for x in top1_flip.detach().cpu().tolist()],
    }
