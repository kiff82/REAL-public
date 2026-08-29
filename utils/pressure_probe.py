"""Observation-only pressure probe for LM token logits.

This module intentionally provides a readout, not a runtime policy. It does not
change generation, gate tokens, resample outputs, or introduce any training
signal. The goal is to surface simple per-step diagnostics for uncertainty and
trajectory stability during evaluation.

The raw composite pressure stays backward-compatible:

    pressure = entropy + (1 - gap) + drift

where `drift` is the historical mean absolute probability change. For newer
cross-run analysis we also export normalized companions:

- `entropy_norm`: entropy divided by `log(vocab_size)` in `[0, 1]`
- `drift_tv`: total-variation drift in `[0, 1]`
- `pressure_norm`: `entropy_norm + (1 - gap) + drift_tv`
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace non-finite generation scores with large finite sentinels."""
    return torch.nan_to_num(logits.detach().float(), nan=0.0, neginf=-1e9, posinf=1e9)


def _normalized_entropy(entropy: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Scale entropy to roughly `[0, 1]` using the uniform-distribution bound."""
    if vocab_size <= 1:
        return torch.zeros_like(entropy)
    return entropy / float(math.log(vocab_size))


@torch.no_grad()
def compute_pressure(logits: torch.Tensor, prev_logits: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Compute entropy, top-2 gap, drift, and composite pressure for one step."""
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dimension, got {tuple(logits.shape)}")

    logits_f = _sanitize_logits(logits)
    log_probs = F.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)

    topk = min(2, int(logits_f.size(-1)))
    top_vals = torch.topk(probs, k=topk, dim=-1).values
    if topk == 1:
        gap = top_vals[..., 0]
    else:
        gap = top_vals[..., 0] - top_vals[..., 1]

    if prev_logits is None:
        drift = torch.zeros_like(entropy)
        drift_tv = torch.zeros_like(entropy)
    else:
        prev_f = _sanitize_logits(prev_logits)
        if prev_f.shape != logits_f.shape:
            raise ValueError(
                f"prev_logits shape {tuple(prev_f.shape)} does not match logits shape {tuple(logits_f.shape)}"
            )
        prev_probs = F.softmax(prev_f, dim=-1)
        delta_probs = (probs - prev_probs).abs()
        drift = delta_probs.mean(dim=-1)
        drift_tv = 0.5 * delta_probs.sum(dim=-1)

    entropy_norm = _normalized_entropy(entropy, int(logits_f.size(-1)))
    pressure = entropy + (1.0 - gap) + drift
    pressure_norm = entropy_norm + (1.0 - gap) + drift_tv
    return {
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "gap": gap,
        "drift": drift,
        "drift_tv": drift_tv,
        "pressure": pressure,
        "pressure_norm": pressure_norm,
    }


@torch.no_grad()
def compute_pressure_trace(step_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Vectorized pressure probe across a generation trajectory."""
    if step_logits.ndim not in {2, 3}:
        raise ValueError(
            f"step_logits must have shape [T, V] or [B, T, V], got {tuple(step_logits.shape)}"
        )

    logits_f = _sanitize_logits(step_logits)
    if logits_f.size(-2) == 0:
        shape = logits_f.shape[:-1]
        zeros = torch.zeros(shape, dtype=logits_f.dtype, device=logits_f.device)
        return {
            "entropy": zeros,
            "entropy_norm": zeros,
            "gap": zeros,
            "drift": zeros,
            "drift_tv": zeros,
            "pressure": zeros,
            "pressure_norm": zeros,
        }

    log_probs = F.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_norm = _normalized_entropy(entropy, int(logits_f.size(-1)))

    topk = min(2, int(logits_f.size(-1)))
    top_vals = torch.topk(probs, k=topk, dim=-1).values
    if topk == 1:
        gap = top_vals[..., 0]
    else:
        gap = top_vals[..., 0] - top_vals[..., 1]

    drift = torch.zeros_like(entropy)
    drift_tv = torch.zeros_like(entropy)
    if logits_f.size(-2) > 1:
        delta_probs = (probs[..., 1:, :] - probs[..., :-1, :]).abs()
        drift[..., 1:] = delta_probs.mean(dim=-1)
        drift_tv[..., 1:] = 0.5 * delta_probs.sum(dim=-1)

    pressure = entropy + (1.0 - gap) + drift
    pressure_norm = entropy_norm + (1.0 - gap) + drift_tv
    return {
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "gap": gap,
        "drift": drift,
        "drift_tv": drift_tv,
        "pressure": pressure,
        "pressure_norm": pressure_norm,
    }


def _trace_summary_fields(trace_values: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Compute short-trace summary features for one 1D metric."""
    if trace_values.ndim != 1:
        raise ValueError(
            "_trace_summary_fields expects a single-example trace with shape [T]; "
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


@torch.no_grad()
def summarize_pressure_trace(trace: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Reduce a per-step pressure trace into a compact JSON-friendly summary."""
    required = ("entropy", "gap", "drift", "pressure")
    missing = [k for k in required if k not in trace]
    if missing:
        raise KeyError(f"Missing pressure trace keys: {missing}")

    pressure = trace["pressure"]
    if pressure.ndim != 1:
        raise ValueError(
            "summarize_pressure_trace expects a single-example trace with shape [T]; "
            f"got {tuple(pressure.shape)}"
        )

    def _f(x: torch.Tensor) -> float:
        return float(x.detach().cpu().item())

    summary = {
        "trace_steps": int(pressure.numel()),
        "steps": int(pressure.numel()),
        "entropy_mean": _f(trace["entropy"].mean()),
        "gap_mean": _f(trace["gap"].mean()),
        "drift_mean": _f(trace["drift"].mean()),
    }
    if "entropy_norm" in trace:
        summary["entropy_norm_mean"] = _f(trace["entropy_norm"].mean())
    if "drift_tv" in trace:
        summary["drift_tv_mean"] = _f(trace["drift_tv"].mean())
    summary.update(_trace_summary_fields(pressure, prefix="pressure"))
    pressure_norm = trace.get("pressure_norm")
    if isinstance(pressure_norm, torch.Tensor):
        summary.update(_trace_summary_fields(pressure_norm, prefix="pressure_norm"))
    return summary
