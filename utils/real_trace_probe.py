"""Compact REAL trace summaries for row-safe reasoning artifacts.

These helpers summarize refinement traces into short scalar/list payloads that
are safe to attach to reasoning rows. They are observation-only and do not
alter REAL commit selection or generation behavior.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


REAL_TRACE_PROBE_SCHEMA_VERSION = 1


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_float_list(values: List[Any]) -> List[Optional[float]]:
    return [_safe_float(value) for value in values]


def _trace_1d(trace: Any, *, name: str) -> Optional[torch.Tensor]:
    if trace is None:
        return None
    if not isinstance(trace, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor when provided")
    trace_f = trace.detach().float().cpu()
    if trace_f.ndim == 2 and trace_f.size(0) == 1:
        trace_f = trace_f[0]
    if trace_f.ndim != 1:
        raise ValueError(f"{name} must have shape [T], got {tuple(trace_f.shape)}")
    return trace_f


def _flatten_steps(traj: Any, *, name: str) -> Optional[torch.Tensor]:
    if traj is None:
        return None
    if not isinstance(traj, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor when provided")
    traj_f = traj.detach().float().cpu()
    if traj_f.ndim >= 2 and traj_f.size(0) == 1 and traj_f.ndim > 2:
        traj_f = traj_f[0]
    if traj_f.ndim < 2:
        raise ValueError(f"{name} must have shape [T, ...], got {tuple(traj_f.shape)}")
    return traj_f.reshape(int(traj_f.size(0)), -1)


def _delta_l2(flat_steps: Optional[torch.Tensor], steps: int) -> List[Optional[float]]:
    if flat_steps is None or steps <= 0:
        return []
    out: List[Optional[float]] = [None]
    if steps <= 1:
        return out
    deltas = flat_steps[1:] - flat_steps[:-1]
    out.extend(_safe_float_list(torch.linalg.vector_norm(deltas, dim=-1).tolist()))
    return out


def _turn_cosine(flat_steps: Optional[torch.Tensor], steps: int) -> List[Optional[float]]:
    if flat_steps is None or steps <= 0:
        return []
    out: List[Optional[float]] = [None]
    if steps == 1:
        return out
    out.append(None)
    if steps <= 2:
        return out
    deltas = flat_steps[1:] - flat_steps[:-1]
    for step_idx in range(2, steps):
        prev_delta = deltas[step_idx - 2]
        curr_delta = deltas[step_idx - 1]
        if float(torch.linalg.vector_norm(prev_delta).item()) == 0.0 or float(torch.linalg.vector_norm(curr_delta).item()) == 0.0:
            out.append(1.0)
            continue
        cosine = F.cosine_similarity(prev_delta.unsqueeze(0), curr_delta.unsqueeze(0), dim=-1)[0]
        out.append(_safe_float(cosine.item()))
    return out


def summarize_prefix_step_geometry(
    *,
    z_traj: Any = None,
    prefix_traj: Any = None,
) -> Dict[str, Any]:
    """Summarize geometric changes across refinement steps."""
    z_flat = _flatten_steps(z_traj, name="z_traj")
    prefix_flat = _flatten_steps(prefix_traj, name="prefix_traj")
    flat = prefix_flat if prefix_flat is not None else z_flat
    if flat is None:
        return {
            "real_trace_num_steps": 0,
            "real_trace_z_delta_l2_values": [],
            "real_trace_prefix_delta_l2_values": [],
            "real_trace_prefix_turn_cosine_values": [],
            "real_trace_max_prefix_turn_step_idx": None,
            "real_trace_max_prefix_turn_magnitude": None,
        }

    steps = int(flat.size(0))
    z_delta = _delta_l2(z_flat, steps)
    prefix_delta = _delta_l2(prefix_flat, steps)
    prefix_turn = _turn_cosine(flat, steps)

    max_turn_step = None
    max_turn_mag = None
    for step_idx, cosine in enumerate(prefix_turn):
        if cosine is None:
            continue
        magnitude = _safe_float(1.0 - float(cosine))
        if magnitude is None:
            continue
        if max_turn_mag is None or magnitude > max_turn_mag:
            max_turn_mag = magnitude
            max_turn_step = int(step_idx)

    return {
        "real_trace_num_steps": steps,
        "real_trace_z_delta_l2_values": z_delta,
        "real_trace_prefix_delta_l2_values": prefix_delta,
        "real_trace_prefix_turn_cosine_values": prefix_turn,
        "real_trace_max_prefix_turn_step_idx": max_turn_step,
        "real_trace_max_prefix_turn_magnitude": max_turn_mag,
    }


def choose_trace_candidate_steps(trace_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Choose compact candidate commit steps from a summarized trace."""
    if not bool(trace_summary.get("real_trace_available")):
        return {
            "final_step": None,
            "min_energy_step": None,
            "last_non_rebound_step": None,
            "max_prefix_turn_preceding_step": None,
            "lowest_prefix_delta_step": None,
            "candidate_step_indices": [],
        }

    final_step = trace_summary.get("real_trace_final_step_idx")
    min_energy_step = trace_summary.get("real_trace_min_energy_step_idx")
    rebound_step = trace_summary.get("real_trace_first_energy_rebound_step_idx")
    last_non_rebound_step = None
    if rebound_step is None:
        last_non_rebound_step = final_step
    elif int(rebound_step) > 0:
        last_non_rebound_step = int(rebound_step) - 1
    else:
        last_non_rebound_step = 0

    max_turn_step = trace_summary.get("real_trace_max_prefix_turn_step_idx")
    max_prefix_turn_preceding_step = None if max_turn_step is None else max(0, int(max_turn_step) - 1)

    prefix_delta = list(trace_summary.get("real_trace_prefix_delta_l2_values") or [])
    lowest_prefix_delta_step = None
    best_delta = None
    for step_idx, value in enumerate(prefix_delta):
        if step_idx == 0 or value is None:
            continue
        if best_delta is None or float(value) < best_delta:
            best_delta = float(value)
            lowest_prefix_delta_step = int(step_idx)

    candidate_indices: List[int] = []
    for value in (
        final_step,
        min_energy_step,
        last_non_rebound_step,
        max_prefix_turn_preceding_step,
        lowest_prefix_delta_step,
    ):
        if value is None:
            continue
        step_idx = int(value)
        if step_idx not in candidate_indices:
            candidate_indices.append(step_idx)

    return {
        "final_step": None if final_step is None else int(final_step),
        "min_energy_step": None if min_energy_step is None else int(min_energy_step),
        "last_non_rebound_step": None if last_non_rebound_step is None else int(last_non_rebound_step),
        "max_prefix_turn_preceding_step": (
            None if max_prefix_turn_preceding_step is None else int(max_prefix_turn_preceding_step)
        ),
        "lowest_prefix_delta_step": None if lowest_prefix_delta_step is None else int(lowest_prefix_delta_step),
        "candidate_step_indices": candidate_indices,
    }


def summarize_real_trace(
    *,
    energy_trace: Any,
    z_traj: Any = None,
    prefix_traj: Any = None,
    selected_step: Any = None,
    final_step: Any = None,
) -> Dict[str, Any]:
    """Summarize a REAL refinement trace into a compact JSON-safe payload."""
    energy = _trace_1d(energy_trace, name="energy_trace")
    if energy is None or int(energy.numel()) <= 0:
        return {
            "real_trace_probe_schema_version": REAL_TRACE_PROBE_SCHEMA_VERSION,
            "real_trace_available": False,
            "real_trace_num_steps": 0,
            "real_trace_selected_step_idx": None,
            "real_trace_final_step_idx": None,
            "real_trace_energy_values": [],
            "real_trace_energy_delta_values": [],
            "real_trace_min_energy_step_idx": None,
            "real_trace_final_minus_min_energy": None,
            "real_trace_first_energy_rebound_step_idx": None,
            "real_trace_has_energy_rebound": False,
            "real_trace_z_delta_l2_values": [],
            "real_trace_prefix_delta_l2_values": [],
            "real_trace_prefix_turn_cosine_values": [],
            "real_trace_max_prefix_turn_step_idx": None,
            "real_trace_max_prefix_turn_magnitude": None,
            "real_trace_candidate_steps": choose_trace_candidate_steps({"real_trace_available": False}),
        }

    steps = int(energy.numel())
    final_step_idx = int(final_step) if final_step is not None else steps - 1
    selected_step_idx = int(selected_step) if selected_step is not None else final_step_idx
    energy_values = _safe_float_list(energy.tolist())
    energy_delta_values: List[Optional[float]] = [None]
    if steps > 1:
        energy_delta_values.extend(_safe_float_list((energy[1:] - energy[:-1]).tolist()))

    min_energy_step_idx = int(energy.argmin().item())
    final_minus_min_energy = _safe_float(float(energy[final_step_idx].item() - energy[min_energy_step_idx].item()))

    first_rebound_step_idx = None
    for step_idx in range(1, steps):
        delta = _safe_float(energy[step_idx].item() - energy[step_idx - 1].item())
        if delta is not None and delta > 0.0:
            first_rebound_step_idx = int(step_idx)
            break

    out: Dict[str, Any] = {
        "real_trace_probe_schema_version": REAL_TRACE_PROBE_SCHEMA_VERSION,
        "real_trace_available": True,
        "real_trace_num_steps": steps,
        "real_trace_selected_step_idx": selected_step_idx,
        "real_trace_final_step_idx": final_step_idx,
        "real_trace_energy_values": energy_values,
        "real_trace_energy_delta_values": energy_delta_values,
        "real_trace_min_energy_step_idx": min_energy_step_idx,
        "real_trace_final_minus_min_energy": final_minus_min_energy,
        "real_trace_first_energy_rebound_step_idx": first_rebound_step_idx,
        "real_trace_has_energy_rebound": bool(first_rebound_step_idx is not None),
    }
    out.update(summarize_prefix_step_geometry(z_traj=z_traj, prefix_traj=prefix_traj))
    out["real_trace_candidate_steps"] = choose_trace_candidate_steps(out)
    return out


def run_self_check() -> Dict[str, Any]:
    """Run a tiny numeric self-check without model weights."""
    energy = torch.tensor([0.8, 0.6, 0.55, 0.62], dtype=torch.float32)
    z_traj = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    prefix_traj = z_traj.unsqueeze(1)
    summary = summarize_real_trace(energy_trace=energy, z_traj=z_traj, prefix_traj=prefix_traj)
    candidates = dict(summary.get("real_trace_candidate_steps") or {})
    if not bool(summary.get("real_trace_available")):
        raise AssertionError("self-check trace should be available")
    if int(summary.get("real_trace_num_steps") or 0) != 4:
        raise AssertionError("self-check step count mismatch")
    if int(summary.get("real_trace_min_energy_step_idx") or -1) != 2:
        raise AssertionError("self-check min-energy step mismatch")
    if not bool(summary.get("real_trace_has_energy_rebound")):
        raise AssertionError("self-check expected rebound")
    if int(candidates.get("final_step") or -1) != 3:
        raise AssertionError("self-check final-step candidate mismatch")
    return {
        "ok": True,
        "summary": summary,
    }


if __name__ == "__main__":
    print(json.dumps(run_self_check(), indent=2, sort_keys=True))
