"""Eval protocol sidecars for basin-facing and reasoning-facing artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EvalProtocol:
    """Metadata describing an evaluation signature and benchmark role (no behavior)."""

    # Dataset / loader
    dataset_name: str
    dataset_config_name: Optional[str]
    split: str
    eval_batches: int
    batch_size: int
    max_length: int

    # Refinement policy
    refine_policy: str
    refine_max_steps: Optional[int] = None
    refine_energy_delta_tol: Optional[float] = None
    refine_patience: Optional[int] = None

    # Prefix / dtype regime
    prefix_norm_clamp: float = 0.0
    prefix_norm_clamp_applied: bool = False
    dtype_mode: str = "fp16"  # fp16/bf16/auto
    dtype_used: Optional[str] = None  # fp16/bf16 when dtype_mode="auto"
    quantization: Optional[str] = None

    # Supervision windowing
    answer_loss_tokens: Optional[int] = None

    # Benchmark role / abstention semantics
    benchmark_role: str = "basin_probe"
    reasoning_eval_mode: bool = False
    allow_abstain: bool = False
    abstain_mode: str = "none"
    abstain_text: Optional[str] = None
    abstain_option_letter: Optional[str] = None
    primary_metric_family: Optional[str] = None
    primary_metric_name: Optional[str] = None

    # Regime / manifold-contact metadata (no behavior)
    evaluation_regime: Optional[str] = None
    dynamic_signal_active: Optional[bool] = None
    frozen_manifold_modified: bool = False
    regime_role: Optional[str] = None
    manifold_contact_evidence_level: Optional[str] = None

    # Reproducibility
    seed: Optional[int] = None
    deterministic: Optional[bool] = None

    # Optional exact-sample filtering
    sample_ids_path: Optional[str] = None
    sample_id_filter_requested_count: Optional[int] = None
    sample_id_filter_matched_count: Optional[int] = None
    sample_id_filter_missing_count: Optional[int] = None
    sample_id_filter_selected_row_count: Optional[int] = None

    # Optional observation-only REAL trace probe
    real_trace_probe: bool = False
    real_trace_probe_schema_version: Optional[int] = None
    real_trace_probe_dump_tensors: bool = False
    real_trace_probe_behavior_change: bool = False

    # Schema
    protocol_version: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EvalProtocol":
        if not isinstance(d, dict):
            raise TypeError("EvalProtocol.from_dict expects a dict")
        allowed = {f.name for f in fields(EvalProtocol)}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return EvalProtocol(**filtered)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def read_json(path: Path) -> "EvalProtocol":
        return EvalProtocol.from_dict(json.loads(path.read_text(encoding="utf-8")))
