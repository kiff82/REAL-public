#!/usr/bin/env python3
"""Network-free illustration of local dynamic contact in a frozen geometry.

This is not an LLM experiment and is not evidence for REAL's empirical claims.
It isolates one structural fact: a single global perturbation cannot generally
repair several opposing local basin errors, while an input-conditioned signal
can learn different request-local corrections without modifying the frozen
decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemoResult:
    """Compact result from the illustrative contact geometry."""

    seed: int
    train_rows: int
    test_rows: int
    updates: int
    states: list[dict[str, Any]]
    base_accuracy: float
    static_accuracy: float
    real_accuracy: float
    frozen_decoder_hash_before: str
    frozen_decoder_hash_after: str
    frozen_decoder_unchanged: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_array(value: np.ndarray) -> str:
    payload = np.ascontiguousarray(value).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()


def _balanced_latent_rows(
    *, rng: np.random.Generator, rows_per_class: int, mixing: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_count = int(mixing.shape[0])
    labels = np.repeat(np.arange(class_count), rows_per_class)
    rng.shuffle(labels)
    targets = np.eye(class_count, dtype=np.float64)[labels]
    noise = rng.normal(loc=0.0, scale=0.025, size=targets.shape)
    prompt_states = targets @ mixing.T + noise
    return prompt_states, targets, labels


def _decode(frozen_decoder: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Decode with one immutable nearest-basin score surface."""
    scores = states @ frozen_decoder.T
    return scores.argmax(axis=1)


def _accuracy(frozen_decoder: np.ndarray, states: np.ndarray, labels: np.ndarray) -> float:
    return float((_decode(frozen_decoder, states) == labels).mean())


def _fit_dynamic_contact_head(
    prompt_states: np.ndarray, targets: np.ndarray, *, ridge: float = 1e-4
) -> np.ndarray:
    """Fit a tiny linear head from prompt state to request-local correction."""
    desired_correction = targets - prompt_states
    gram = prompt_states.T @ prompt_states
    return np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0], dtype=np.float64),
        prompt_states.T @ desired_correction,
    )


def run_demo(
    *,
    seed: int = 7,
    train_rows_per_class: int = 64,
    test_rows_per_class: int = 32,
    updates: int = 4,
    update_rate: float = 0.25,
) -> DemoResult:
    """Run the deterministic frozen-geometry demonstration."""
    if updates != 4:
        raise ValueError("The public structural demo fixes four updates / five states z0..z4")
    if not 0.0 < update_rate <= 1.0:
        raise ValueError("update_rate must be in (0, 1]")

    rng = np.random.default_rng(seed)

    # Each target basin is initially pulled toward the next basin. Balanced
    # opposing errors make one global offset a poor repair surface.
    mixing = np.array(
        [
            [0.20, 0.80, 0.00],
            [0.00, 0.20, 0.80],
            [0.80, 0.00, 0.20],
        ],
        dtype=np.float64,
    )
    frozen_decoder = np.eye(3, dtype=np.float64)
    decoder_hash_before = _sha256_array(frozen_decoder)

    h_train, target_train, _labels_train = _balanced_latent_rows(
        rng=rng, rows_per_class=train_rows_per_class, mixing=mixing
    )
    h_test, target_test, labels_test = _balanced_latent_rows(
        rng=rng, rows_per_class=test_rows_per_class, mixing=mixing
    )

    base_accuracy = _accuracy(frozen_decoder, h_test, labels_test)

    # Least-squares fixed field. Because the three local errors oppose one
    # another, their balanced global average is nearly zero.
    static_field = (target_train - h_train).mean(axis=0, keepdims=True)
    static_accuracy = _accuracy(frozen_decoder, h_test + static_field, labels_test)

    contact_head = _fit_dynamic_contact_head(h_train, target_train)
    target_correction = h_test @ contact_head

    # z0 is the zero contact signal. Four request-local updates produce z1..z4;
    # the frozen decoder commits from h + z4.
    z = np.zeros_like(h_test)
    state_trace: list[dict[str, Any]] = [
        {
            "state": "z0",
            "accuracy": _accuracy(frozen_decoder, h_test + z, labels_test),
            "mean_contact_norm": float(np.linalg.norm(z, axis=1).mean()),
        }
    ]
    for step in range(1, updates + 1):
        z = z + update_rate * (target_correction - z)
        state_trace.append(
            {
                "state": f"z{step}",
                "accuracy": _accuracy(frozen_decoder, h_test + z, labels_test),
                "mean_contact_norm": float(np.linalg.norm(z, axis=1).mean()),
            }
        )

    real_accuracy = _accuracy(frozen_decoder, h_test + z, labels_test)
    decoder_hash_after = _sha256_array(frozen_decoder)

    return DemoResult(
        seed=seed,
        train_rows=int(h_train.shape[0]),
        test_rows=int(h_test.shape[0]),
        updates=updates,
        states=state_trace,
        base_accuracy=base_accuracy,
        static_accuracy=static_accuracy,
        real_accuracy=real_accuracy,
        frozen_decoder_hash_before=decoder_hash_before,
        frozen_decoder_hash_after=decoder_hash_after,
        frozen_decoder_unchanged=decoder_hash_before == decoder_hash_after,
        interpretation=(
            "Illustrative only: the input-conditioned correction resolves opposing local basin errors "
            "that one balanced global field does not; this is not language-model evidence."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    args = parser.parse_args()

    result = run_demo(seed=args.seed)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return

    print("REAL structural contact demo — illustrative, not empirical LLM evidence")
    print(f"base accuracy:   {result.base_accuracy:.3f}")
    print(f"static accuracy: {result.static_accuracy:.3f}")
    print(f"REAL accuracy:   {result.real_accuracy:.3f}")
    print("state trace:")
    for state in result.states:
        print(
            f"  {state['state']}: accuracy={state['accuracy']:.3f}, "
            f"mean_contact_norm={state['mean_contact_norm']:.3f}"
        )
    print(f"frozen decoder unchanged: {result.frozen_decoder_unchanged}")


if __name__ == "__main__":
    main()
