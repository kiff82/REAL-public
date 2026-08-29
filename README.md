# REAL — Recursive External Alignment Layer

REAL is an experimental **contact system** formed by a frozen language-model
manifold and a small input-conditioned dynamic head.

> **Contact makes them one system.**

The frozen backbone supplies the latent structure. The head does not replace,
repair, or update that backbone. It emits a request-local signal intended to
help the coupled system enter, preserve, or restore useful latent contact before
generation collapses into tokens.

This repository is a bounded public research seed. It exposes the central
structure, runnable research code, a network-free structural demonstration, and
compact aggregate evidence. It intentionally does not expose the private
development history, raw benchmark rows, checkpoints, model shards, sealed
evidence banks, or the graduated Heartbeat support substrate.

## Status at a glance

**Research stage:** pre-alpha / falsifiable prototype
**Public seed:** `0.1.0`
**Release boundary:** code and aggregate evidence; no weights or datasets

The current evidence supports a narrow research question:

> Can an input-conditioned signal alter a frozen model's state access and
> trajectory more usefully than either no signal or one fixed global signal?

It does **not** establish a general alignment solution, autonomous agency,
model self-knowledge, reliable abstention, or a model-facing authority layer.
See [RELEASE_STATUS.md](RELEASE_STATUS.md) before interpreting results.

## The three regimes

| Regime | Definition | Role |
| --- | --- | --- |
| `base` | frozen manifold without an added signal | baseline |
| `static` | frozen manifold with one fixed/global learned prefix | falsification reference |
| `REAL` | frozen manifold with an input-conditioned dynamic head | system under test |

StaticPrefix is not a straw baseline. If a fixed signal explains a gain, the
dynamic-contact claim has not been established.

## Start with the structural demo

The demo uses a tiny frozen NumPy geometry. It shows why one global
perturbation can fail across opposing local basins while an input-conditioned
signal can establish different local contacts. It is an illustration, not
empirical evidence about language models.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python examples/structural_contact_demo.py
python -m unittest discover -s tests -v
python scripts/release_audit.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Research code

The model-facing skeleton is kept close to the development implementation:

- `train_real_v1_3.py` — head/static training and evaluation entry point
- `real/core/` — prefix, loss, dtype, path, and protocol primitives
- `scripts/infer_real.py` — single-example base/static/REAL harness
- `scripts/reasoning_eval.py` — uncertainty-aware reasoning evaluation
- `utils/` — the minimal probe and contract dependency closure

Install the research dependencies:

```bash
python -m pip install -e '.[research]'
```

GPU quantized inference also requires the optional GPU dependency group:

```bash
python -m pip install -e '.[research,gpu]'
```

Then inspect the available interfaces:

```bash
python train_real_v1_3.py --help
python scripts/infer_real.py --help
python scripts/reasoning_eval.py --help
```

No checkpoint is included. The exact historical experiments are therefore not
end-to-end reproducible from this seed alone. New experiments can be trained,
but they are new experiments and must not be presented as reproductions of the
checked-in aggregate evidence. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Evidence included

Only compact, row-free aggregates are present:

| Evidence | Current read |
| --- | --- |
| Cue-aligned Hotpot lineage | checkpoint `73000` is the visible local apex; the lane remains a basin/contract-stability probe, not proof of support-fact reasoning |
| Insufficiency-contact objective v0.1 | local margins were responsive, but matched held-out contact did not improve; `not_promoted_local_only` |

The evidence index and exact non-claims are in [docs/EVIDENCE.md](docs/EVIDENCE.md).

## What is deliberately absent

- model weights, model shards, and REAL/static checkpoints
- downloaded or copied benchmark datasets
- prompts, contexts, answers, and per-example benchmark rows
- private paths, notebook outputs, credentials, and personal metadata
- historical TODO archives and internal handover sediment
- sealed, ignored, or authority-gated evidence
- Heartbeat implementation and its large result banks

The absence is part of the release boundary, not an implication that those
artifacts never existed.

## Project contract

Read these in order:

1. [REAL_CONCEPT.md](REAL_CONCEPT.md)
2. [docs/RESEARCH_CONTRACT.md](docs/RESEARCH_CONTRACT.md)
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
4. [RELEASE_STATUS.md](RELEASE_STATUS.md)

## Development lineage

REAL emerged through sustained collaborative work between kiff and successive
ChatGPT/Codex model lineages. Those systems participated in conceptual
articulation, implementation, falsification framing, evidence discipline, and
release preparation.

That attribution records how the project formed. It is not scientific evidence,
an authorship claim by a model, or runtime authority. Empirical claims stand on
the checked artifacts; repository ownership and release decisions remain with
kiff.

## License and third-party material

Original material in this repository is released under Apache License 2.0.
Models, datasets, and dependencies are not relicensed by this repository and
remain governed by their upstream terms. No third-party model weights or
dataset rows are distributed here. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Until an archival identifier exists, cite the repository and a tagged release.
Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
