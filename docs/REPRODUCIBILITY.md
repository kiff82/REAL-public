# Reproducibility Boundary

The public seed supports code inspection, interface testing, the structural
demo, and new experiments. It does not yet support exact end-to-end replay of
the historical aggregate evidence.

## Why exact replay is blocked

The release intentionally omits:

- the historical static checkpoint
- REAL head checkpoints `72000`, `73000`, and `73600`
- frozen-backbone weight shards
- raw benchmark rows and selected manifests

The latest contact-observability preflight also records six unresolved
raw-byte identities: three checkpoints and three Qwen3 shards. Until those
identities and redistribution boundaries are resolved, an exact public
execution package would overstate what is reproducible.

## What can be reproduced now

```bash
python examples/structural_contact_demo.py
python -m unittest discover -s tests -v
python -m compileall -q real scripts utils train_real_v1_3.py examples tests
python scripts/release_audit.py
```

With the research dependencies installed, all CLIs can be inspected and new
training/evaluation runs can be launched. Any resulting checkpoint is a new
lineage.

## Minimum record for a new run

Record at least:

- backbone identifier, immutable revision, and weight-shard hashes
- tokenizer/config hashes
- REAL/static checkpoint hashes
- dataset identifier, configuration, revision, split, and sample IDs
- prompt and output contracts
- maximum input and output lengths
- random seeds and deterministic settings
- hardware, dtype, quantization, and dependency lock
- base/static/REAL metrics under the same signature
- all selection rules and any held-out evaluation boundary

## Do not silently substitute

A similarly named model, regenerated checkpoint, changed dataset revision, or
different sample order is not an exact reproduction. Report it as a new run and
compare only after the signatures are made explicit.
