# Contributing

Contributions are welcome when they preserve the contact-system research
contract and make claims easier to falsify.

## Before opening a change

1. Read `REAL_CONCEPT.md`, `docs/RESEARCH_CONTRACT.md`, and
   `RELEASE_STATUS.md`.
2. State which contact question the change addresses: establish, preserve,
   restore, lose, or fail to establish contact.
3. Keep backbone parameters frozen unless proposing a clearly separate project.
4. Preserve matched `base`, `static`, and `REAL` comparisons.
5. Put new behavior behind an explicit option until its compatibility and
   evidence boundaries are established.

## Evidence requirements

Do not describe a gain as dynamic contact unless it survives a matched static
comparison. Do not describe forced-answer accuracy as reasoning by itself. Do
not turn a measurement into a detector, controller, or runtime policy without
separate held-out evidence.

Every empirical contribution should report model revision, checkpoint hashes,
dataset revision and split, sample universe, seeds, decode/refinement policy,
and exact metric protocol. Negative and null results belong in the record.

## Public-data boundary

Do not commit dataset rows, copied prompts/contexts, checkpoints, model shards,
personal paths, secrets, or unreviewed generated outputs. Aggregate evidence is
preferred. Row-level evidence requires explicit provenance and redistribution
review.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q real scripts utils train_real_v1_3.py examples tests
python scripts/release_audit.py
python scripts/build_release_manifest.py --check
```
