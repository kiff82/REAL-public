# Evidence Index

This release includes compact aggregate evidence selected to preserve both
positive shape and failed promotions. It excludes benchmark source rows and
private developmental artifacts.

## Cue-aligned Hotpot lineage

Machine-readable aggregate: [REAL_EVIDENCE_V0_1.json](../evidence/REAL_EVIDENCE_V0_1.json)

Matched `forward` runs on 256 rows place checkpoint `73000` above `72000` and
`73600` on answer exact match and full keyfacts-contract completion:

| Checkpoint | Answer EM | Full contract |
| --- | ---: | ---: |
| `72000` | 0.2265625 | 0.5390625 |
| `73000` | 0.2890625 | 0.6484375 |
| `73600` | 0.2109375 | 0.515625 |

Interpretation is deliberately narrow: `73000` is a visible apex on this
lineage, and `73600` breaks the requested scaffold earlier. The lane remains a
forced-contract/basin probe. It does not establish general support-fact
reasoning or grounding.

## Insufficiency-contact objective v0.1

Machine-readable aggregate: [REAL_EVIDENCE_V0_1.json](../evidence/REAL_EVIDENCE_V0_1.json)

The objective moved candidate margins in a local microscope. On the held-out
48-row selected manifest, however, false-certainty
`never_found_insufficiency_contact` remained `11` for matched control,
`objective_final`, and `objective_early_final`.

Decision: `not_promoted_local_only`.

This is evidence that the objective is connected to the local signal, not that
it generalizes or improves abstention.

## Evidence discipline

When adding public results:

- publish matched base/static/REAL signatures
- identify model and revision, checkpoint bytes, dataset/split, sample universe,
  seed, decode policy, and metric protocol
- separate final-output snapshots from trajectory/contact evidence
- include negative, null, and failed-holdout results
- never call a selected diagnostic manifest representative without evidence
- do not promote a probe into a detector, controller, or runtime policy merely
  because it separates one checked-in universe
