# Research Contract

REAL is a **contact system**.

The frozen backbone and the learned head are separable implementation components, but the project object is their coupled state. REAL begins when the learned head makes contact with the frozen manifold.

> **Contact makes them one system.**

---

## Non-negotiable invariants

1. The frozen backbone is the manifold: fixed weights, fixed latent structure, fixed collapse modes.
2. The learned head is a contact-making signal, not a second model mind and not a replacement for the backbone.
3. REAL is the coupled system: frozen manifold in contact with an input-conditioned dynamic head.
4. REAL does not correct, repair, teach, or update the frozen manifold.
5. StaticPrefix is a baseline and falsification reference, not the system of record.
6. Snapshots, probe rows, threshold atlases, detector maps, and final prefixes are measurements of the contact system. They are not the contact system.
7. A branch is central only when it improves, constrains, explains, or falsifies how REAL establishes, preserves, restores, loses, or fails to establish manifold contact.

---

## Regime roles

| Regime | Definition | Role |
|---|---|---|
| `base` | frozen manifold with no added learned contact layer | baseline |
| `static` | frozen manifold with a fixed/global learned contact field | falsification reference |
| `REAL` | frozen manifold in contact with an input-conditioned dynamic head | system under test |

Static-only conclusions do not stand in for REAL conclusions. REAL is supported only when the dynamic contact system explains or improves behavior beyond matched base/static alternatives.

---

## What counts as REAL-relevant work

REAL-relevant work does at least one of these:

1. improves contact formation between head and manifold
2. preserves useful contact across refinement or decode
3. restores contact after drift or scaffold break
4. measures REAL against matched `base` and `static` regimes
5. shows how a probe reveals or falsifies manifold contact
6. explains a failure mode as contact-never-formed, contact-lost, contact-disregarded, or snapshot-only

If a branch cannot answer one of these, it belongs in support or archive space rather than the center of `main`.

---

## Baseline-only work

Baseline-only work is comparison work that helps falsify the dynamic-contact claim.

Examples:

- `base -> static -> REAL` comparisons
- StaticPrefix sanity checks
- base-model parity checks
- prompt-only or sample-universe controls

Useful baseline work is still not central by itself. It becomes central when it constrains what can be claimed about REAL contact.

---

## Probe-only work

Probe-only work is acceptable when the probe’s role relative to REAL contact is explicit.

Required question:

> What contact problem would become harder to see or solve if this probe disappeared?

If that question cannot be answered cleanly, the probe should be treated as `audit_only` or moved toward archive status.

---

## Contact evidence levels

| Level | Meaning |
|---|---|
| `snapshot_only` | final output, final prefix, scalar row, or derived artifact only |
| `trajectory_observed` | stepwise REAL trace is observed, but contact with frozen-model structure is inferred indirectly |
| `manifold_contact_observed` | each step has measured evidence of contact, such as loss, margin, entropy, boundary likelihood, source coupling, scaffold viability, or abstention margin |
| `manifold_contact_used_by_head` | measured contact evidence shapes the head update, commit decision, or runtime refinement policy |

No level implies modifying frozen weights.

---

## Terminology discipline

Prefer:

- `contact system` for REAL as the coupled object
- `contact-making head` for the learned module
- `dynamic contact signal` for the head’s input-conditioned runtime signal
- `state refinement` for runtime `z_t -> z_{t+1}` dynamics
- `offline head fitting` for parameter updates to the learned head or related components
- `signal-shaping objective` for objectives that change how the contact signal is formed
- `manifold contact` for measured evidence of coupling to frozen-model structure

Use carefully:

- `controller` only for named implementation policies such as `refine_adaptive()` or `accept_reject_v1`; do not use it as the ontology of REAL.
- `steering` only when it is clear that the frozen manifold remains the source of language/knowledge and the question is contact, not domination.

Avoid:

- language implying that REAL is only the head
- language implying that the frozen model sits outside REAL once contact is formed
- treating final snapshots as the system
- using training as a generic explanation of runtime behavior

---

## Required question for future branches

Every future branch should answer this near the top:

> How does this branch show whether the REAL contact system establishes, preserves, restores, loses, or fails to establish contact with the frozen manifold?
