# REAL as a Contact System

REAL means **Recursive External Alignment Layer**.

In this repo, REAL is not merely a learned head and not merely a frozen model. REAL is the **coupled system** formed when a learned input-conditioned signal makes contact with a frozen language-model manifold.

> **Contact makes them one system.**

That sentence is the core premise.

---

## The core insight

The frozen model already contains a rich latent landscape: knowledge fragments, reasoning fragments, source-coupled paths, abstention-compatible regions, scaffold trajectories, and collapse modes.

Failures are therefore not always failures of missing stored knowledge. Some failures are failures of:

- entering the wrong region
- failing to enter a needed region at all
- losing contact after a useful state appears
- collapsing to tokens before the better trajectory becomes stable
- ignoring an insufficiency-compatible state during decode

REAL tests whether a small dynamic contact layer can improve those state-access and stability failures without modifying the frozen backbone.

---

## What REAL is

REAL is the **contact system** formed by:

1. **Frozen manifold** — the fixed backbone and its already-present latent structure.
2. **Contact-making head** — a small learned module that emits an input-conditioned dynamic signal during the current request.

REAL is evaluated through an **evidence layer**: probes and energy/loss reads used to test whether the signal is actually in contact with useful frozen-model structure. The evidence layer is not a third component of REAL; it is how the contact system is tested.

The head does not become the protagonist. The backbone does not become a passive object outside REAL. The object of study is their coupled behavior.

Implementation components are separable. The system under study is not.

---

## What REAL is not

REAL is not:

- a second mind
- a planner
- a tool user
- persistent memory
- an autonomous agent
- a backbone weight update
- a guarantee that the model already knows every answer

REAL does not correct, repair, teach, or update the frozen manifold. It changes the conditions under which that manifold is contacted during the current request.

---

## Regime definitions

| Regime | Definition |
|---|---|
| `base` | frozen manifold with no added learned contact layer |
| `static` | frozen manifold with a fixed/global learned contact field |
| `REAL` | frozen manifold in contact with an input-conditioned dynamic head |

The `static` regime is not an enemy. It is the falsification baseline. If static explains the gain, REAL has not shown dynamic contact.

---

## Basic mechanism

REAL touches the frozen backbone through a narrow interface:

- **Read:** summarize the prompt state, usually as pooled prompt embeddings `h`.
- **Refine:** evolve a latent state `z_t` across a small number of steps.
- **Measure:** during training and analysis, compare each refinement state to frozen-model energy/loss or other contact evidence.
- **Write:** turn the selected/refined latent into a small prefix signal.
- **Decode:** let the frozen backbone resolve the continuation under that signal.

Conceptual loop:

```text
prompt -> frozen-manifold state read -> contact-state refinement -> prefix signal -> frozen-manifold decode
```

Implementation sketch:

```text
z_{t+1} = z_t + f(z_t, h, e_t, Δe_t, step_embed)
```

where `e_t` is the head’s predicted energy, trained against measured frozen-model energy.

---

## What “state alignment” means here

State alignment means improving the coupled system’s ability to:

1. **enter** the relevant latent region
2. **stay** there under small perturbations or longer generation
3. **transition** across scaffold boundaries without collapse
4. **abstain** when the provided context does not support an answer
5. **remain source-coupled** when the task depends on supplied evidence

These are measured, not assumed.

---

## Manifold contact

A branch is about REAL only when it says something about contact.

Examples of contact evidence include:

- stepwise frozen-model CE loss
- predicted-vs-measured energy calibration
- entropy and logit-margin pressure
- source-occlusion sensitivity
- scaffold-boundary survival
- answer-vs-abstain candidate margins
- per-refinement-step replay traces

A final answer alone is usually not enough. It is a snapshot. REAL is a trajectory/contact question.

---

## Current failure read

The current insufficiency-contact result says the most important false-certainty failure mode is often:

```text
never_found_insufficiency_contact
```

That means the coupled system did not enter an insufficiency-compatible region before decode collapse.

This is not merely “the head failed” and not proof that “the model lacked the state.” It is a failure of the contact relation. The next objective should therefore shape contact formation, especially for insufficiency/abstention states.

---

## The vision in one sentence

REAL is a frozen language-model manifold made dynamically contactable by a small learned head, tested by whether that coupled system can establish, preserve, or restore useful latent contact better than no signal or a fixed global signal.
