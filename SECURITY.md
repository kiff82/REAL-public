# Security and Responsible Disclosure

REAL is research code, not a hardened production system. Do not expose its
training, inference, or evaluation processes directly to untrusted users.

## Reporting a problem

For security-sensitive findings, use the repository's private security-advisory
channel rather than a public issue. Include the affected revision, a minimal
reproduction, and the expected impact. Do not attach credentials, private
datasets, or model weights to a public report.

## Release hygiene

The public tree must not contain:

- credentials, API keys, cookies, or private-key material
- absolute user/home paths or personal metadata
- raw private prompts or benchmark source rows
- model shards, checkpoints, optimizer states, or cache contents
- sealed/internal evidence or authority packages

Run before every release:

```bash
python scripts/release_audit.py
python scripts/build_release_manifest.py --check
```

The scanner is a release guard, not a proof that the repository is safe. Review
the complete diff and, for any repository with prior history, scan every branch,
tag, and historical object before making it public.

## Model execution

Loading remote model code is not required by this project and should remain
disabled. Pin immutable model revisions, review upstream artifacts, and avoid
executing untrusted checkpoints. Historical REAL code expects ordinary
Transformers model loading and does not grant runtime authority to any learned
head or probe.
