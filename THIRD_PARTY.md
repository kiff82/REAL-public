# Third-Party Models, Data, and Software

Apache License 2.0 applies only to original material distributed in this
repository. It does not relicense any model, dataset, dependency, or upstream
content.

No third-party model weights or benchmark source rows are included in this
public seed.

## Model referenced by the latest preflight

- `Qwen/Qwen3-4B-Instruct-2507`
- research revision recorded by the private source:
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- upstream model card and terms:
  <https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507>
- upstream license identifier at release preparation: Apache-2.0

Downloading or using the model is a separate act governed by its upstream
terms. This repository does not provide the model or certify a local copy.

## Datasets referenced by the research code

The training and evaluation entry points can request datasets from upstream
providers. Users must review the exact dataset card, revision, source-material
terms, and intended use before downloading or redistributing anything.

- HotpotQA: <https://hotpotqa.github.io/> — upstream states CC BY-SA 4.0.
- SQuAD / SQuAD 2.0: <https://rajpurkar.github.io/SQuAD-explorer/>.
- FEVER: <https://fever.ai/dataset/fever.html> — annotations incorporate
  Wikipedia material and carry upstream attribution/share-alike conditions.

The code also exposes optional loaders for BoolQ, ARC, MMLU-Pro, GPQA-diamond,
AIME, TruthfulQA, XSum, and other Hugging Face datasets. Their presence as a
loader name is not permission to redistribute their content.

## Python dependencies

PyTorch, Transformers, Accelerate, Datasets, NumPy, scikit-learn, and optional
bitsandbytes are installed from their upstream distributions and retain their
own licenses. Dependency installation is not bundled into this repository.

## Evidence aggregates

Files below `evidence/` contain aggregate measurements and no intentionally
redistributed prompt, context, answer, or per-example dataset rows. If a future
contribution adds row-level material, its provenance and redistribution rights
must be reviewed before merge.
