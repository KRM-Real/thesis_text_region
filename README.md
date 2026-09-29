# Thesis Prototype Specification Package

## What this package describes

This package specifies a complete research prototype for:

> Machine Learning Classification of AI-Generated and Real Text-Containing Images Using Visual Text-Region Features

The pipeline is:

```text
future user-supplied receipt dataset
  -> audited manifest and leakage-controlled split
  -> text-region localization
  -> nonsemantic visual feature extraction
  -> region-to-image aggregation
  -> Logistic Regression / SVM / Random Forest comparison
  -> bounded image-level prediction, evaluation report, and model-supported explanation
```

## Public demo and Streamlit deployment

This repository contains a runnable receipt-focused pilot demo. The app entrypoint is
`streamlit_app.py` and the public deployment uses the selected pilot model package and
its matching evaluation report. The model measures visual text-region geometry and
pixels; it does not use recognized receipt words or meaning.

### Run locally

Use Python 3.13 to match the saved model package, then install the pinned runtime:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

EasyOCR downloads its detection-only CRAFT weights into the local `easyocr-models/`
cache the first time an image is analyzed. Those weights are intentionally not part of
the repository. The detector is initialized once and reused across Streamlit reruns.

### Deploy on Streamlit Community Cloud

Create an app with these settings:

| Setting | Value |
|---|---|
| Repository | `KRM-Real/thesis_text_region` |
| Branch | `main` |
| Main file | `streamlit_app.py` |
| Python | `3.13` |
| Secrets | None required |

The first analysis may take longer while the detector weights are downloaded. Later
analyses reuse the cached detector within the running app.

### Research-use boundary

The public app is a thesis prototype for the controlled receipt category. Its output is
a model classification under the saved pilot dataset rules, not proof of authenticity,
provenance, fraud, or image origin. The evaluation panel is a fixed report from the
held-out pilot run; it does not change when a new image is uploaded.

The public release contains source code, tests, configuration, one selected model
package, and the report artifacts required by the UI. It does not contain receipt
images, the dataset, raw feature matrices, detector weights, or unrelated experiment
runs.

The image remains the unit of classification. Regions and crops are intermediate measurement units.

Evaluation keeps the predeclared feature-group comparison primary and provides status-labeled individual-feature diagnostics as supporting evidence. At inference time, the UI shows a per-image explanation only for model families where contributions can be verified; the saved evaluation report remains a summary of its selected held-out run and does not change with the uploaded image.

## Important data rule

The synced sources directory is read-only reference material. The supplied archive in sources/dataset.zip is not used, copied, imported, or treated as the prototype dataset. The prototype must be implemented and tested without it. The eventual dataset will be added later through the contract in DATASET.md.

This rule is intentional because the prototype specification must remain aligned with the approved receipt scope rather than silently adopting an unapproved or mismatched archive.

## Thesis-fixed decisions

- Binary target: non-AI-generated versus AI-generated text-containing images.
- Controlled image category: receipts.
- Evidence: visual features from localized text regions.
- Exclusion: recognized words, spelling, language, topic, arithmetic, and semantics.
- Unit of analysis: one image.
- Candidate traditional classifiers: Logistic Regression, SVM, and Random Forest, subject to feasibility and validation.
- Required evaluation: accuracy, precision, recall, F1-score, and confusion matrix.
- Main contribution: a reproducible experimental pipeline and evidence about usefulness and limitations.
- Prototype role: demonstration of the evaluated research pipeline, not a production authenticity detector.

## Prototype defaults

The documents use clearly labeled engineering defaults so implementation can begin without pretending that unresolved thesis decisions are already final:

- Python application with command-line and local UI entry points;
- future manifest under data/manifests/ and user images under data/incoming/;
- group-aware train/validation/test split, initially approximately 70/15/15 when feasible;
- replaceable text-detector adapter;
- explicit no-detection record;
- fixed-length aggregation using documented summary statistics;
- versioned JSON, CSV or Parquet, and model-directory artifacts;
- fixture-only tests before the future dataset is supplied.

These defaults can change after feasibility testing. Any change must be recorded in the run configuration and must not change the thesis research question without approval.

## File map

| File | Role |
|---|---|
| PRD.md | Product requirements, scope locks, functional requirements, defaults, and acceptance criteria. |
| AGENTS.md | Existing project guard plus implementation instructions for coding agents. |
| ARCHITECTURE.md | Modules, interfaces, repository structure, data flow, and failure boundaries. |
| DATASET.md | Future dataset contract, manifest schema, onboarding, provenance, and leakage controls. |
| TEXT_REGION_PIPELINE.md | Detector adapter, geometry, crop/mask policy, semantics exclusion, and failure handling. |
| FEATURES.md | Candidate visual feature groups, formulas, aggregation, schema, and missing-value rules. |
| ML_PIPELINE.md | Splits, preprocessing, candidate models, model packages, inference, and CLI targets. |
| EVALUATION.md | Evaluation stages, metrics, model comparison, ablation, leakage gates, and claim boundaries. |
| PROTOTYPE_UI.md | Single-image UI flow, result contract, warnings, and accessibility expectations. |
| TESTING.md | Unit, integration, leakage, reproducibility, and UI smoke test strategy. |
| IMPLEMENTATION_PLAN.md | Staged implementation order with gates and a future dataset-onboarding phase. |
| REPRODUCIBILITY.md | Run IDs, hashes, configuration snapshots, artifact layout, and rerun protocol. |

## Reading order for implementation

1. This README
2. PRD.md
3. ARCHITECTURE.md
4. DATASET.md
5. TEXT_REGION_PIPELINE.md
6. FEATURES.md
7. ML_PIPELINE.md
8. EVALUATION.md
9. PROTOTYPE_UI.md
10. TESTING.md
11. REPRODUCIBILITY.md
12. IMPLEMENTATION_PLAN.md

## Target repository layout

```text
configs/       versioned configuration and feature schemas
data/          future user data, manifests, and test fixtures
artifacts/     generated localization, feature, model, and report outputs
src/           reusable pipeline modules
ui/            thin local presentation layer
tests/         unit, contract, integration, leakage, and smoke tests
```

User data, detector weights, and large model files should be ignored by version control by default.

## Intended implementation flow

Before the future dataset exists:

1. create the contracts and repository skeleton;
2. generate fixtures;
3. implement and test the detector adapter boundary;
4. implement visual features and aggregation;
5. train candidate models on fixtures only as a software test;
6. build the model package and inference service;
7. build the UI;
8. pass all acceptance and reproducibility checks.

After the user supplies the dataset:

1. create the source and audited manifests;
2. verify receipt-category inclusion and provenance;
3. run duplicate, near-duplicate, and shortcut audits;
4. create and lock the split manifest;
5. select and freeze the detector and feature configuration;
6. run the final comparison and evaluation;
7. report limitations and bounded conclusions.

## Grounding and source boundaries

The specification package is grounded in the thesis materials already present in this project:

- the receipt-focused Chapter 1 in sources/Receipt_Chapter1.docx;
- the current Chapter 2 manuscript in sources/Chapter 2 .docx;
- the review and alignment guide in Chapter_2_Review_and_Mastery_Guide.md;
- the project constraints in AGENTS.md.

The package preserves the current Chapter 1 and Chapter 2 decisions. It does not add research results, final sample counts, final detector selection, final feature values, or final classifier results.

## Completion standard

The package is implementation-ready when a coding agent can build the fixture-based end-to-end prototype without asking where to put user data, how to avoid OCR semantics, how to preserve image-level splits, how to serialize artifacts, or how to test the workflow. The future dataset remains a controlled onboarding step, not an implicit dependency.

