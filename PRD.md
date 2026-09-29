# Prototype Product Requirements

## Purpose

This document defines the functional requirements for a research prototype that demonstrates the thesis pipeline:

```text
receipt image -> text-region localization -> visual region features -> image-level aggregation -> traditional classifier -> prediction
```

The prototype is a demonstration of the research pipeline. It is not a production authenticity detector, a receipt-fraud system, an OCR application, or a universal detector for every image category or generator.

## Scope locks

The following decisions are fixed by the thesis context and must not be changed by implementation convenience:

| Decision | Requirement |
|---|---|
| Research task | Binary classification of non-AI-generated versus AI-generated text-containing images. |
| Controlled category | Receipt images are the selected experimental image category. |
| Evidence | Use visual measurements from localized text regions. |
| Text semantics | Do not use recognized words, spelling, language, topic, arithmetic, or meaning. |
| Unit of analysis | The image is the unit of classification. Regions and crops are intermediate measurements. |
| Model comparison | Compare traditional classifiers such as Logistic Regression, SVM, and Random Forest, subject to feasibility and validation. |
| Main contribution | A documented, reproducible experimental pipeline and evidence about the usefulness and limitations of the representation. |
| Prototype role | Demonstrate the evaluated pipeline using the same detector, feature definitions, aggregation, preprocessing, and model artifact. |
| Claims | Results apply only to the selected dataset and controlled testing conditions. They must not be presented as universal authenticity judgments. |

The supplied archive under `sources/` is not runtime data for this prototype. The implementation must not copy, import, train on, or hard-code paths to that archive. The future dataset will be added through the contract in [DATASET.md](DATASET.md).

## Audience and primary use cases

### Researcher or developer

The user can run the same pipeline used for the thesis experiment, inspect intermediate artifacts, and reproduce a prediction from a saved configuration and model package.

### Reviewer or adviser

The user can see which regions were localized, which feature groups were computed, which model version was used, and what limitations apply to the prediction without seeing or relying on OCR text.

### Experiment operator

The user can validate a future dataset, create leakage-controlled partitions, train the candidate classifiers, compare metrics, and export a complete run record.

## Product goals

1. Make the thesis pipeline executable from input image to image-level prediction.
2. Make every transformation inspectable and repeatable.
3. Keep text localization separate from text recognition and semantics.
4. Keep all regions from one image in the same data partition.
5. Support fair comparison of the candidate traditional classifiers on the same image-level feature matrix.
6. Fail clearly when an input, detector, feature group, or model artifact is invalid.
7. Preserve the distinction between a prototype prediction and an authenticity claim.

## Non-goals

The prototype must not:

- transcribe or display recognized receipt text as an analytical result;
- classify individual crops as independent samples;
- determine whether a receipt records a legitimate transaction;
- verify provenance, legal authenticity, or document tampering;
- identify the exact image generator;
- claim cross-generator, cross-category, multilingual, or degradation robustness unless a separately approved experiment adds those tests;
- train a new deep-learning detector as part of the prototype;
- treat the future dataset as available before it is supplied and audited.

## Functional requirements

### Input and dataset requirements

- **FR-001:** Accept a future dataset through the canonical manifest and folder contract in [DATASET.md](DATASET.md).
- **FR-002:** Normalize input labels to `non_ai_generated` and `ai_generated`; accept `real` only as an explicit alias that is recorded as a normalization event.
- **FR-003:** Validate file existence, decodability, supported format, dimensions, label, category, provenance fields, and checksum before training.
- **FR-004:** Require a receipt-category inclusion decision for thesis-valid training. Images that are only generic posters, advertisements, or other categories must be flagged for review rather than silently included.
- **FR-005:** Create or load a saved image-level split manifest before fitting any data-dependent transformation.

### Localization requirements

- **FR-006:** Run a configurable text-region detector that returns geometry and confidence, not recognized text as a feature.
- **FR-007:** Apply the same detector configuration and preprocessing to both classes.
- **FR-008:** Save normalized region geometry and detection quality metadata for every image.
- **FR-009:** Keep the no-detection case explicit and auditable; do not silently drop images.

### Feature requirements

- **FR-010:** Compute only visual region features defined in [FEATURES.md](FEATURES.md).
- **FR-011:** Support geometry/density, stroke/shape, spacing/alignment, edge/gradient, contrast/sharpness, and texture feature groups.
- **FR-012:** Aggregate a variable number of regions into one fixed-length vector per image.
- **FR-013:** Store a feature schema and feature provenance with every feature matrix.
- **FR-014:** Ensure feature extraction never receives recognized words, language predictions, semantic embeddings, or OCR confidence as classifier inputs.

### Model requirements

- **FR-015:** Train Logistic Regression, SVM, and Random Forest when the selected configuration and available data support them.
- **FR-016:** Use identical image partitions and the same feature records for the model comparison.
- **FR-017:** Fit imputation, scaling, feature selection, and threshold selection only on training/development data.
- **FR-018:** Persist the selected model, preprocessing pipeline, feature schema, class-label mapping, threshold, detector configuration, and software metadata as one versioned model package.
- **FR-019:** Return an image-level class label, class scores or probabilities when supported, and the model package identifier.

### UI and reporting requirements

- **FR-020:** Allow a user to submit one image for inference after a compatible model package is loaded.
- **FR-021:** Show a region overlay and numeric processing summary without displaying recognized text.
- **FR-022:** Show the predicted class using the bounded terms `AI-generated` or `Non-AI-generated under the dataset rules`.
- **FR-023:** Show warnings for unsupported category, no detected regions, low detector confidence, missing model artifacts, or input conditions outside the training contract.
- **FR-024:** Export a prediction record that includes the input checksum, configuration, model identifier, region count, feature vector checksum, prediction, score, and warnings.
- **FR-025:** Evaluation reports may include supporting individual-feature diagnostics for the frozen model, including held-out permutation changes, class-wise feature distributions, missingness, and model coefficients where available. Feature-group comparison remains the primary feature-usefulness evidence; individual-feature results are descriptive, not causal.
- **FR-026:** Provide a local explanation only when the fitted model supports a faithful additive account. Logistic Regression contributions use log-odds and linear SVM contributions use decision-margin units; nonlinear or unsupported models must be labeled global-only.
- **FR-027:** Export the local explanation and interpretation status as a versioned JSON artifact alongside the prediction record, without recognized text or semantic information.

## Prototype defaults

These are implementation defaults, not new thesis decisions. They may be changed after a documented feasibility check without changing the research question.

| Area | Default |
|---|---|
| Runtime | Python application with a command-line pipeline and a small local UI. |
| Canonical data entry | A user-supplied `data/manifests/audited_manifest.csv` plus image files under `data/incoming/`. |
| Labels | `non_ai_generated` and `ai_generated`; display labels are `Non-AI-generated` and `AI-generated`. |
| Split | Stratified group-aware train/validation/test split, initially 70/15/15 when the final sample size supports it. |
| Detector | A replaceable text-detector adapter selected through the detector feasibility gate; no detector name is thesis-fixed. |
| Region representation | Bounding boxes or polygons converted to normalized coordinates and padded visual crops. |
| Missing regions | Retain the image row with an explicit no-detection indicator and a documented missing-value policy. |
| Model threshold | Select on validation data; use 0.5 only when no alternative threshold is justified and record that choice. |
| UI | Single-image inference first; batch inference is an implementation extension after the single-image flow passes. |
| Artifact format | JSON for metadata, CSV or Parquet for tabular records, and a versioned model directory. |

## Acceptance criteria

The prototype is accepted when all of the following are true:

- A clean checkout can install the documented environment without requiring the sources archive.
- A fixture dataset can complete the full flow from manifest validation through prediction.
- A future user-provided receipt dataset can be onboarded without changing source code paths.
- Every image receives one image-level record, including an explicit status for no detections or invalid processing.
- The detector never contributes recognized text to the feature matrix.
- A saved split manifest proves that one source image and all derived regions remain in one partition.
- The three candidate classifier adapters consume the same feature schema and split manifest.
- The final test set is not used for detector tuning, feature selection, hyperparameter selection, or threshold selection.
- The UI prediction is reproducible from the saved model package and the input checksum.
- The UI displays a clear research-use warning and never claims to establish authenticity.
- Supported local explanations reconstruct the fitted model score within a declared tolerance; unsupported models fall back to measured visual values and global diagnostics.
- Individual-feature analysis is status-labeled, uses development-only rows for redundancy diagnostics, and does not replace the predeclared feature-group comparison.
- Automated tests cover valid inputs, invalid inputs, no detections, malformed detector output, leakage checks, artifact loading, deterministic inference, and UI smoke behavior.
- The implementation passes the checklist in [TESTING.md](TESTING.md) and produces the artifacts in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Open decisions that remain explicit

The following items must be finalized during implementation and recorded in the run manifest:

1. final receipt data source, provenance, sample count, and inclusion rules;
2. detector backend, version, weights, geometry output, and threshold;
3. exact feature formulas and any unavailable feature groups;
4. aggregation statistics and no-detection handling;
5. final classifier hyperparameters and whether SVM uses a linear or nonlinear kernel;
6. split proportions and group-definition rules if the final dataset requires a different design.


