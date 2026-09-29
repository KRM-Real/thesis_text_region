# Prototype Implementation Plan

## Delivery objective

Build a tested, dataset-independent prototype skeleton first. Add the future user-supplied receipt dataset only after the end-to-end pipeline, fixtures, artifact contracts, and UI path are working.

This plan sequences work without changing the thesis scope or silently deciding unresolved methodology.

## Phase 0: Repository and scope guard

### Tasks

- Read README.md, PRD.md, ARCHITECTURE.md, DATASET.md, TEXT_REGION_PIPELINE.md, FEATURES.md, ML_PIPELINE.md, EVALUATION.md, PROTOTYPE_UI.md, TESTING.md, and REPRODUCIBILITY.md before coding.
- Confirm that the sources directory is not a runtime data dependency.
- Create the target repository structure and ignore rules for future data, weights, and generated artifacts.
- Add a configuration loader and run-identity helper.

### Exit gate

- A clean checkout can import the package with no user dataset.
- The test runner starts and the scope guard test confirms that no code path imports sources/.

## Phase 1: Contracts and fixture generator

### Tasks

- Implement image, region, feature, prediction, and artifact contracts.
- Implement canonical label normalization.
- Build a generated fixture-image set and manifest.
- Add the deterministic fixture detector.
- Add unit tests for schema and failure behavior.

### Exit gate

- Fixture manifest validation succeeds.
- Detector output contains geometry only.
- Fixture tests prove that one image produces one record.

## Phase 2: Dataset validation and split management

### Tasks

- Implement manifest loading and path safety checks.
- Implement decoding, dimensions, format, and checksum validation.
- Implement group-aware split creation.
- Implement exact-duplicate and related-group leakage checks.
- Produce audit and split artifacts.

### Exit gate

- Valid fixtures create a deterministic split manifest.
- Intentionally duplicated or cross-group fixtures fail the leakage gate.
- No real thesis dataset is required.

## Phase 3: Text-region pipeline

### Tasks

- Implement the detector adapter protocol.
- Implement geometry normalization, clipping, region ordering, crop padding, and overlays.
- Add no-detection and malformed-geometry handling.
- Create the detector feasibility report template.
- Integrate the selected production backend only after its dependency and weight requirements are documented.

### Exit gate

- Production adapter and fixture adapter satisfy the same contract.
- The pipeline can produce localization artifacts for fixture images.
- No recognized text is persisted.

## Phase 4: Visual features and aggregation

### Tasks

- Implement geometry and density features first.
- Implement spacing and alignment features.
- Implement edge, gradient, contrast, and sharpness features.
- Implement stroke, shape, and texture groups where the selected implementation is stable.
- Implement fixed-length aggregation and feature schema generation.
- Add feature quality and missingness reports.

### Exit gate

- Every enabled feature has a formula ID and test coverage.
- Feature schema is stable across repeated runs.
- Unsupported groups are explicit and do not receive fabricated values.

## Phase 5: Model pipelines

### Tasks

- Implement training-only imputation and scaling.
- Add Logistic Regression, SVM, and Random Forest adapters.
- Add small predeclared hyperparameter configuration.
- Implement validation-based model and threshold selection.
- Implement model-package save/load and schema checks.

### Exit gate

- All candidate models train on fixtures when the fixture size is sufficient.
- The same split and feature schema are used for all candidates.
- Model packages load and predict deterministically.

## Phase 6: Evaluation and reports

### Tasks

- Implement accuracy, precision, recall, F1-score, and confusion matrix.
- Implement model comparison and feature-group ablation reports.
- Implement per-image prediction records.
- Implement test-set access guard and configuration-hash checks.
- Implement a human-readable summary with limitations.

### Exit gate

- Invalid leakage or test-use conditions stop evaluation.
- Reports contain only measured fixture results and clearly identify them as tests.
- The reporting path can later accept a user dataset without code changes.

### Supporting feature diagnostics

- Keep the predeclared feature-group comparison as the primary feature-usefulness evidence.
- Add status-labeled individual-feature diagnostics for a frozen model: held-out permutation changes, class distributions, missingness, effect-size summaries, and linear coefficients/contributions where supported.
- Compute redundancy diagnostics from training and validation rows only; never use test diagnostics to select features or revise a model.
- Preserve `software_test`, `posthoc_pilot`, and `predeclared_final` status distinctions in configuration and reports.

### Exit gate

- Feature-analysis JSON/CSV artifacts are hashed and validated against the model, schema, split, and run manifest.
- Reports explain correlation and non-causality limits and distinguish fixture, post-hoc pilot, and predeclared analyses.

## Phase 7: Inference service and UI

### Tasks

- Implement one shared inference entry point.
- Add model-package compatibility checks.
- Add single-image input, processing status, region overlay, result, warning, and export screens.
- Add exact local contributions for supported linear models, and a global-only fallback for nonlinear or unsupported models.
- Add a “Why this label?” view and versioned explanation export without exposing OCR content.
- Add UI smoke tests.

### Exit gate

- The UI result matches command-line inference for the same fixture and model package.
- Supported local contributions reconstruct the fitted model score within tolerance; unsupported estimators never display invented contributions.
- No OCR text or semantic content appears as an explanation.
- Missing model, invalid input, no detection, and schema mismatch states are safe.

## Phase 8: Future dataset onboarding

This phase begins only when the user supplies the dataset after the prototype is done.

### Tasks

- Register the dataset under data/incoming/ without using sources/.
- Create and review the source manifest.
- Confirm the receipt-category inclusion rule.
- Record provenance and label decisions.
- Audit duplicates, near-duplicates, source shortcuts, and acquisition differences.
- Create the immutable split manifest.
- Run detector feasibility and localization review on development data.
- Freeze feature and model configurations.
- Run the final evaluation once.

### Exit gate

- Dataset audit, leakage gate, localization review, and configuration lock all pass.
- Final results are reported with bounded claims and no invented generalization.

## Codex implementation sequence

When implementing, work in small reviewable changes:

1. contracts and fixtures;
2. data validation and split gate;
3. localization adapter;
4. features and aggregation;
5. model package;
6. evaluation;
7. inference;
8. UI;
9. future dataset onboarding.

After each change:

- run the relevant tests;
- inspect generated artifacts;
- update the change report;
- do not modify thesis source documents;
- do not add the sources archive to runtime paths.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Future dataset does not match receipt scope | Block thesis-valid training and create a review report. |
| Detector output is unstable | Keep backend replaceable; freeze only after feasibility review. |
| No-detection rate is high | Report it, test the declared policy, and do not hide or impute it without a quality flag. |
| Feature count is too large for the sample size | Use predeclared group reduction or regularization; record the decision. |
| Class/source shortcuts dominate | Run the dataset shortcut audit and interpret results as limited. |
| Test leakage | Enforce split, artifact, and code-level guards. |
| Prototype diverges from experiment | Use one shared inference service and one model package. |
| Package dependency is unavailable | Use a fixture adapter for tests, document the missing production backend, and do not claim full pipeline completion. |

## Completion definition

The prototype is complete before the future dataset is added when:

- all core stages run on fixtures;
- all required artifacts are generated;
- acceptance criteria in PRD.md, TESTING.md, and REPRODUCIBILITY.md pass;
- the UI demonstrates the end-to-end flow;
- the implementation does not import or train on sources/dataset.zip;
- unresolved thesis decisions remain visible as configuration gates rather than hidden defaults.


