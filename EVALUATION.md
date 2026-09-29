# Evaluation and Reporting

## Evaluation purpose

Evaluation answers the bounded thesis question: whether the selected visual text-region representation provides useful information for distinguishing non-AI-generated and AI-generated receipt images under the approved dataset and testing conditions.

It does not answer whether the system authenticates real-world receipts, identifies a generator, or generalizes to every text-containing image.

## Evaluation principles

1. The image is the unit of analysis.
2. Every region derived from one image stays in the image's partition.
3. The test set is held out until all development choices are frozen.
4. The same partitions and feature schema are used across candidate classifiers.
5. Required metrics are reported together; no single score is treated as sufficient.
6. Feature-group results are interpreted as evidence under the experiment, not causal proof.
7. No result is invented when a run fails or a feature group is unavailable.

## Evaluation stages

### Stage 0: Dataset and leakage gate

Before training, verify:

- the audited manifest is complete and hashed;
- labels and receipt-category decisions are valid;
- exact duplicates are absent across partitions;
- near-duplicates, templates, related captures, and transformed copies are group-isolated;
- derived crops and masks are not treated as independent samples;
- source, acquisition, resolution, format, and content shortcuts are documented;
- both classes have adequate representation for the planned split.

### Stage 1: Development

Use only training and validation data to:

- choose the detector configuration after representative localization review;
- finalize feature formulas and enabled groups;
- fit imputation and scaling;
- choose feature subsets;
- choose model hyperparameters;
- select a decision threshold when needed.

Keep a development decision log with the reason, data used, and resulting configuration hash.

### Stage 2: Final test

After the configuration is frozen:

1. refit the selected pipeline on training plus validation data only;
2. evaluate on the untouched test split;
3. save per-image predictions and aggregate metrics;
4. do not revise the model because of test errors;
5. if a test problem is discovered, mark the run invalid and create a new run after fixing the configuration.

## Required metrics

The positive class is ai_generated unless the final approved methodology explicitly records another convention.

Report:

- accuracy;
- precision for the positive class;
- recall for the positive class;
- F1-score for the positive class;
- confusion matrix with true negatives, false positives, false negatives, and true positives.

Definitions:

```text
accuracy  = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

When a denominator is zero, report an explicit undefined or configured zero-division result and record the convention. Do not hide it.

## Optional diagnostics

The following may be reported as diagnostics when justified by the final class distribution or threshold behavior, but they do not replace the required metrics:

- balanced accuracy;
- Matthews correlation coefficient;
- precision-recall curve or area;
- threshold table;
- calibration diagnostics;
- bootstrap or repeated-split uncertainty intervals.

Adding an optional diagnostic is not a change to the thesis scope. Present it as supporting evidence and document why it was included.

## Classifier comparison

The comparison table must use:

- the same image IDs;
- the same split manifest;
- the same feature schema;
- the same positive-class definition;
- the same test set;
- comparable threshold rules.

For each model, save validation and test rows separately. A model that wins on the test set after test inspection is not a valid selected model.

## Feature-group comparison

The prototype should compare the full representation with predeclared groups:

1. geometry and density;
2. geometry plus spacing and alignment;
3. stroke and shape;
4. edges, gradients, contrast, and sharpness;
5. texture and local pixel relationships;
6. all enabled groups.

For each group, report:

- feature count;
- missingness and no-detection rate;
- model and settings;
- validation metrics;
- final test metrics;
- changes relative to the full representation;
- limitations or unavailable calculations.

Do not claim that a group causes a result. Correlated features can share or obscure information.

## Individual-feature diagnostics

Individual-feature diagnostics are supporting interpretation, not a replacement for the predeclared feature-group comparison. For a frozen model, a report may show held-out permutation changes in F1, accuracy, and model score; class-wise medians and interquartile ranges; missingness; Cliff's delta; and linear coefficients or mean absolute contributions when supported. These values describe the fitted model and split and must not be presented as causal effects.

The analysis status must be recorded in configuration and every report:

- `software_test` identifies generated-fixture checks and is not thesis evidence;
- `posthoc_pilot` identifies exploratory diagnostics computed after the model/test split was already used;
- `predeclared_final` is reserved for an analysis specified before final evaluation.

Feature correlations/redundancy are calculated on training and validation rows only. Held-out permutation results are descriptive and must not be used to select features, tune the model, revise the threshold, or make a second claim of test performance. Correlated features may share or obscure importance. Quality and no-detection fields describe processing conditions, not image origin.

When local explanations are available for an individual prediction, Logistic Regression values use log-odds and linear SVM values use decision-margin units. The explanation must reconstruct the model output and state its scale. Nonlinear or unsupported models are reported as global-only; no additive attribution should be fabricated.

## Leakage checks

The evaluation runner must fail closed when any of these checks fails:

| Check | Rule |
|---|---|
| Duplicate checksum | One exact file hash cannot occur in more than one split. |
| Related group | One group ID cannot occur in more than one split. |
| Crop lineage | Every crop inherits the source image split. |
| Preprocessing fit | Imputer, scaler, selector, and threshold use development data only. |
| Label leakage | Feature columns contain no label, path, source, or semantic text fields unless explicitly approved as diagnostic-only and excluded from the model. |
| Test access | The test set is not loaded by tuning or feature-selection code. |
| Configuration drift | The test report's hashes match the saved detector, feature, model, and split artifacts. |

## Required output artifacts

For each run, write:

- dataset audit summary;
- split manifest and checksum;
- detector configuration and localization summary;
- feature schema and feature quality report;
- model comparison table;
- validation metrics;
- test metrics;
- confusion matrix CSV or image;
- per-image test predictions;
- feature-group ablation report;
- error review table;
- run manifest and environment record;
- a human-readable limitations section.

Suggested report layout:

```text
artifacts/reports/<run-id>/
  summary.md
  dataset_audit.json
  split_manifest.csv
  model_comparison.csv
  validation_metrics.json
  test_metrics.json
  confusion_matrix.csv
  test_predictions.csv
  feature_group_ablation.csv
  feature_analysis.json
  feature_influence.csv
  feature_group_summary.csv
  feature_redundancy.csv
  error_review.csv
  run_manifest.json
```

Prediction-time local explanations are exported separately as `prediction-explanation-2.json` with the prediction record. Their model, input, schema, and interpretation status must remain traceable to the same inference run.

## Error review

The error review is descriptive, not a route to untracked tuning. For false positives and false negatives, record:

- sample ID and checksum;
- true and predicted labels;
- model score and threshold;
- region count and no-detection status;
- quality flags;
- acquisition or provenance notes that are already permitted by the manifest;
- a visual review note that does not transcribe or interpret the receipt text.

If the error review suggests a dataset problem, create a new run after the issue is resolved. Do not quietly remove test images.

## Interpretation boundaries

Permitted conclusions:

- the representation was or was not useful under the selected conditions;
- one candidate classifier produced stronger or weaker measured performance;
- some feature groups were more stable, less stable, or unavailable;
- localization and acquisition conditions limited the result.

Prohibited conclusions without separate approved experiments:

- universal detector performance;
- reliable authenticity for any new receipt;
- exact generator identification;
- generalization to non-receipt text-containing images;
- robustness to unseen generators, languages, compression, blur, resizing, or re-digitization;
- semantic or transaction-level validity.

## Acceptance criteria

- Required metrics and confusion matrices are produced for every valid model comparison.
- The positive class and averaging convention are stated in every report.
- Test evaluation uses an untouched test split and a frozen configuration.
- Leakage gates fail the run rather than producing a possibly inflated result.
- Feature-group comparisons use predeclared configurations.
- Reports contain no invented counts, scores, or conclusions when the run did not complete.
- Every reported value can be traced to a run ID, input manifest, model package, and configuration hash.

