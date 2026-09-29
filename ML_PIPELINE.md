# Machine Learning Pipeline

## Purpose

This pipeline converts one image-level feature row into a binary prediction and a reproducible evaluation record. It compares traditional classifiers on the same feature schema and partitions.

## End-to-end stages

1. Load the audited manifest.
2. Load or create the group-aware split manifest.
3. Run the locked text-region and feature pipeline.
4. Validate that exactly one feature row exists per image.
5. Fit training-only preprocessing.
6. Train the candidate classifiers.
7. Select hyperparameters and, if needed, a decision threshold using development data only.
8. Refit the selected configuration on the permitted development data.
9. Evaluate once on the untouched test split.
10. Save the model package, predictions, metrics, and run manifest.
11. Use the same package for prototype inference.

No stage may use the supplied sources archive as data. Before the future dataset is supplied, use only test fixtures.

## Label contract

Canonical internal labels:

```text
0 = non_ai_generated
1 = ai_generated
```

Display labels:

```text
Non-AI-generated under the dataset rules
AI-generated
```

The word real may appear as a documented input alias, but it must be normalized and must not be used in user-facing claims that imply universal authenticity.

## Split policy

The split manifest is image-level and group-aware. It must be created before fitting any imputer, scaler, selector, or model setting.

Required properties:

- every included sample appears exactly once;
- each group appears in exactly one split;
- both labels are represented in each required split when the sample size permits;
- split seed and algorithm version are recorded;
- the split manifest is hashed and immutable for the final run.

Prototype default: stratified group-aware train, validation, and test partitions at approximately 70/15/15. If the final dataset is too small for that ratio, stop and record an approved alternative rather than silently changing it.

## Feature matrix contract

The classifier input is a numeric matrix with:

- one row per image;
- one column per feature-schema field marked as model input;
- no sample ID, file path, label, OCR field, or semantic field;
- fixed column order;
- finite values after the training-fitted preprocessing pipeline;
- an explicit no-detection indicator when the default missing-region policy is used.

The feature matrix must be joined to labels using sample IDs after split creation, not by directory order.

## Preprocessing

Preprocessing is part of the model artifact. The default order is:

1. select the feature columns from the locked schema;
2. validate dtype and finite-value rules;
3. impute missing values using training rows only;
4. optionally add missingness indicators;
5. standardize continuous features for Logistic Regression and SVM using training parameters;
6. leave Random Forest inputs unscaled unless a documented experiment changes that behavior;
7. fit the estimator.

Any constant-column removal, feature selection, dimensionality reduction, or class weighting must be selected using training/development data only and recorded in the configuration.

## Candidate classifiers

The thesis names Logistic Regression, Support Vector Machine, and Random Forest as candidate traditional classifiers. Their final inclusion is subject to feasibility and validation.

### Logistic Regression

Prototype default:

- binary logistic regression;
- regularization enabled;
- training-only standardization;
- maximum iteration limit recorded;
- small C search selected on the validation split.

Interpretation: standardized coefficients describe the fitted model's association with the decision boundary. They are not causal effects.

### Support Vector Machine

Prototype default:

- compare a linear SVM and an RBF SVM only if the dataset size and validation design support both;
- standardize continuous inputs using training-only parameters;
- record kernel, C, gamma, and score-to-threshold rule;
- use the decision function as the model score unless calibrated probabilities are explicitly added.

### Random Forest

Prototype default:

- fixed random seed;
- a small, predeclared search over tree count, maximum depth, and minimum leaf size;
- no scale-dependent preprocessing requirement;
- feature importance retained as diagnostic information only.

Importance values must not be presented as causal explanations and must be interpreted with grouped ablations and correlated-feature warnings.

## Model selection

The default development procedure is:

1. fit each candidate and each predeclared feature-group configuration on the training split;
2. select settings using validation metrics and error inspection;
3. choose a decision threshold using validation scores when the default threshold is not adequate;
4. freeze the configuration;
5. refit the selected pipeline on training plus validation data only;
6. evaluate the frozen model once on the test split.

If repeated resampling or nested cross-validation is needed because of the final sample size, document that as a configuration change and keep a final untouched evaluation procedure.

## Required comparisons

At minimum, the evaluation runner should support:

- each enabled full feature representation with Logistic Regression, SVM, and Random Forest;
- the predeclared feature-group ablations in FEATURES.md;
- the same split manifest and sample IDs for every comparison;
- a comparison table containing accuracy, precision, recall, F1-score, and confusion-matrix counts.

The system must not select a model because it performs best on the test set. The test set is for the final comparison, not for feature engineering or tuning.

## Model package format

Each trained package is a directory such as artifacts/models/model-RUN_ID/ containing:

```text
model-RUN_ID/
  package_manifest.json
  estimator.joblib
  preprocessor.joblib
  feature_schema.json
  detector_config.json
  feature_config.json
  split_manifest_sha256.txt
  validation_metrics.json
  training_summary.json
```

package_manifest.json must contain:

- model ID and creation timestamp;
- label mapping;
- estimator family and hyperparameters;
- feature schema version and feature column order;
- preprocessor version;
- detector and feature configuration hashes;
- split manifest hash;
- training and validation sample counts;
- decision threshold and selection method;
- runtime and dependency versions;
- a warning that the package is limited to the documented study conditions.

## Prediction record

A prediction record must include:

```json
{
  "schema_version": "prediction-1",
  "input_sha256": "...",
  "model_id": "...",
  "label": "ai_generated",
  "display_label": "AI-generated",
  "score": 0.73,
  "threshold": 0.50,
  "region_count": 4,
  "no_detection": false,
  "feature_schema_version": "...",
  "warnings": [
    "Prototype result; not proof of authenticity."
  ]
}
```

The numeric score is a model score or calibrated probability as declared in the package. It must not be called a confidence in the underlying truth unless calibration has been evaluated and documented.

## Local explanation contract

The shared inference service may attach a versioned explanation record to a prediction. For Logistic Regression, feature contributions add in log-odds space; for a linear SVM, they add in decision-margin space. The implementation must verify that the contributions reconstruct the fitted model output within the declared numerical tolerance. Probability contributions are not additive.

Random Forest, RBF SVM, and any unsupported package must be marked `global_only`; the UI may show measured feature values and the compatible evaluation report, but must not invent local contributions. Explanations describe fitted model behavior, not causes, authenticity, or provenance.

## CLI targets

The implementation should expose commands equivalent to:

```text
python -m src.cli validate-data --manifest ...
python -m src.cli create-splits --manifest ...
python -m src.cli extract-features --config ...
python -m src.cli train --config ...
python -m src.cli evaluate --model-id ...
python -m src.cli predict --model-id ... --image ...
```

The exact command framework is a prototype default. Each command must write a machine-readable result and a human-readable summary.

## Failure handling

- Refuse to train when the manifest audit or leakage gate fails.
- Refuse to fit when the feature schema contains text, object columns, or label-derived fields.
- Refuse inference when the package schema does not match the current pipeline.
- Refuse a final test run when the test set was read during tuning.
- Mark unsupported feature groups explicitly.
- Never silently fall back to a different model, detector, split, or feature schema.

## Acceptance criteria

- All candidates receive the same image-level rows and split manifest.
- Preprocessing is fit only on the permitted development data and is serialized.
- The final test set is not used for model or threshold selection.
- A saved package can produce the same prediction for the same input and configuration.
- A prediction can be traced to the feature schema, detector configuration, split manifest, and run ID.
- The UI calls this pipeline rather than duplicating it.


