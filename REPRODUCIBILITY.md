# Reproducibility and Artifact Protocol

## Purpose

This protocol makes each experiment and prototype prediction traceable. It is intentionally separate from the thesis dataset: no user data is required for the implementation and no files under sources/ are runtime inputs.

## Run identity

Every command that writes artifacts creates a run ID derived from:

- UTC creation time;
- short configuration hash;
- source revision identifier;
- optional human-readable label.

The run ID must be written into every report and artifact manifest. A changed detector, feature schema, split, model setting, or dataset manifest creates a new run ID.

## Required run manifest

run_manifest.json must record:

- run ID;
- source revision;
- configuration file path and SHA-256;
- audited manifest path and SHA-256;
- split manifest path and SHA-256;
- detector name, version, weights identifier, and configuration hash;
- crop and preprocessing settings;
- feature schema version and hash;
- aggregation version;
- candidate model settings;
- selected model ID;
- threshold and threshold-selection method;
- random seeds;
- training, validation, and test sample counts;
- software and dependency versions;
- operating-system and hardware context when relevant;
- start and completion times;
- status and failure reason, if any.

## Determinism

Set and record seeds for:

- split creation;
- model training;
- hyperparameter search;
- fixture generation;
- any stochastic detector or preprocessing component.

Determinism means the same declared inputs, environment, seed, and configuration produce the same feature schema and the same prediction within documented numerical tolerance. If a dependency or hardware path is nondeterministic, record that limitation rather than claiming exact repeatability.

## Data and artifact hashes

Hash:

- exact input image bytes;
- source, audited, and split manifests;
- detector weights when available;
- configuration files;
- feature schema;
- model package files;
- exported prediction records.

Store hashes in machine-readable files and include them in the human-readable report.

## Versioned artifact layout

```text
artifacts/
  runs/<run-id>/
    run_manifest.json
    config_snapshot.yaml
    dataset/
      audit.json
      manifest_sha256.txt
      split_manifest.csv
    localization/
      per_image/
      localization_summary.json
    features/
      image_features.parquet
      feature_schema.json
      feature_quality.json
    models/
      <model-id>/
    reports/
      summary.md
      model_comparison.csv
      validation_metrics.json
      test_metrics.json
      test_predictions.csv
      confusion_matrix.csv
      feature_group_ablation.csv
      feature_analysis.json
      feature_influence.csv
      feature_group_summary.csv
      feature_redundancy.csv
    predictions/
      <prediction-id>.json
      prediction-explanation-2.json
```

Generated artifacts may be large. The repository should keep only approved small metadata and fixture artifacts; ignore future user images, detector weights, and large model files by default.

## Configuration snapshots

Never report results without saving the exact configuration used. A configuration snapshot must include:

- dataset manifest location;
- split policy and seed;
- detector settings;
- image preprocessing;
- crop/mask policy;
- feature groups and formulas;
- aggregation statistics;
- model candidates and hyperparameters;
- threshold policy;
- output paths.

## Reproduction procedure

To reproduce a final run:

1. check out the recorded source revision;
2. install the recorded environment;
3. verify all model weights and user data hashes;
4. load the exact configuration snapshot;
5. verify the audited and split manifest hashes;
6. rerun feature extraction only if the hashes and schema match;
7. load the saved model package;
8. compare predictions and metrics within declared tolerances;
9. record any environment or dependency differences.

## Prediction traceability

Every prototype prediction must be attributable to:

- input image checksum;
- model ID;
- model package checksum;
- detector configuration hash;
- feature schema hash;
- aggregation version;
- preprocessing parameters;
- threshold;
- warning list.

Feature-analysis runs also record the analysis status and settings, hashes of the feature-analysis JSON/CSV artifacts, source feature-matrix/schema hashes, and split hash. If image checksums are reused from a compatible prior evaluation instead of reopening source images, record the reference run ID and prediction-artifact hash and validate that its model, schema, split, threshold, sample IDs, labels, and scores match.

The UI must expose these identifiers through an export or diagnostic view.

## Dataset privacy and handling

- Do not commit the future dataset to the repository unless explicitly approved.
- Do not put recognized receipt text in manifests, logs, or prediction artifacts.
- Avoid logging raw image paths outside the project data root.
- Keep user data in configured directories and validate path traversal.
- Do not use the synced sources archive as a hidden fallback.

## Human-readable report template

Each final report should state:

1. research question and bounded scope;
2. dataset inclusion and provenance rules;
3. sample counts and class distribution;
4. split and leakage controls;
5. detector and localization configuration;
6. feature groups and aggregation;
7. candidate models and selection rule;
8. required metrics and confusion matrix;
9. feature-group comparison;
10. failure cases and limitations;
11. exact artifact identifiers and hashes;
12. claims that are intentionally not made.

## Acceptance criteria

- A run can be reconstructed from its manifest, configuration, model package, and source revision.
- All generated results include hashes and a run ID.
- Repeated fixture runs are deterministic within declared tolerances.
- The future dataset can be added without changing the reproducibility protocol.
- No source archive is used as a hidden input.

