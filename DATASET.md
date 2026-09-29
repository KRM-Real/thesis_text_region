# Dataset Contract and Onboarding

## Purpose

This document defines how a future dataset will be added to the prototype. It does not select a new dataset, invent sample counts, or treat the synced `sources/dataset.zip` as runtime data. The dataset will be supplied after the prototype is implemented and tested.

## Thesis-fixed dataset scope

- The study is binary: `non_ai_generated` versus `ai_generated`.
- Receipts are the controlled text-containing image category.
- `non_ai_generated` means the image is labeled as such under the final dataset rules. It does not mean universal authenticity, unedited origin, or legal validity.
- `ai_generated` means the image is labeled as AI-generated under the final provenance rules.
- The exact source, receipt type, number of samples, generators, acquisition conditions, and inclusion criteria remain to be finalized after dataset research and pilot testing.

## Explicit non-use rule

The archive in `sources/dataset.zip` is reference material only and must not be used by the implementation. Do not:

- import it from code;
- copy it into `data/`;
- use its filenames, counts, labels, or metadata as defaults;
- train or evaluate on it;
- use it to create a hidden fixture or baseline;
- document its contents as the thesis dataset.

The prototype must run with the dataset directories empty, using tests and small generated fixtures only.

## Canonical future layout

```text
data/
├── incoming/
│   ├── non_ai_generated/
│   │   └── <user-supplied images>
│   └── ai_generated/
│       └── <user-supplied images>
├── manifests/
│   ├── source_manifest.csv
│   ├── audited_manifest.csv
│   └── split_manifest.csv
└── fixtures/
    └── <test-only generated images>
```

`data/incoming/` is a staging area. The manifest is the source of truth for training; directory names alone are not sufficient provenance.

## Required manifest fields

The canonical manifest is CSV or Parquet with one row per image.

| Field | Required | Meaning |
|---|---:|---|
| `sample_id` | yes | Stable, unique identifier that is not derived from recognized text. |
| `relative_path` | yes | Path relative to the project data root. Absolute paths are rejected for portable runs. |
| `label` | yes | `non_ai_generated` or `ai_generated`; `real` may be normalized once and logged. |
| `category` | yes | Must be `receipt` for thesis-valid evaluation. |
| `source_id` | yes | Source collection or acquisition group identifier. |
| `group_id` | yes | Group for split isolation, such as original image, template, or known related set. |
| `provenance_type` | yes | Description of how the file entered the dataset, without recognized text. |
| `generator_id` | conditional | Required for AI-generated items when known; use an explicit unknown value when permitted by the final data rules. |
| `acquisition_type` | yes | Example: scan, camera capture, screenshot, or digital copy. Must reflect the final data rules. |
| `width` | yes | Decoded image width. |
| `height` | yes | Decoded image height. |
| `format` | yes | Decoded image format. |
| `sha256` | yes | Hash of the exact file bytes used by the run. |
| `split` | generated | `train`, `validation`, or `test`; generated only after audit and grouping. |
| `inclusion_status` | generated | `included`, `excluded`, or `review`. |
| `exclusion_reason` | generated | Required when the status is not `included`. |

Recognized words, OCR strings, language, topic, merchant name, receipt amount, and other semantics must not be added as features or split keys. If such information exists in an external source record, keep it outside the model manifest and do not pass it to the pipeline.

## Onboarding workflow

1. **Register files.** Copy or link only the user-supplied dataset into `data/incoming/` and create `source_manifest.csv`.
2. **Decode audit.** Verify image readability, dimensions, format, color mode, and file checksum.
3. **Label audit.** Verify the binary label and record the evidence used for each label.
4. **Category audit.** Confirm that each included image is a receipt under the approved inclusion definition.
5. **Provenance audit.** Record source, acquisition condition, generator information, and related-image groups.
6. **Duplicate audit.** Detect exact duplicates by checksum and review near-duplicates or transformed copies.
7. **Shortcut audit.** Compare format, dimensions, compression, resolution, source, background, layout, and acquisition variables across labels.
8. **Eligibility decision.** Mark each row `included`, `excluded`, or `review`; do not silently discard rows.
9. **Split creation.** Create a deterministic group-aware split manifest. Split before fitting imputation, scaling, feature selection, or model settings.
10. **Lock the manifest.** Hash the audited and split manifests and include both hashes in the run record.

## Validation rules

Training must fail if:

- an included file is missing or its checksum does not match;
- a label is outside the two canonical classes;
- an included category is not `receipt`;
- a sample has no stable `sample_id` or `group_id`;
- a duplicate or related group crosses partitions;
- a required provenance field is missing;
- a class has no samples in a required split;
- class balance or sample counts are too small for the configured experiment and no approved fallback is recorded.

Validation may produce a review report, but it must not repair labels, infer provenance, or assign a receipt category without a human-approved rule.

## Split and leakage rules

- The image is the atomic observation.
- All regions, crops, masks, augmentations, and transformed copies derived from one source image remain in the same split.
- Known template families, acquisition batches, source groups, and near-duplicate groups remain together when they could reveal the label.
- The test split is created before model tuning and is not used to select detector settings, features, hyperparameters, or thresholds.
- If a relation is uncertain, keep the samples together or mark them for review rather than assuming independence.

## Prototype defaults

| Item | Default |
|---|---|
| Data root | `data/` |
| Manifest | `data/manifests/audited_manifest.csv` |
| Image formats | PNG, JPEG, and other formats explicitly supported by the decoder; the run records the actual set. |
| Label names | `non_ai_generated`, `ai_generated` |
| Category | `receipt` |
| Split | Stratified group-aware split, initially 70/15/15 when feasible. |
| Unknown metadata | Keep an explicit `unknown` value and a warning; never infer it from a filename. |
| Missing text regions | Retain the sample with a no-detection indicator unless the final approved data rule excludes it. |

These defaults are scaffolding for implementation. The final thesis dataset rules must be recorded as a versioned configuration before final evaluation.

## Required onboarding artifacts

- `source_manifest.csv`
- `audited_manifest.csv`
- `split_manifest.csv`
- `dataset_audit.json`
- `duplicate_review.csv`
- `shortcut_audit.json`
- `manifest_sha256.txt`
- a short human-readable dataset decision note

## Acceptance criteria

- The prototype starts with no user dataset and still passes all fixture tests.
- Adding the future dataset requires changing only data files and configuration, not source code paths.
- Every included image is traceable to a checksum, provenance record, category decision, and split.
- No OCR output, recognized word, or semantic metadata enters the feature matrix.
- The final evaluation cannot start until the audit and leakage gates pass.



