# Testing Strategy

## Testing purpose

Testing must show that the prototype implements the documented pipeline correctly and safely. It must not be used to claim scientific performance before the future thesis dataset is supplied and audited.

The test suite uses generated fixtures and deterministic detector outputs. It must not read or depend on files under sources/.

## Test layers

### Unit tests

Unit tests isolate one function or contract:

- label normalization;
- path and checksum validation;
- image decoding;
- geometry clipping and normalization;
- polygon area and overlap calculations;
- deterministic region IDs;
- crop padding;
- each feature formula;
- aggregation for zero, one, and many regions;
- missing-value and no-detection behavior;
- model input-column selection;
- model package serialization;
- prediction label mapping.

### Contract tests

Contract tests verify that every detector adapter:

- returns valid geometry;
- discards recognized text fields;
- produces deterministic output for a fixed fixture;
- reports status and warnings;
- respects the detector configuration hash.

Feature extractors must return numeric values, schema names, quality flags, and no semantic fields.

### Integration tests

Integration tests run:

```text
fixture image -> fixture detector -> crops -> features -> aggregation -> preprocessing -> model -> prediction record
```

They must verify one output row per image and a stable model package.

### Leakage tests

Test cases must fail when:

- the same checksum appears in multiple splits;
- related group IDs cross splits;
- derived crops are given independent split assignments;
- a feature column contains a label, path, OCR value, or semantic field;
- a scaler or imputer is fit using validation or test rows;
- test rows are loaded by tuning code;
- a changed configuration reuses an old feature or model artifact.

### Reproducibility tests

For a fixed fixture set, seed, configuration, and model package:

- feature schema order is unchanged;
- feature values are stable within the declared numerical tolerance;
- model package hashes are stable where the serializer permits it;
- predictions are identical or within a declared tolerance;
- the same input checksum maps to the same prediction record.
- feature-analysis artifacts are deterministic for a fixed split, model, and analysis seed;
- redundancy diagnostics use training and validation rows only;
- supported linear explanations reconstruct the model output, while nonlinear or unsupported models are labeled global-only.

### UI smoke tests

The UI smoke path must verify:

- the app starts without the future dataset;
- a valid model fixture can be loaded;
- a valid fixture image returns a result;
- the overlay contains geometry but no recognized text;
- invalid input shows a safe error;
- no-detection shows a warning;
- schema mismatch prevents inference;
- the UI result matches the shared inference service.
- the evaluation view shows the feature-analysis status and interpretation limits;
- local explanation JSON is downloadable, and unsupported estimators do not receive invented additive attributions.

## Generated fixtures

The test fixture generator should create small images with simple geometric text-like marks. It may create:

- blank images;
- one horizontal region;
- several separated regions;
- overlapping regions;
- rotated or polygon regions when supported;
- different image sizes and color modes;
- intentionally invalid detector geometry.

Fixture labels are test labels only. They are not receipts, are not thesis data, and must never be included in reports as experimental results.

## Test case matrix

| ID | Scenario | Expected result |
|---|---|---|
| T-001 | Valid manifest with two canonical labels | Validation succeeds. |
| T-002 | Input label uses real alias | Normalized once and logged. |
| T-003 | Unknown label | Validation fails with field error. |
| T-004 | Missing image file | Row is invalid and training is blocked. |
| T-005 | Bad checksum | Validation fails. |
| T-006 | Non-receipt category in thesis-valid manifest | Row is review or excluded; training cannot silently include it. |
| T-007 | Exact duplicate across splits | Leakage gate fails. |
| T-008 | Two crops from one image in different splits | Leakage gate fails. |
| T-009 | Detector returns recognized text | Text is discarded and absent from all artifacts. |
| T-010 | Detector returns empty list | Explicit no-detection record is created. |
| T-011 | Detector returns invalid box | Invalid geometry is reported; no fabricated region is created. |
| T-012 | Region lies outside image | Geometry is clipped deterministically and flagged. |
| T-013 | Zero-height box | Region is rejected. |
| T-014 | Feature has undefined value | Missing value and quality flag are emitted. |
| T-015 | Aggregator receives many regions | Exactly one fixed-length image record is returned. |
| T-016 | Training-only scaler check | Validation/test statistics do not affect fitted parameters. |
| T-017 | Feature schema drift | Model loading is refused. |
| T-018 | Model artifact missing | Inference fails safely. |
| T-019 | Same fixture run twice | Prediction and traceability fields are stable. |
| T-020 | UI invalid upload | Clear error; no prediction. |
| T-021 | UI no-detection input | Result is allowed only under policy and includes warning. |
| T-022 | UI result versus CLI result | Labels and scores agree within tolerance. |

## Numerical tolerances

The test suite must distinguish exact identity from approximate numerical equivalence:

- geometry normalization: tolerance recorded in the test configuration;
- floating-point feature values: tolerance appropriate to the operation;
- model score: tolerance documented per estimator and serializer;
- labels and status fields: exact match required.

Do not weaken a test tolerance simply to make a failing implementation pass. Record the reason for any change.

## Test commands

The implementation should support commands equivalent to:

```text
pytest
pytest tests/unit
pytest tests/integration
pytest tests/smoke
```

The exact runner is a prototype default. Continuous integration should run unit, contract, integration, and leakage tests without any user dataset or network access.

## Definition of done

Implementation work is test-complete when:

- all required unit and contract tests pass;
- integration tests cover the full fixture pipeline;
- leakage tests fail on intentionally invalid examples;
- deterministic inference passes;
- deterministic feature analysis and supported explanation reconstruction pass;
- UI smoke tests pass;
- coverage includes every failure mode listed in the architecture documents;
- test output identifies the configuration and fixture version;
- no test imports or scans the sources archive.

## Acceptance criteria

- The test suite can run from a clean checkout with empty user-data directories.
- Tests prove that the image, not the crop, is the unit of modeling.
- Tests prove that OCR text and semantics are excluded.
- Tests prove that final model artifacts cannot be used with an incompatible schema.
- Test fixtures cannot be mistaken for thesis results.


