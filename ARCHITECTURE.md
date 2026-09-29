# Prototype Architecture

## Architectural goal

The architecture separates data preparation, text-region localization, visual feature extraction, image-level modeling, evaluation, and the demonstration UI. Each stage has a stable input/output contract so a stage can be inspected or replaced without changing the thesis scope.

```text
                     +----------------------+
future dataset ----> | manifest and audit   |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | group-aware splits   |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | text-region adapter  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | crops and geometry  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | visual features     |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | region aggregation  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     | model pipelines     |
                     | LR / SVM / RF        |
                     +----------+-----------+
                                |
              +-----------------+-----------------+
              v                                   v
       evaluation artifacts                 local prototype UI
```

## Repository structure

The following structure is the implementation target. Empty data directories may contain `.gitkeep` files, but user data and model weights must remain outside version control unless explicitly approved.

```text
.
├── README.md
├── AGENTS.md
├── PRD.md
├── ARCHITECTURE.md
├── DATASET.md
├── TEXT_REGION_PIPELINE.md
├── FEATURES.md
├── ML_PIPELINE.md
├── EVALUATION.md
├── PROTOTYPE_UI.md
├── TESTING.md
├── IMPLEMENTATION_PLAN.md
├── REPRODUCIBILITY.md
├── pyproject.toml
├── configs/
│   ├── prototype.yaml
│   └── feature_schema.json
├── data/
│   ├── incoming/
│   │   ├── non_ai_generated/
│   │   └── ai_generated/
│   ├── manifests/
│   └── fixtures/
├── artifacts/
│   ├── localization/
│   ├── features/
│   ├── splits/
│   ├── models/
│   └── reports/
├── src/
│   ├── cli.py
│   ├── config.py
│   ├── contracts.py
│   ├── data/
│   ├── text_regions/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── inference/
├── ui/
└── tests/
    ├── unit/
    ├── integration/
    ├── fixtures/
    └── smoke/
```

The supplied `sources/` directory is reference-only. It is not part of this runtime tree and must not be used as a default data location.

## Module boundaries

### `src/contracts.py`

Defines the types shared by modules. The central records are:

```python
class ImageRecord:
    sample_id: str
    path: str
    label: str | None
    category: str
    group_id: str
    checksum_sha256: str

class TextRegion:
    region_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None
    polygon: list[tuple[float, float]] | None

class ImageFeatureRecord:
    sample_id: str
    label: str | None
    region_count: int
    no_detection: bool
    values: dict[str, float]
    quality_flags: list[str]
```

Recognized text is not a field in `TextRegion` or `ImageFeatureRecord`. A detector adapter may receive it internally from a third-party engine, but it must discard it before returning the contract object.

### `src/data/`

Responsibilities:

- load and normalize the user-provided manifest;
- validate labels, categories, paths, dimensions, formats, and checksums;
- assign or validate group identifiers;
- identify exact and near-duplicate risks;
- create a saved split manifest before model fitting.

This module must not perform feature extraction or model fitting.

### `src/text_regions/`

Responsibilities:

- provide a detector interface;
- adapt one selected backend to `TextRegion` objects;
- apply fixed preprocessing and thresholds;
- save geometry and quality metadata;
- handle no-detection and malformed-output cases.

The backend is replaceable. Detector selection is a feasibility decision, not a change to the research question.

### `src/features/`

Responsibilities:

- crop or mask localized regions;
- compute only the visual measurements in [FEATURES.md](FEATURES.md);
- aggregate region values into one fixed-length image record;
- emit a versioned feature schema and feature availability report.

No feature function may accept decoded OCR text, language, or semantic embeddings.

### `src/models/`

Responsibilities:

- build the preprocessing and estimator pipeline;
- fit candidate models on image-level records;
- select settings using training/development data only;
- save and load model packages;
- return bounded prediction records.

### `src/evaluation/`

Responsibilities:

- calculate required metrics;
- produce confusion matrices and per-image predictions;
- run feature-group comparisons;
- produce supporting individual-feature diagnostics for a frozen model, with status and split provenance;
- check leakage and data-quality gates;
- create reports without modifying the test set.

### `src/inference/`

Responsibilities:

- load a complete model package;
- run the same detector, crop, feature, aggregation, and preprocessing steps used during training;
- return a prediction plus traceability metadata;
- attach a local explanation only when the fitted model supports a verifiable additive explanation; otherwise return an explicit global-only status and measured feature values;
- never retrain or alter artifacts during inference.

The shared inference result may include a versioned `PredictionExplanation`. Logistic Regression contributions are expressed in log-odds and linear SVM contributions in decision-margin units. Neither scale is presented as a causal account or proof of image origin.

### `ui/`

The UI is a thin client over `src/inference/`. It must not reimplement image processing or model logic. See [PROTOTYPE_UI.md](PROTOTYPE_UI.md).

## Interface contracts

### Detector interface

```python
class TextRegionDetector(Protocol):
    name: str
    version: str

    def detect(
        self,
        image: ImageArray,
        config: DetectorConfig,
    ) -> DetectionResult:
        """Return geometry and quality metadata only; no recognized text."""
```

`DetectionResult` includes `regions`, `image_width`, `image_height`, `detector_config_hash`, `elapsed_ms`, `status`, and `warnings`.

### Feature interface

```python
class RegionFeatureExtractor(Protocol):
    group_name: str
    schema_version: str

    def extract(
        self,
        image: ImageArray,
        regions: list[TextRegion],
        config: FeatureConfig,
    ) -> RegionFeatureResult:
        """Return numeric visual values and quality flags only."""
```

### Aggregation interface

```python
class ImageAggregator(Protocol):
    schema_version: str

    def aggregate(
        self,
        sample_id: str,
        region_results: list[RegionFeatureResult],
        image_shape: tuple[int, int],
    ) -> ImageFeatureRecord:
        """Return exactly one fixed-length record for one image."""
```

### Predictor interface

```python
class ImagePredictor(Protocol):
    model_id: str

    def predict(self, image: ImageArray) -> PredictionRecord:
        """Run the locked inference pipeline and return one image-level result."""
```

## Configuration and immutability

Every run loads one configuration file and records its SHA-256 checksum. Configuration contains:

- dataset manifest path;
- split seed and split policy;
- detector backend, version, weights identifier, threshold, and preprocessing;
- crop padding and image normalization;
- enabled feature groups and schema version;
- aggregation statistics;
- model candidates and hyperparameter search space;
- decision threshold policy;
- output directory and run identifier.

After a final test run begins, its configuration and split manifest are immutable. A changed configuration creates a new run identifier.

## Failure boundaries

| Failure | Owning module | Required behavior |
|---|---|---|
| Missing or unreadable image | data | Mark invalid and stop training unless an explicit exclusion report is accepted. |
| Invalid label/category | data | Reject from thesis-valid manifest and explain the field error. |
| Duplicate or related sample across splits | data/evaluation | Fail the leakage gate. |
| Detector returns malformed geometry | text_regions | Reject that image record or mark it failed; never fabricate a region. |
| No detected regions | text_regions/features | Keep an explicit no-detection record under the configured policy. |
| Feature calculation error | features | Record feature-level failure and stop if required features are missing. |
| Missing model artifact | inference | Show a configuration error; do not train in the UI. |
| Prediction outside trained schema | inference | Refuse prediction and report schema mismatch. |

## Traceability rule

The model artifact, prediction record, and UI must be able to answer:

1. which input checksum was processed;
2. which detector and detector configuration were used;
3. how many regions were returned;
4. which feature schema and aggregation version were used; and
5. which preprocessing and model artifact produced the result.


