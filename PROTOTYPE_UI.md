# Prototype User Interface

## Purpose

The UI demonstrates the already evaluated pipeline for one image at a time. It is a thin presentation layer over the inference service described in ARCHITECTURE.md and ML_PIPELINE.md.

The UI must make the research boundary visible: it presents a model output under documented conditions, not proof of authenticity.

## Prototype scope

### Included

- load one compatible model package;
- upload one image;
- run the locked detector, crop, feature, aggregation, and model stages;
- display nonsemantic region overlays;
- display numeric processing status and warnings;
- display the bounded prediction label and model score;
- export a prediction record.

### Deferred

- user accounts and authentication;
- public hosting;
- document editing or receipt verification;
- OCR text display;
- generator attribution;
- batch review dashboards;
- production monitoring or scalable job queues.

## Recommended screen flow

### 1. Model and configuration screen

Show:

- model package ID;
- model family;
- feature schema version;
- detector name and version;
- model package creation time;
- compatible category and label contract;
- a warning that the model is limited to the study conditions.

The UI must refuse to run if the model package is incomplete or the current pipeline schema does not match the package.

### 2. Image input screen

Allow:

- selecting one local image;
- validating file format and decodability;
- displaying the input dimensions and checksum;
- optionally recording a user-supplied category confirmation.

The UI must not infer receipt provenance, label, or authenticity from the filename. If category confirmation is required, it must be an explicit user input or a model-package rule.

### 3. Processing screen

Show stage status:

1. image decoded;
2. text regions localized;
3. crops or masks prepared;
4. visual features extracted;
5. image-level record built;
6. model prediction returned.

Do not display recognized words, OCR strings, language, merchant names, amounts, or semantic explanations.

### 4. Result screen

Show:

- display label: AI-generated or Non-AI-generated under the dataset rules;
- model score or calibrated probability with its meaning stated;
- decision threshold;
- region count;
- no-detection status;
- feature schema and model package ID;
- quality flags and warnings;
- a region overlay on the original image;
- a “Why this label?” section when the selected model supports a faithful local explanation, with its contribution scale stated plainly;
- measured feature values and a global-only notice when additive local explanations are not supported;
- a short research-use disclaimer.

Do not show a sentence such as “this receipt is authentic” or “this image is definitely fake.”

The evaluation report summarizes the selected model's saved held-out run and does not change for each uploaded image. The “Why this label?” section is per-image and describes the selected fitted model's evidence for that prediction when a faithful local explanation is supported.

### 5. Artifact export

Allow download or saving of:

- prediction JSON;
- region overlay image;
- localization metadata without recognized text;
- the versioned local explanation when available, or an explicit global-only explanation record;
- the input checksum and model ID;
- warnings and quality flags.

## UI result contract

Prediction JSON must follow the model pipeline contract:

```json
{
  "schema_version": "prediction-1",
  "input_sha256": "...",
  "model_id": "...",
  "display_label": "AI-generated",
  "score": 0.73,
  "score_type": "decision_score",
  "threshold": 0.50,
  "region_count": 4,
  "no_detection": false,
  "warnings": [
    "Prototype result; not proof of authenticity."
  ]
}
```

The separate `prediction-explanation-2.json` export includes a traceability object with the input checksum, model/run IDs, feature-schema and aggregation versions, detector configuration hash, and feature-vector checksum. It contains visual feature measurements only.

## Error and warning states

| State | User-facing behavior |
|---|---|
| Unsupported format | Explain supported formats and do not start inference. |
| Decode failure | Explain that the image could not be read; do not guess. |
| Missing model package | Ask the user to load a compatible evaluated package. |
| Schema mismatch | Stop and identify the incompatible schema versions. |
| Detector unavailable | Stop and report configuration/dependency failure. |
| No detected regions | Continue only under the configured policy and show a prominent warning. |
| Too few or invalid regions | Return a processing warning or refusal according to the locked quality policy. |
| Input category unknown | Mark category as unknown; do not claim a thesis-valid interpretation. |
| Prediction completed | Show result and disclaimer; do not imply certainty. |

## Accessibility and clarity

- Use plain language and explain “non-AI-generated under the dataset rules” once.
- Make warnings visually distinct without relying only on color.
- Provide readable contrast and keyboard-accessible controls.
- Show progress and disable duplicate submissions while processing.
- Avoid technical error dumps in the main result; provide an expandable diagnostic record.
- Never use the recognized receipt text as an explanation or accessibility label.

## UI acceptance criteria

- A user can load a valid model package and process one valid image.
- The UI returns the same prediction as the command-line inference path for the same input and package.
- A region overlay is shown without OCR text.
- No-detection and model/schema failures are understandable and safe.
- The result includes model ID, input checksum, threshold, region count, and warnings.
- Supported local explanations reconstruct the fitted model score and state their scale; unsupported models are labeled global-only.
- The evaluation view distinguishes primary feature-group comparisons from supporting per-feature diagnostics and labels post-hoc pilot results.
- The UI visibly states that the prediction is a research prototype result and not proof of authenticity.
- No code in the UI performs independent feature extraction or model loading outside the shared inference service.
