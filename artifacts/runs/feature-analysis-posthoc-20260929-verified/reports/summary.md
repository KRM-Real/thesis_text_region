# Pilot feature-matrix evaluation report

Run ID: `20260928T180733Z-fc8296f7-178f5179-evaluate`

This run reused per-image checksums from a prior evaluation report after validating the same model, feature schema, split, test sample IDs, labels, thresholds, and prediction scores. No source images were reopened; the reference report and prediction artifact hashes are recorded in run_manifest.json.

Positive class: `ai_generated`. Threshold rule: `score >= threshold => ai_generated`.

Metrics: `{"accuracy": 1.0, "confusion_matrix": {"fn": 0, "fp": 0, "tn": 128, "tp": 128}, "f1": 1.0, "positive_class": "ai_generated", "precision": 1.0, "recall": 1.0, "sample_count": 256, "threshold": 0.9213612857537781, "threshold_convention": "score >= threshold => ai_generated", "zero_division": 0}`

The image is the unit of analysis; all regions remain inside the image split.

## Feature interpretation

Interpretation status: `posthoc_pilot`. Feature-group comparison is the primary evidence; individual-feature rankings are supporting diagnostics.

Permutation importance describes how this frozen model's held-out F1 changed when one feature column was shuffled. It is descriptive and was not used to fit, tune, or select the model.

### Feature groups

- Geometry and density: test F1 0.9509; difference from all enabled -0.0491.
- Geometry plus spacing and alignment: test F1 0.9769; difference from all enabled -0.0231.
- Stroke and shape: test F1 0.9883; difference from all enabled -0.0117.
- Edges, gradients, contrast, and sharpness: test F1 0.9884; difference from all enabled -0.0116.
- Texture and local pixel relationships: test F1 0.9612; difference from all enabled -0.0388.
- All enabled groups (reference): test F1 1.0000; difference from all enabled +0.0000.

### Individual-feature diagnostic

The largest measured permutation F1 decrease was for `geometry.center_y.max` (geometry_density), at 0.0008 on this test split. This ranking can be affected by correlated features and the selected model.

### Limits

- Feature-group ablation is the primary feature-usefulness comparison for the thesis objectives.
- Individual permutation results and class-separation summaries are descriptive for this held-out split and were not used to fit, tune, or select the model.
- Linear coefficients and local contributions describe fitted model associations; they are not causal effects or proof of image origin.
- Correlated features can share or obscure importance; redundancy pairs use development rows only.
- Quality and no-detection indicators describe processing conditions and are not authenticity evidence.
- This report is limited to the documented receipt dataset, model, configuration, and split.
