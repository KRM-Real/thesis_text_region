# Visual Text Region Features

## Purpose

This document defines the visual feature representation used after text-like regions have been localized. The features describe pixels and geometry. They do not describe what the receipt says.

The exact equations and enabled groups remain subject to detector and pilot feasibility. The definitions below are implementation-ready prototype defaults and must be versioned before thesis evaluation.

## Feature rules that cannot be violated

1. The feature extractor receives image pixels and region geometry only.
2. It never receives recognized words, spelling, language, topic, merchant, amount, date, arithmetic, or semantic embeddings.
3. Region-level values are intermediate evidence. One image produces one fixed-length record.
4. Detector confidence is stored as quality metadata and is excluded from the classifier feature vector by default.
5. Every feature name, formula version, missing-value rule, and source stage is recorded in the feature schema.
6. Preprocessing parameters estimated from data are fitted on training data only.

## Region input

Each region has:

- original image dimensions;
- validated box or polygon geometry;
- an optional padded crop;
- an optional polygon mask;
- a stable region ID;
- nonsemantic quality flags.

Geometry is normalized to the original image width and height. Pixel measurements are computed from the configured crop or mask representation and not from a differently processed copy for each class.

## Candidate feature groups

### Geometry and text density

These features describe the amount and arrangement of detected regions, not their contents.

| Feature | Prototype definition | Level |
|---|---|---|
| geometry.area_ratio | Region area divided by full image area. | Region |
| geometry.width_ratio | Region width divided by image width. | Region |
| geometry.height_ratio | Region height divided by image height. | Region |
| geometry.aspect_ratio | Width divided by height after a minimum-height guard. | Region |
| geometry.center_x | Region center x coordinate normalized to image width. | Region |
| geometry.center_y | Region center y coordinate normalized to image height. | Region |
| geometry.orientation | Geometry-only orientation when supplied by the detector; otherwise missing. | Region |
| density.region_count | Number of valid regions. | Image |
| density.union_coverage | Union of region areas divided by image area, preventing double counting of overlaps. | Image |
| density.area_sum_ratio | Sum of region areas divided by image area, retained separately from union coverage. | Image |

### Spacing and alignment

These features describe the spatial organization of region geometry without identifying characters or words.

| Feature | Prototype definition | Level |
|---|---|---|
| layout.nearest_horizontal_gap | Nearest horizontal gap between non-overlapping region boxes, normalized by image width. | Region then image |
| layout.nearest_vertical_gap | Nearest vertical gap between non-overlapping region boxes, normalized by image height. | Region then image |
| layout.centroid_dispersion_x | Standard deviation of normalized region centers along x. | Image |
| layout.centroid_dispersion_y | Standard deviation of normalized region centers along y. | Image |
| layout.row_alignment_error | Deviation from geometry-only horizontal row grouping using a fixed tolerance. | Image |
| layout.overlap_ratio | Pairwise intersection-over-union summary for region boxes. | Image |

Pairwise calculations must use a deterministic order and a documented empty-pair policy.

### Stroke appearance

Stroke features are candidate measurements of visible line-width behavior. If a selected stroke estimator is not stable for the chosen detector output, the feature group must be marked unavailable rather than approximated silently.

| Feature | Prototype definition | Level |
|---|---|---|
| stroke.width_median | Median valid stroke-width estimate inside the crop or mask. | Region |
| stroke.width_std | Standard deviation of valid stroke-width estimates. | Region |
| stroke.width_iqr | Interquartile range of valid stroke-width estimates. | Region |
| stroke.valid_fraction | Valid stroke-response pixels divided by crop pixels. | Region |
| stroke.width_p90 | Ninetieth percentile of valid stroke widths. | Region |

These are not class rules. A higher or lower value must not be described as intrinsically AI-generated.

### Shape and connected-component structure

Shape features use pixel-level connected components or contours inside the localized region. They are geometry descriptors, not OCR character identities.

| Feature | Prototype definition | Level |
|---|---|---|
| shape.component_count | Count of valid foreground-like connected components under a fixed binarization rule. | Region |
| shape.component_area_median | Median component area normalized by crop area. | Region |
| shape.component_aspect_median | Median component width-to-height ratio with guards. | Region |
| shape.compactness_mean | Per-component perimeter-squared divided by area, summarized across valid components. | Region |
| shape.solidity_mean | Component area divided by convex-hull area, summarized across valid components. | Region |
| shape.hole_count | Number of enclosed holes under the fixed contour rule. | Region |

The binarization rule must be independent of the class label and recorded in the configuration.

### Edges and gradients

These features describe local intensity transitions in the region pixels.

| Feature | Prototype definition | Level |
|---|---|---|
| edge.canny_density | Edge pixels divided by valid crop pixels under a fixed Canny configuration. | Region |
| gradient.magnitude_mean | Mean Sobel gradient magnitude. | Region |
| gradient.magnitude_std | Standard deviation of Sobel gradient magnitude. | Region |
| gradient.magnitude_p90 | Ninetieth percentile of gradient magnitude. | Region |
| gradient.orientation_entropy | Entropy of quantized gradient orientations, excluding near-zero magnitude pixels. | Region |
| edge.boundary_continuity | Continuity summary for connected edge components touching detected region boundaries. | Region |

### Contrast and sharpness

These features measure local visual quality and are sensitive to acquisition conditions. They must be interpreted together with the dataset audit.

| Feature | Prototype definition | Level |
|---|---|---|
| contrast.gray_std | Standard deviation of grayscale intensity in the region representation. | Region |
| contrast.local_ring_delta | Difference between the crop intensity center and a padded background ring when a ring is available. | Region |
| contrast.p10_p90_range | P90 minus P10 grayscale intensity. | Region |
| sharpness.laplacian_variance | Variance of the Laplacian response. | Region |
| sharpness.tenengrad_mean | Mean squared or absolute gradient energy under the fixed Tenengrad definition. | Region |

The crop padding and ring definition must be constant across the dataset.

### Texture and local pixel relationships

These features summarize repeated local intensity patterns without interpreting the image content.

| Feature | Prototype definition | Level |
|---|---|---|
| texture.gray_entropy | Entropy of a fixed grayscale histogram. | Region |
| texture.lbp_uniform_fraction | Fraction of local binary pattern codes classified as uniform. | Region |
| texture.lbp_entropy | Entropy of the LBP histogram. | Region |
| texture.glcm_contrast | GLCM contrast under fixed distance, angle, levels, and normalization. | Region |
| texture.glcm_homogeneity | GLCM homogeneity under the same fixed settings. | Region |

Texture settings must be recorded because quantization, distance, and angle can materially change the values.

## Region-to-image aggregation

Traditional classifiers require one fixed-length row per image. For every numeric region-level feature, the prototype default aggregator computes:

```text
mean, median, standard deviation, IQR, minimum, maximum, p10, p90
```

The aggregator also emits image-level geometry fields such as region count, union coverage, area-sum ratio, no-detection indicator, and quality counts. The final feature schema fixes the order and names.

The default aggregation is a recommendation for the prototype. It is not evidence that these statistics are optimal. The final run records any reduction or expansion and explains why it was chosen.

Example names:

```text
contrast.gray_std.mean
contrast.gray_std.median
contrast.gray_std.iqr
geometry.area_ratio.p90
density.region_count
density.union_coverage
quality.no_detection
```

## Missing and invalid values

- If a feature is mathematically undefined for a valid region, emit a missing value and a feature-quality flag.
- If a feature group is unavailable because the detector output cannot support it, mark the group unavailable in the schema; do not substitute a different formula under the same name.
- If an image has no valid regions, emit the fixed schema with missing region-derived values and quality.no_detection = 1 under the default policy.
- The imputer, if used, is fitted on training rows only and is persisted in the model package.
- Missingness indicators are allowed as technical quality indicators; they must not encode hidden label information.

## Feature groups for controlled comparison

The evaluation should compare the complete representation with predeclared subsets:

1. geometry and density only;
2. geometry plus spacing and alignment;
3. stroke and shape;
4. edge, gradient, contrast, and sharpness;
5. texture and local pixel relationships;
6. all enabled groups.

The exact subset list is a prototype default. Any unavailable group is reported as unavailable rather than treated as a zero-information result.

## Feature schema artifact

feature_schema.json must contain, for every column:

- name;
- group;
- level before aggregation;
- formula ID and schema version;
- source crop or geometry fields;
- dtype and expected range when known;
- missing-value policy;
- whether the column is used by the classifier;
- whether it is diagnostic-only.

The feature matrix must include sample_id and label only as keys or metadata. They are never passed as numeric model inputs.

## Quality checks

Before model fitting, report:

- feature count by group;
- missingness by split and label;
- constant or near-constant columns;
- non-finite values;
- feature ranges and obvious unit errors;
- pairwise correlation or redundancy diagnostics for interpretation;
- whether any feature column contains text or object values.

Do not use the test set to remove features or choose a feature subset.

## Acceptance criteria

- Every image yields one fixed-length row or an explicit processing failure record.
- No recognized text or semantic value appears in the feature schema or model matrix.
- The same formulas and crop policy run for both classes.
- Feature names and ordering are stable across training and inference.
- Aggregation prevents region-level rows from becoming independent samples.
- Feature-group ablations can be rerun from configuration without changing the source code.
- Missing-region handling is deterministic and recorded.



