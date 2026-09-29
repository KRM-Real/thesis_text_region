"""Command-line entry points for the fixture-first research prototype."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ModelPackageManifest, RunManifest
from .data.manifest import ManifestError, audit_manifest, load_manifest, write_audit
from .data.prepare import prepare_dataset
from .data.splits import create_split_manifest, validate_split_leakage, write_split_manifest
from .evaluation.feature_analysis import compute_feature_analysis, human_summary, write_feature_analysis
from .evaluation.metrics import evaluate_scores, run_feature_group_ablation, write_evaluation_report
from .evaluation.report import EvaluationReport, load_evaluation_report
from .features.batch import extract_one_record, initialize_feature_worker
from .fixtures import generate_fixture_dataset
from .inference.service import predict_image_path
from .models.package import load_model_package, save_model_package
from .models.train import predict_scores, refit_selected, train_and_select
from .prototype_config import load_prototype_config, write_config_snapshot
from .reproducibility import (
    hardware_context,
    make_run_id,
    runtime_versions,
    sha256_file,
    sha256_json,
    source_revision,
    utc_now,
    write_json,
)
from .text_regions.adapters import EasyOCRDetectorAdapter, FixtureTextDetector


def _config_and_run(config_path: str, label: str, output_dir: str | None) -> tuple[dict[str, Any], Path, str]:
    config = load_prototype_config(config_path)
    run_id = make_run_id(config["_sha256"], source_revision(), label)
    target = Path(output_dir) if output_dir else Path(config["artifacts"]["root"]) / run_id
    target.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config, target / "config_snapshot.yaml")
    return config, target, run_id


def _write_run_manifest(target: Path, run_id: str, config: dict[str, Any], status: str, **kwargs: Any) -> None:
    detector_settings = dict(config.get("detector", {}))
    detector_settings["config_sha256"] = sha256_json(detector_settings)
    manifest = RunManifest(
        run_id=run_id,
        source_revision=source_revision(),
        configuration_path=str(config.get("_path", "configs/prototype.yaml")),
        configuration_sha256=config["_sha256"],
        audited_manifest_path=kwargs.get("audited_manifest_path"),
        audited_manifest_sha256=kwargs.get("audited_manifest_sha256"),
        split_manifest_path=kwargs.get("split_manifest_path"),
        split_manifest_sha256=kwargs.get("split_manifest_sha256"),
        detector=detector_settings,
        crop_preprocessing=config.get("preprocessing", {}),
        feature_schema_version=config.get("features", {}).get("schema_version", "features-1"),
        feature_schema_sha256=kwargs.get("feature_schema_sha256", ""),
        aggregation_version=config.get("aggregation", {}).get("schema_version", "aggregation-1"),
        model_settings=config.get("models", {}),
        selected_model_id=kwargs.get("selected_model_id"),
        threshold=kwargs.get("threshold"),
        threshold_method=kwargs.get("threshold_method"),
        random_seeds={"split": int(config.get("dataset", {}).get("split", {}).get("seed", 42)), "model": int(config.get("models", {}).get("random_seed", 42))},
        sample_counts=kwargs.get("sample_counts", {}),
        software_versions=runtime_versions(),
        hardware_context=hardware_context(),
        started_at_utc=kwargs.get("started_at_utc", utc_now()),
        completed_at_utc=utc_now() if status in {"completed", "failed"} else None,
        status=status,
        failure_reason=kwargs.get("failure_reason"),
        feature_analysis=kwargs.get("feature_analysis"),
    )
    write_json(target / "run_manifest.json", manifest.to_dict())


def _study_language(records: list[Any] | None) -> tuple[str, str, str]:
    """Return report wording that matches the provenance of the input manifest."""
    provenance = {str(getattr(record, "provenance_type", "")).lower() for record in (records or [])}
    fixture_only = bool(provenance) and provenance.issubset({"fixture", "generated_fixture"})
    if fixture_only:
        return (
            "Fixture evaluation report",
            "This report is software-test evidence only. The generated fixtures are not thesis data and the metrics do not establish authenticity or generalization.",
            "Fixture/software-test package; limited to documented study conditions and not thesis evidence.",
        )
    return (
        "Pilot evaluation report",
        "This report summarizes the supplied receipt dataset under the documented pilot configuration and split. The metrics do not establish universal authenticity or generalization.",
        "Pilot dataset package; limited to the supplied receipt dataset, configuration, and split; not evidence of universal authenticity or generalization.",
    )


def command_generate_fixtures(args: argparse.Namespace) -> int:
    manifest = generate_fixture_dataset(args.root, args.count_per_label, args.seed)
    print(json.dumps({"manifest": str(manifest), "status": "completed", "fixture_only": True}, indent=2))
    return 0


def command_prepare_dataset(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "prepare-dataset", args.output_dir)
    try:
        result = prepare_dataset(
            args.input_root,
            args.output_root,
            args.manifest_root,
            real_dir=args.real_dir,
            ai_dir=args.ai_dir,
            source_id=args.source_id,
            category=args.category,
            target_width=args.target_width,
            target_height=args.target_height,
            seed=args.seed,
            near_duplicate_distance=args.near_duplicate_distance,
        )
        result["run_id"] = run_id
        write_json(target / "dataset_processing.json", result)
        _write_run_manifest(
            target,
            run_id,
            config,
            "completed",
            audited_manifest_path=result["audited_manifest"],
            audited_manifest_sha256=sha256_file(result["audited_manifest"]),
            sample_counts={
                "source": int(result["source_count"]),
                "selected_balanced": int(result["selected_count"]),
                "review": int(result["review_count"]),
                "excluded": int(result["excluded_count"]),
                "skipped": int(result["skipped_count"]),
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        _write_run_manifest(target, run_id, config, "failed", failure_reason=str(exc))
        print(f"dataset preparation failed: {exc}")
        return 2


def command_validate_data(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "validate-data", args.output_dir)
    try:
        records = load_manifest(args.manifest, args.data_root, verify_checksums=not args.skip_checksum)
        validate_split_leakage(records)
        audit = audit_manifest(records, args.data_root)
        write_audit(target / "dataset_audit.json", audit)
        dataset_dir = target / "dataset"
        write_json(dataset_dir / "audit.json", audit)
        (dataset_dir / "manifest_sha256.txt").write_text(sha256_file(args.manifest) + "\n", encoding="utf-8")
        pd.DataFrame(audit.get("near_duplicate_candidates", [])).to_csv(dataset_dir / "duplicate_review.csv", index=False)
        write_json(dataset_dir / "shortcut_audit.json", audit.get("shortcut_diagnostics", {}))
        (dataset_dir / "decision_note.md").write_text("# Dataset decision note\n\nThis audit is a prototype gate. Receipt inclusion and provenance remain subject to human review.\n", encoding="utf-8")
        _write_run_manifest(target, run_id, config, "completed", audited_manifest_path=str(args.manifest), audited_manifest_sha256=sha256_file(args.manifest), sample_counts={"total": len(records)})
        print(json.dumps({"run_id": run_id, "status": "completed", "sample_count": len(records), "audit": str(target / "dataset_audit.json")}, indent=2))
        return 0
    except Exception as exc:
        _write_run_manifest(target, run_id, config, "failed", failure_reason=str(exc))
        print(f"validation failed: {exc}")
        return 2


def command_create_splits(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "create-splits", args.output_dir)
    try:
        records = load_manifest(args.manifest, args.data_root)
        split_records = create_split_manifest(records, int(config["dataset"]["split"].get("seed", 42)))
        split_path = target / "dataset" / "split_manifest.csv"
        write_split_manifest(split_path, split_records)
        split_hashes = {"audited_manifest_sha256": sha256_file(args.manifest), "split_manifest_sha256": sha256_file(split_path)}
        write_json(target / "dataset" / "manifest_sha256.json", split_hashes)
        (target / "dataset" / "manifest_sha256.txt").write_text("audited_manifest_sha256=" + split_hashes["audited_manifest_sha256"] + "\n" + "split_manifest_sha256=" + split_hashes["split_manifest_sha256"] + "\n", encoding="utf-8")
        counts = {split: sum(record.split == split for record in split_records) for split in ("train", "validation", "test")}
        _write_run_manifest(target, run_id, config, "completed", audited_manifest_path=str(args.manifest), audited_manifest_sha256=sha256_file(args.manifest), split_manifest_path=str(split_path), split_manifest_sha256=sha256_file(split_path), sample_counts=counts)
        print(json.dumps({"run_id": run_id, "status": "completed", "split_manifest": str(split_path), "counts": counts}, indent=2))
        return 0
    except Exception as exc:
        _write_run_manifest(target, run_id, config, "failed", failure_reason=str(exc))
        print(f"split creation failed: {exc}")
        return 2


def _detector(config: dict[str, Any], fixture: bool):
    return FixtureTextDetector() if fixture else EasyOCRDetectorAdapter(config)


def _feature_quality_report(frame: pd.DataFrame, schema_entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model_columns = sorted(name for name, entry in schema_entries.items() if entry.get("model_input", True) and not entry.get("diagnostic_only", False))
    missingness: dict[str, Any] = {}
    ranges: dict[str, Any] = {}
    constant_columns: list[str] = []
    nonfinite_columns: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    split_series = frame["split"].astype(str) if "split" in frame.columns else pd.Series("", index=frame.index)
    label_series = frame["label"].astype(str) if "label" in frame.columns else pd.Series("", index=frame.index)
    for name in model_columns:
        series = pd.to_numeric(frame.get(name, pd.Series(index=frame.index, dtype=float)), errors="coerce")
        entry = schema_entries[name]
        group = str(entry.get("group", "unknown"))
        group_counts[group] = group_counts.get(group, 0) + 1
        missingness[name] = {
            "total": int(series.isna().sum()),
            "by_split": {str(split): int(series[split_series == str(split)].isna().sum()) for split in sorted(split_series.unique())},
            "by_label": {str(label): int(series[label_series == str(label)].isna().sum()) for label in sorted(label_series.unique())},
        }
        finite = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]
        ranges[name] = {"min": float(finite.min()) if not finite.empty else None, "max": float(finite.max()) if not finite.empty else None, "unique_non_missing": int(finite.nunique())}
        nonfinite = int((~np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)) & series.notna().to_numpy()).sum())
        if nonfinite:
            nonfinite_columns[name] = nonfinite
        if finite.nunique() <= 1:
            constant_columns.append(name)
    return {"feature_count": len(model_columns), "feature_count_by_group": group_counts, "missingness": missingness, "ranges": ranges, "constant_columns": constant_columns, "nonfinite_columns": nonfinite_columns}


def command_extract_features(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "extract-features", args.output_dir)
    records = load_manifest(args.manifest, args.data_root)
    feature_dir = target / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, int(args.workers))
    write_json(
        target / "feature_extraction_settings.json",
        {
            "manifest": str(args.manifest),
            "data_root": str(args.data_root),
            "workers": worker_count,
            "fixture_detector": bool(args.fixture_detector),
            "detector": config.get("detector", {}),
            "feature_config": config.get("features", {}),
            "aggregation_config": config.get("aggregation", {}),
        },
    )
    payloads = [record.to_dict() for record in records]
    if worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_feature_worker,
            initargs=(config, str(args.data_root), str(target), bool(args.fixture_detector), worker_count),
        ) as executor:
            results = list(executor.map(extract_one_record, payloads, chunksize=1))
    else:
        initialize_feature_worker(config, str(args.data_root), str(target), bool(args.fixture_detector), worker_count)
        results = [extract_one_record(payload) for payload in payloads]
    rows: list[dict[str, Any]] = [result["row"] for result in results]
    image_records: list[dict[str, Any]] = [result["image_record"] for result in results]
    region_records: list[dict[str, Any]] = [record for result in results for record in result["region_records"]]
    quality_counts: dict[str, int] = {}
    schemas: dict[str, dict[str, Any]] = {}
    for result in results:
        for flag in result["quality_flags"]:
            quality_counts[flag] = quality_counts.get(flag, 0) + 1
        schemas.update(result["schemas"])
    rows.sort(key=lambda row: str(row.get("sample_id", "")))
    image_records.sort(key=lambda row: str(row.get("sample_id", "")))
    region_records.sort(key=lambda row: (str(row.get("sample_id", "")), str(row.get("region_id", "")), str(row.get("feature", ""))))
    frame = pd.DataFrame(rows)
    frame.to_csv(feature_dir / "image_features.csv", index=False)
    (feature_dir / "image_feature_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in image_records), encoding="utf-8")
    (feature_dir / "region_features.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in region_records), encoding="utf-8")
    pd.DataFrame(region_records, columns=["sample_id", "label", "region_id", "feature", "group", "value", "status", "formula_id", "reason"]).to_csv(feature_dir / "region_features.csv", index=False)
    schema_payload = {"schema_version": config["features"]["schema_version"], "aggregation_version": config["aggregation"]["schema_version"], "columns": list(schemas.values()), "model_columns": sorted(name for name, entry in schemas.items() if entry.get("model_input", True) and not entry.get("diagnostic_only", False))}
    write_json(feature_dir / "feature_schema.json", schema_payload)
    write_json(feature_dir / "feature_quality.json", {"rows": len(rows), "failed_rows": int(frame.get("processing_status", pd.Series(dtype=str)).eq("failed").sum()), "quality_flags": quality_counts, **_feature_quality_report(frame, schemas)})
    _write_run_manifest(target, run_id, config, "completed", audited_manifest_path=str(args.manifest), audited_manifest_sha256=sha256_file(args.manifest), sample_counts={"images": len(rows), "failed": int(frame.get("processing_status", pd.Series(dtype=str)).eq("failed").sum())}, feature_schema_sha256=sha256_file(feature_dir / "feature_schema.json"))
    print(json.dumps({"run_id": run_id, "status": "completed", "features": str(feature_dir / "image_features.csv"), "rows": len(rows)}, indent=2))
    return 0


def _load_feature_data(features_path: str | Path, split_manifest: str | Path | None = None) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame = pd.read_csv(features_path)
    if "sample_id" not in frame.columns or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("feature matrix must contain exactly one non-empty row per sample_id")
    if "processing_status" in frame.columns and frame["processing_status"].eq("failed").any():
        raise ValueError("feature extraction contains failed rows; resolve them before training or evaluation")
    schema_path = Path(features_path).with_name("feature_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    columns = list(schema.get("model_columns", []))
    if not columns:
        columns = [column for column in frame.columns if column not in {"sample_id", "label", "split", "processing_status", "processing_error"}]
    if split_manifest:
        split_frame = pd.read_csv(split_manifest, usecols=["sample_id", "split"])
        frame = frame.drop(columns=["split"], errors="ignore").merge(split_frame, on="sample_id", how="left", validate="one_to_one")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"feature schema columns missing from matrix: {missing[:5]}")
    forbidden_tokens = ("recognized_text", "ocr", "word", "semantic", "language", "merchant", "amount", "filename", "relative_path", "source_id", "generator")
    forbidden_columns = [column for column in columns if any(token in column.lower() for token in forbidden_tokens)]
    if forbidden_columns:
        raise ValueError(f"feature schema contains forbidden semantic or provenance columns: {forbidden_columns[:5]}")
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() != frame[column].notna().sum():
            raise ValueError(f"feature column is not numeric: {column}")
        frame[column] = converted
    frame["label"] = frame["label"].map({"non_ai_generated": 0, "ai_generated": 1, "REAL": 0, "AI_GENERATED": 1})
    if frame["label"].isna().any():
        raise ValueError("feature matrix contains unknown labels")
    return frame, columns, schema


def _reference_input_checksums(
    reference: EvaluationReport,
    predictions: list[dict[str, Any]],
) -> dict[str, str]:
    """Reuse image checksums only when a prior report exactly matches this test evaluation."""
    reference_by_id = {str(row["sample_id"]): row for row in reference.predictions}
    current_by_id = {str(row["sample_id"]): row for row in predictions}
    if len(reference_by_id) != len(reference.predictions) or len(current_by_id) != len(predictions):
        raise ValueError("reference evaluation contains duplicate sample IDs")
    if reference_by_id.keys() != current_by_id.keys():
        raise ValueError("reference evaluation test sample IDs do not exactly match the current test split")

    checksums: dict[str, str] = {}
    for sample_id, current in current_by_id.items():
        prior = reference_by_id[sample_id]
        for field in ("true_label", "predicted_label"):
            if prior[field] != current[field]:
                raise ValueError(f"reference evaluation {field} does not match current prediction for {sample_id}")
        for field in ("score", "threshold"):
            if not np.isclose(float(prior[field]), float(current[field]), rtol=1e-9, atol=1e-12):
                raise ValueError(f"reference evaluation {field} does not match current prediction for {sample_id}")
        checksum = str(prior.get("input_sha256", "")).strip().lower()
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise ValueError(f"reference evaluation has no valid image checksum for {sample_id}")
        checksums[sample_id] = checksum
    return checksums


def command_train(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "train", args.output_dir)
    audited_manifest_sha = None
    audited_records = []
    if args.audited_manifest:
        audited_records = load_manifest(args.audited_manifest, args.data_root)
        validate_split_leakage(audited_records)
        audited_manifest_sha = sha256_file(args.audited_manifest)
    frame, columns, schema = _load_feature_data(args.features, args.split_manifest)
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("training and validation rows are required")
    selected = train_and_select(train, validation, columns, config)
    development = pd.concat([train, validation], ignore_index=True)
    selected = refit_selected(selected, development)
    model_id = f"model-{run_id}"
    package_dir = target / "models" / model_id
    split_hash = sha256_file(args.split_manifest) if args.split_manifest else ""
    feature_schema_hash = sha256_file(Path(args.features).with_name("feature_schema.json"))
    _, _, study_warning = _study_language(audited_records)
    manifest = ModelPackageManifest(
        model_id=model_id,
        created_at_utc=utc_now(),
        label_mapping={"non_ai_generated": 0, "ai_generated": 1},
        estimator_family=selected.family,
        hyperparameters=selected.hyperparameters,
        feature_schema_version=str(schema.get("schema_version", "features-1")),
        feature_columns=columns,
        preprocessor_version="median-imputer-v1+optional-standard-scaler-v1",
        detector_config_sha256=sha256_json(config.get("detector", {})),
        feature_config_sha256=sha256_json(config.get("features", {})),
        split_manifest_sha256=split_hash,
        training_sample_count=len(train),
        validation_sample_count=len(validation),
        decision_threshold=selected.threshold,
        threshold_method=selected.threshold_method,
        runtime_versions=runtime_versions(),
        study_warning=study_warning,
        run_id=run_id,
        feature_schema_sha256=feature_schema_hash,
    )
    save_model_package(
        package_dir,
        selected,
        manifest,
        metadata={
            "detector_config": config.get("detector", {}),
            "feature_config": config.get("features", {}),
            "feature_schema": schema,
            "validation_metrics": selected.validation_metrics,
            "training_summary": {
                "model_id": model_id,
                "run_id": run_id,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "development_rows_after_refit": len(development),
                "candidate_count": selected.validation_metrics.get("candidate_count", 0),
                "study_warning": manifest.study_warning,
            },
            "split_manifest_sha256": split_hash,
        },
    )
    reports_dir = target / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison = selected.validation_metrics.get("candidate_comparison", [])
    comparison_rows = []
    for candidate in comparison:
        row = {"family": candidate["family"], "hyperparameters": json.dumps(candidate["hyperparameters"], sort_keys=True), "threshold": candidate["threshold"], "threshold_method": candidate["threshold_method"]}
        row.update(candidate["validation_metrics"])
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(reports_dir / "model_comparison.csv", index=False)
    write_json(reports_dir / "validation_metrics.json", {"selected_model_id": model_id, "estimator_family": selected.family, "validation_metrics": selected.validation_metrics})
    write_json(target / "models" / "model_comparison.json", {"selected_model_id": model_id, "estimator_family": selected.family, "validation_metrics": selected.validation_metrics})
    _write_run_manifest(target, run_id, config, "completed", audited_manifest_path=str(args.audited_manifest) if args.audited_manifest else None, audited_manifest_sha256=audited_manifest_sha, split_manifest_path=str(args.split_manifest or ""), split_manifest_sha256=split_hash, selected_model_id=model_id, threshold=selected.threshold, threshold_method=selected.threshold_method, sample_counts={"train": len(train), "validation": len(validation)}, feature_schema_sha256=feature_schema_hash)
    print(json.dumps({"run_id": run_id, "status": "completed", "model_id": model_id, "package": str(package_dir), "validation": selected.validation_metrics}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "evaluate", args.output_dir)
    frame, columns, schema = _load_feature_data(args.features, args.split_manifest)
    if args.audited_manifest and args.reference_evaluation_run:
        raise ValueError("use either --audited-manifest or --reference-evaluation-run, not both")
    audited_manifest_sha = None
    checksums: dict[str, str] = {}
    audited_records = []
    if args.audited_manifest:
        audited_records = load_manifest(args.audited_manifest, args.data_root)
        validate_split_leakage(audited_records)
        checksums = {record.sample_id: record.sha256 for record in audited_records}
        audited_manifest_sha = sha256_file(args.audited_manifest)
    package = load_model_package(args.model_package, columns)
    test = frame[frame["split"] == "test"].copy()
    if test.empty:
        raise ValueError("test split is empty")
    from .models.train import FittedCandidate

    candidate = FittedCandidate(package["manifest"]["estimator_family"], package["manifest"]["hyperparameters"], columns, package["preprocessor"], package["estimator"], float(package["manifest"]["decision_threshold"]), package["manifest"]["threshold_method"], {}, np.asarray([]))
    scores = predict_scores(candidate, test)
    metrics = evaluate_scores(test["label"].astype(int).to_numpy(), scores, candidate.threshold)
    predictions = []
    for row_index, (sample_id, label, score) in enumerate(zip(test["sample_id"], test["label"], scores)):
        source_row = test.iloc[row_index]
        predictions.append({
            "sample_id": sample_id,
            "true_label": int(label),
            "score": float(score),
            "predicted_label": "ai_generated" if score >= candidate.threshold else "non_ai_generated",
            "threshold": candidate.threshold,
            "region_count": source_row.get("density.region_count"),
            "no_detection": source_row.get("quality.no_detection"),
            "input_sha256": checksums.get(str(sample_id), ""),
        })
    reference_run_id = None
    reference_predictions_sha256 = None
    if args.reference_evaluation_run:
        reference_report = load_evaluation_report(
            args.reference_evaluation_run,
            package["manifest"],
            active_config=config,
        )
        checksums = _reference_input_checksums(reference_report, predictions)
        for prediction in predictions:
            prediction["input_sha256"] = checksums[str(prediction["sample_id"])]
        reference_run_id = reference_report.run_id
        reference_predictions_sha256 = sha256_file(
            Path(args.reference_evaluation_run) / "reports" / "test_predictions.csv"
        )
    ablation_rows = run_feature_group_ablation(frame, schema, package["manifest"]["estimator_family"], package["manifest"]["hyperparameters"])
    analysis_settings = config.get("evaluation", {}).get("feature_analysis", {})
    analysis_report = compute_feature_analysis(
        frame,
        schema,
        candidate,
        ablation_rows,
        analysis_settings,
        run_id=run_id,
        model_id=str(package["manifest"]["model_id"]),
    )
    partition_counts = {
        name: int(frame["split"].astype(str).eq(name).sum())
        for name in ("train", "validation", "test")
    }
    reports_dir = target / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablation_rows).to_csv(reports_dir / "feature_group_ablation.csv", index=False)
    analysis_artifacts = write_feature_analysis(reports_dir, analysis_report)
    analysis_hashes = {name: sha256_file(path) for name, path in analysis_artifacts.items()}
    if args.audited_manifest:
        report_title, report_note, _ = _study_language(audited_records)
    elif args.reference_evaluation_run:
        report_title = "Pilot feature-matrix evaluation report"
        report_note = (
            "This run reused per-image checksums from a prior evaluation report after validating the same model, "
            "feature schema, split, test sample IDs, labels, thresholds, and prediction scores. No source images "
            "were reopened; the reference report and prediction artifact hashes are recorded in run_manifest.json."
        )
    elif "fixture" in str(package["manifest"].get("study_warning", "")).lower():
        report_title = "Fixture evaluation report"
        report_note = "This report is software-test evidence only. It was generated from fixture features and does not establish thesis performance."
    else:
        report_title = "Pilot feature-matrix evaluation report"
        report_note = "This run reused an image-level feature matrix and saved split. The original image manifest and image checksums were not re-verified by this command; use the recorded source hashes and audit artifacts for provenance."
    summary = (
        f"# {report_title}\n\n"
        f"Run ID: `{run_id}`\n\n"
        f"{report_note}\n\n"
        f"Positive class: `{metrics['positive_class']}`. Threshold rule: `{metrics['threshold_convention']}`.\n\n"
        f"Metrics: `{json.dumps(metrics, sort_keys=True)}`\n\n"
        "The image is the unit of analysis; all regions remain inside the image split.\n\n"
        f"{human_summary(analysis_report)}\n"
    )
    write_evaluation_report(reports_dir, metrics, predictions, summary)
    write_json(reports_dir / "error_review.json", {"status": "descriptive_only", "false_positives": [row for row in predictions if row["true_label"] == 0 and row["predicted_label"] == "ai_generated"], "false_negatives": [row for row in predictions if row["true_label"] == 1 and row["predicted_label"] == "non_ai_generated"]})
    _write_run_manifest(
        target,
        run_id,
        config,
        "completed",
        audited_manifest_path=str(args.audited_manifest) if args.audited_manifest else None,
        audited_manifest_sha256=audited_manifest_sha,
        split_manifest_path=str(args.split_manifest or ""),
        split_manifest_sha256=sha256_file(args.split_manifest) if args.split_manifest else "",
        selected_model_id=package["manifest"]["model_id"],
        threshold=candidate.threshold,
        threshold_method=candidate.threshold_method,
        sample_counts=partition_counts,
        feature_schema_sha256=sha256_file(Path(args.features).with_name("feature_schema.json")),
        feature_analysis={
            "status": analysis_report.status,
            "artifact_sha256": analysis_hashes,
            "source_feature_matrix_path": str(Path(args.features)),
            "source_feature_matrix_sha256": sha256_file(args.features),
            "source_feature_schema_path": str(Path(args.features).with_name("feature_schema.json")),
            "source_feature_schema_sha256": sha256_file(Path(args.features).with_name("feature_schema.json")),
            "split_manifest_sha256": sha256_file(args.split_manifest) if args.split_manifest else "",
            "input_checksum_provenance": (
                "audited_manifest" if args.audited_manifest else
                "validated_reference_evaluation" if args.reference_evaluation_run else
                "not_reverified"
            ),
            "reference_evaluation_run_path": str(Path(args.reference_evaluation_run)) if args.reference_evaluation_run else None,
            "reference_evaluation_run_id": reference_run_id,
            "reference_test_predictions_sha256": reference_predictions_sha256,
        },
    )
    print(json.dumps({"run_id": run_id, "status": "completed", "metrics": metrics}, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    config, target, run_id = _config_and_run(args.config, "predict", args.output_dir)
    image_path = Path(args.image)
    result = predict_image_path(
        image_path,
        args.model_package,
        config,
        category="receipt" if args.category == "receipt" else None,
        fixture_detector=args.fixture_detector,
    )
    prediction = result.prediction.to_dict()
    prediction["run_id"] = run_id
    target.joinpath("localization").mkdir(parents=True, exist_ok=True)
    write_json(target / "predictions" / f"{image_path.stem}.json", prediction)
    localization_payload = result.extraction.detection.to_dict()
    localization_payload.update({"image_sha256": result.prediction.input_sha256, "model_id": result.prediction.model_id})
    write_json(target / "localization" / f"{image_path.stem}.json", localization_payload)
    result.overlay.save(target / "localization" / f"{image_path.stem}_overlay.png", format="PNG")
    _write_run_manifest(target, run_id, config, "completed", selected_model_id=result.prediction.model_id, threshold=result.prediction.threshold, threshold_method="package", sample_counts={"images": 1}, feature_schema_sha256=result.prediction.feature_schema_sha256)
    print(json.dumps(prediction, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixture-first visual text-region research prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("generate-fixtures")
    fixture.add_argument("--root", default="data")
    fixture.add_argument("--count-per-label", type=int, default=12)
    fixture.add_argument("--seed", type=int, default=42)
    fixture.set_defaults(func=command_generate_fixtures)
    prepare = sub.add_parser("prepare-dataset")
    prepare.add_argument("--input-root", required=True)
    prepare.add_argument("--output-root", default="data/processed/textfake_receipts_v1")
    prepare.add_argument("--manifest-root", default="data/manifests")
    prepare.add_argument("--real-dir", default="0_real")
    prepare.add_argument("--ai-dir", default="1_fake")
    prepare.add_argument("--source-id", default="textfake")
    prepare.add_argument("--category", choices=["receipt"], default="receipt")
    prepare.add_argument("--target-width", type=int, default=1024)
    prepare.add_argument("--target-height", type=int, default=1536)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--near-duplicate-distance", type=int, default=0)
    prepare.add_argument("--config", default="configs/prototype.yaml")
    prepare.add_argument("--output-dir")
    prepare.set_defaults(func=command_prepare_dataset)
    validate = sub.add_parser("validate-data")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--data-root", default="data")
    validate.add_argument("--config", default="configs/prototype.yaml")
    validate.add_argument("--output-dir")
    validate.add_argument("--skip-checksum", action="store_true")
    validate.set_defaults(func=command_validate_data)
    split = sub.add_parser("create-splits")
    split.add_argument("--manifest", required=True)
    split.add_argument("--data-root", default="data")
    split.add_argument("--config", default="configs/prototype.yaml")
    split.add_argument("--output-dir")
    split.set_defaults(func=command_create_splits)
    extract = sub.add_parser("extract-features")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--data-root", default="data")
    extract.add_argument("--config", default="configs/prototype.yaml")
    extract.add_argument("--output-dir")
    extract.add_argument("--fixture-detector", action="store_true")
    extract.add_argument("--workers", type=int, default=1)
    extract.set_defaults(func=command_extract_features)
    train = sub.add_parser("train")
    train.add_argument("--features", required=True)
    train.add_argument("--split-manifest")
    train.add_argument("--audited-manifest")
    train.add_argument("--data-root", default="data")
    train.add_argument("--config", default="configs/prototype.yaml")
    train.add_argument("--output-dir")
    train.set_defaults(func=command_train)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--split-manifest")
    evaluate.add_argument("--model-package", required=True)
    evaluate.add_argument("--audited-manifest")
    evaluate.add_argument("--reference-evaluation-run", help="Reuse test-image checksums from a compatible completed evaluation without reopening images")
    evaluate.add_argument("--data-root", default="data")
    evaluate.add_argument("--config", default="configs/prototype.yaml")
    evaluate.add_argument("--output-dir")
    evaluate.set_defaults(func=command_evaluate)
    predict = sub.add_parser("predict")
    predict.add_argument("--model-package", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--config", default="configs/prototype.yaml")
    predict.add_argument("--output-dir")
    predict.add_argument("--category", choices=["receipt", "unknown"], default="unknown")
    predict.add_argument("--fixture-detector", action="store_true")
    predict.set_defaults(func=command_predict)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (ManifestError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
