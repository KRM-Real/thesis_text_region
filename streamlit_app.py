"""Thin Streamlit client over the shared prototype inference service."""

from __future__ import annotations

import io
import json
from pathlib import Path
from dataclasses import asdict

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from src.evaluation.report import EvaluationReport, EvaluationReportError, discover_evaluation_runs, load_evaluation_report
from src.evaluation.feature_analysis import GROUP_LABELS
from src.inference.service import predict_image_bytes
from src.prototype_config import load_prototype_config
from src.reproducibility import sha256_bytes
from src.text_regions.adapters import EasyOCRDetectorAdapter


ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="Receipt text-region prototype", page_icon=":material/receipt_long:", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("inference_result", None)
    st.session_state.setdefault("processing", False)


@st.cache_data(show_spinner=False)
def _config() -> dict:
    return load_prototype_config(ROOT / "configs" / "prototype.yaml")


@st.cache_resource(show_spinner=False)
def _production_detector() -> EasyOCRDetectorAdapter:
    """Share the initialized detection-only reader across Streamlit reruns."""

    return EasyOCRDetectorAdapter(_config())


@st.cache_data(show_spinner=False)
def _model_packages() -> list[tuple[str, str]]:
    packages: list[tuple[str, str]] = []
    for manifest in sorted((ROOT / "artifacts" / "runs").glob("*/models/*/package_manifest.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            packages.append((str(payload["model_id"]), str(manifest.parent)))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(packages, key=lambda item: Path(item[1]).stat().st_mtime if Path(item[1]).exists() else 0, reverse=True)


def _package_manifest(package_path: str | None) -> dict:
    if not package_path:
        return {}
    try:
        return json.loads((Path(package_path) / "package_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _package_role(manifest: dict) -> str:
    warning = str(manifest.get("study_warning", "")).lower()
    if "fixture" in warning or "software-test" in warning:
        return "software-test fixture"
    if "pilot dataset" in warning or "supplied receipt" in warning:
        return "receipt pilot"
    return "study package"


@st.cache_data(show_spinner=False)
def _evaluation_runs(model_id: str) -> list[str]:
    return [str(path) for path in discover_evaluation_runs(ROOT / "artifacts" / "runs", model_id)]


@st.cache_data(show_spinner=False)
def _load_evaluation_report(run_path: str, package_path: str, config: dict) -> EvaluationReport:
    package_manifest = _package_manifest(package_path)
    return load_evaluation_report(run_path, package_manifest, config)


def _evaluation_run_label(run_path: str) -> str:
    path = Path(run_path)
    try:
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path.name
    run_id = str(manifest.get("run_id", path.name))
    completed = str(manifest.get("completed_at_utc", "completion time unavailable"))
    return f"{run_id} · {completed}"


def _label_name(value: int | str) -> str:
    return "AI-generated" if str(value) in {"1", "ai_generated"} else "Non-AI-generated"


def _render_feature_evidence(report: EvaluationReport) -> None:
    analysis = report.feature_analysis
    group_rows = analysis.group_summary if analysis is not None else report.feature_group_ablation
    st.subheader("Feature-group evidence", icon=":material/category:")
    if not group_rows:
        st.info("This evaluation run does not contain a feature-group comparison.", icon=":material/info:")
    else:
        group_frame = pd.DataFrame(group_rows)
        if "display_name" not in group_frame:
            group_frame["display_name"] = group_frame["feature_group"].map(GROUP_LABELS).fillna(group_frame["feature_group"])
        if "delta_test_f1_vs_all" not in group_frame and "test_f1" in group_frame:
            all_rows = group_frame.loc[group_frame["feature_group"] == "all_enabled", "test_f1"]
            all_f1 = float(all_rows.iloc[0]) if not all_rows.empty else None
            group_frame["delta_test_f1_vs_all"] = group_frame["test_f1"].map(
                lambda value: float(value) - all_f1 if all_f1 is not None and pd.notna(value) else None
            )
        if "test_f1" in group_frame:
            chart_data = group_frame[group_frame["test_f1"].notna()]
            if not chart_data.empty:
                group_chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("test_f1:Q", title="Held-out F1 score", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("display_name:N", sort=alt.SortField(field="test_f1", order="descending"), title="Feature representation"),
                    color=alt.Color("feature_group:N", legend=None),
                    tooltip=[
                        alt.Tooltip("display_name:N", title="Feature group"),
                        alt.Tooltip("feature_count:Q", title="Feature columns"),
                        alt.Tooltip("test_f1:Q", title="Test F1", format=".4f"),
                        alt.Tooltip("delta_test_f1_vs_all:Q", title="Difference from all features", format="+.4f"),
                    ],
                )
                st.altair_chart(group_chart)
        visible_group_columns = [
            column for column in (
                "display_name", "feature_count", "status", "validation_f1", "test_accuracy",
                "test_precision", "test_recall", "test_f1", "delta_test_f1_vs_all",
                "test_missing_cell_rate", "test_no_detection_rate",
            ) if column in group_frame.columns
        ]
        st.dataframe(group_frame[visible_group_columns], hide_index=True, width="stretch")
        st.caption(
            "Feature-group ablation is the main feature-usefulness comparison. It compares each registered subset with the all-feature reference; it does not show causal effects."
        )

    st.subheader("Individual-feature diagnostics", icon=":material/analytics:")
    if analysis is None:
        st.info(
            "This evaluation run predates the individual-feature analysis. Its core metrics and any saved feature-group comparison remain available.",
            icon=":material/info:",
        )
        return

    if analysis.status == "posthoc_pilot":
        st.warning(
            "Post-hoc pilot analysis: individual-feature rankings describe this held-out split and were not used to fit or select the model.",
            icon=":material/science:",
        )
    elif analysis.status == "software_test":
        st.info("These feature results are generated-fixture software checks, not thesis evidence.", icon=":material/science:")
    else:
        st.info("This feature analysis was marked predeclared for the reported run. Interpret it within the study's receipt scope.", icon=":material/science:")
    st.caption(
        "Permutation F1 decrease estimates how much this frozen model's held-out F1 changed when one feature was shuffled. Positive values indicate a drop; zero or negative values mean the feature added little measured value in this run. Correlated features can share importance."
    )
    feature_rows = [asdict(item) for item in analysis.feature_influence]
    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        st.info("No individual model-input features were available for this analysis.", icon=":material/info:")
    else:
        top_count = int(analysis.analysis_config.get("top_global_features", 15))
        top_frame = feature_frame.head(top_count)
        feature_chart = alt.Chart(top_frame).mark_bar().encode(
            x=alt.X("permutation_f1_drop_mean:Q", title="Mean test F1 decrease after shuffling"),
            y=alt.Y("feature:N", sort=alt.SortField(field="permutation_f1_drop_mean", order="descending"), title="Visual feature"),
            color=alt.Color("group:N", title="Feature group"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("permutation_f1_drop_mean:Q", title="Mean F1 decrease", format=".5f"),
                alt.Tooltip("permutation_accuracy_drop_mean:Q", title="Mean accuracy decrease", format=".5f"),
                alt.Tooltip("permutation_abs_score_change_mean:Q", title="Mean absolute score change", format=".5f"),
                alt.Tooltip("cliffs_delta_ai_vs_non_ai:Q", title="Cliff's delta (AI vs non-AI)", format=".4f"),
            ],
        )
        st.altair_chart(feature_chart)
        search = st.text_input("Filter feature names", key="feature_analysis_search", placeholder="Type a feature name or group")
        shown = feature_frame
        if search.strip():
            query = search.strip().casefold()
            mask = feature_frame["feature"].str.casefold().str.contains(query, regex=False) | feature_frame["group"].str.casefold().str.contains(query, regex=False)
            shown = feature_frame[mask]
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            column_config={
                "permutation_f1_drop_mean": st.column_config.NumberColumn("Mean F1 decrease", format="%.5f"),
                "permutation_accuracy_drop_mean": st.column_config.NumberColumn("Mean accuracy decrease", format="%.5f"),
                "permutation_abs_score_change_mean": st.column_config.NumberColumn("Mean score change", format="%.5f"),
                "cliffs_delta_ai_vs_non_ai": st.column_config.NumberColumn("Cliff's delta", format="%.4f"),
                "non_ai_missing_rate": st.column_config.NumberColumn("Non-AI missing rate", format="percent"),
                "ai_missing_rate": st.column_config.NumberColumn("AI missing rate", format="percent"),
            },
        )
    st.subheader("Redundancy and interpretation limits", icon=":material/warning:")
    st.caption("Spearman correlations are computed from training and validation rows only. They flag measurements that may share importance; they do not establish why a class differs.")
    if analysis.redundancy:
        redundancy_frame = pd.DataFrame(analysis.redundancy)
        st.warning(f"{len(redundancy_frame)} highly correlated feature pair(s) met the configured threshold.", icon=":material/compare_arrows:")
        st.dataframe(redundancy_frame.head(100), hide_index=True, width="stretch")
        if len(redundancy_frame) > 100:
            st.caption("Showing the 100 strongest pairs; the download contains all recorded pairs.")
    else:
        st.info("No feature pairs met the configured correlation threshold in the development split.", icon=":material/info:")
    for note in analysis.interpretation_notes:
        st.caption(f"• {note}")

    with st.container(horizontal=True):
        for filename, label, mime in (
            ("feature_analysis.json", "Download feature analysis JSON", "application/json"),
            ("feature_influence.csv", "Download feature influence CSV", "text/csv"),
            ("feature_group_summary.csv", "Download feature-group summary CSV", "text/csv"),
            ("feature_redundancy.csv", "Download redundancy CSV", "text/csv"),
        ):
            path = report.run_path / "reports" / filename
            if path.is_file():
                st.download_button(label, path.read_bytes(), filename, mime, icon=":material/download:")


def _render_evaluation_report(report: EvaluationReport) -> None:
    metrics = report.metrics
    st.subheader("Evaluation report", icon=":material/assessment:")
    st.caption(
        f"Run: {report.run_id} · model: {report.model_id} · {metrics['sample_count']} held-out images. "
        "This is the saved test-run summary, not a recalculation for the image uploaded above; per-image model evidence appears under ‘Why this label?’ when supported."
    )
    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}", border=True)
            st.metric("Precision", f"{metrics['precision']:.3f}", border=True)
            st.metric("Recall", f"{metrics['recall']:.3f}", border=True)
            st.metric("F1 score", f"{metrics['f1']:.3f}", border=True)
        st.caption(f"Positive class: {metrics['positive_class']} · threshold rule: {metrics['threshold_convention']}")
        st.info(
            f"This report describes {metrics['sample_count']} images from this held-out evaluation run. "
            "It does not establish universal authenticity or generalization to unseen generators and sources.",
            icon=":material/science:",
        )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        with st.container(border=True):
            st.markdown("**Metric summary**")
            metrics_frame = pd.DataFrame(
                {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 score"],
                    "Value": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]],
                }
            )
            st.bar_chart(metrics_frame, x="Metric", y="Value", y_label="Metric value")
    with chart_right:
        with st.container(border=True):
            st.markdown("**Confusion matrix**")
            confusion_frame = pd.DataFrame(
                [
                    {"Actual": "Non-AI-generated", "Predicted": "Non-AI-generated", "Count": report.confusion_matrix["tn"]},
                    {"Actual": "Non-AI-generated", "Predicted": "AI-generated", "Count": report.confusion_matrix["fp"]},
                    {"Actual": "AI-generated", "Predicted": "Non-AI-generated", "Count": report.confusion_matrix["fn"]},
                    {"Actual": "AI-generated", "Predicted": "AI-generated", "Count": report.confusion_matrix["tp"]},
                ]
            )
            heatmap = alt.Chart(confusion_frame).mark_rect().encode(
                x=alt.X("Predicted:N", sort=["Non-AI-generated", "AI-generated"], title="Predicted label"),
                y=alt.Y("Actual:N", sort=["AI-generated", "Non-AI-generated"], title="Actual label"),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Images"),
                tooltip=["Actual:N", "Predicted:N", "Count:Q"],
            )
            labels = alt.Chart(confusion_frame).mark_text(fontSize=16).encode(
                x=alt.X("Predicted:N", sort=["Non-AI-generated", "AI-generated"]),
                y=alt.Y("Actual:N", sort=["AI-generated", "Non-AI-generated"]),
                text="Count:Q",
                color=alt.value("black"),
            )
            st.altair_chart(heatmap + labels)
    with st.container(border=True):
        st.markdown("**Scores and decision threshold**")
        score_frame = pd.DataFrame(report.predictions)
        score_frame["Actual label"] = score_frame["true_label"].map(_label_name)
        score_frame["Correctness"] = score_frame.apply(
            lambda row: "Correct" if row["predicted_label"] == ("ai_generated" if row["true_label"] == 1 else "non_ai_generated") else "Error",
            axis=1,
        )
        points = alt.Chart(score_frame).mark_circle(size=48, opacity=0.65).encode(
            x=alt.X("Actual label:N", sort=["Non-AI-generated", "AI-generated"], title="True label"),
            y=alt.Y("score:Q", title="Model score"),
            color=alt.Color("Correctness:N", scale=alt.Scale(domain=["Correct", "Error"], range=["#2e7d32", "#c62828"]), title="Result"),
            tooltip=["sample_id:N", "Actual label:N", "score:Q", "predicted_label:N", "region_count:Q", "no_detection:N"],
        )
        threshold = alt.Chart(pd.DataFrame({"threshold": [float(metrics["threshold"])]})).mark_rule(strokeDash=[6, 4], color="#5f6368").encode(
            y="threshold:Q"
        )
        st.altair_chart(points + threshold)
    if report.active_configuration_matches is False:
        st.caption(
            "This report was created with an earlier overall configuration revision. Its model, detector, feature schema, split, and threshold were validated against the selected package."
        )
    _render_feature_evidence(report)
    st.subheader("Error review", icon=":material/report:")
    if report.errors:
        st.warning(f"{len(report.errors)} descriptive error(s) are present in this evaluation run.", icon=":material/warning:")
        errors_frame = pd.DataFrame(report.errors)
        errors_frame["true_label"] = errors_frame["true_label"].map(_label_name)
        errors_frame["predicted_label"] = errors_frame["predicted_label"].map(_label_name)
        st.dataframe(errors_frame, hide_index=True, width="stretch")
    else:
        st.success("No errors were observed in this held-out evaluation run.", icon=":material/check_circle:")
    with st.expander("Report notes", icon=":material/description:"):
        st.markdown(report.summary_markdown)
    predictions_csv = pd.DataFrame(report.predictions).to_csv(index=False).encode("utf-8")
    errors_payload = {
        "status": "descriptive_only",
        "false_positives": [row for row in report.errors if row["true_label"] == 0],
        "false_negatives": [row for row in report.errors if row["true_label"] == 1],
    }
    confusion_csv = pd.DataFrame([report.confusion_matrix]).to_csv(index=False).encode("utf-8")
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True).encode("utf-8")
    errors_json = json.dumps(errors_payload, indent=2, sort_keys=True).encode("utf-8")
    with st.container(horizontal=True):
        st.download_button("Download metrics JSON", metrics_json, "test_metrics.json", "application/json", icon=":material/download:")
        st.download_button("Download predictions CSV", predictions_csv, "test_predictions.csv", "text/csv", icon=":material/download:")
        st.download_button("Download confusion CSV", confusion_csv, "confusion_matrix.csv", "text/csv", icon=":material/download:")
        st.download_button("Download error review", errors_json, "error_review.json", "application/json", icon=":material/download:")
        st.download_button("Download summary", report.summary_markdown.encode("utf-8"), "summary.md", "text/markdown", icon=":material/download:")


def _render_evaluation_panel(package_path: str | None, config: dict) -> None:
    if not package_path:
        return
    package_manifest = _package_manifest(package_path)
    model_id = str(package_manifest.get("model_id", ""))
    if not model_id:
        return
    with st.expander("Evaluation report", expanded=False, icon=":material/assessment:"):
        run_paths = _evaluation_runs(model_id)
        if not run_paths:
            st.info(
                "No completed evaluation report matches this model package. "
                "Run the frozen evaluation command before interpreting aggregate metrics.",
                icon=":material/info:",
            )
            return
        selected_run = st.selectbox(
            "Evaluation run",
            run_paths,
            format_func=_evaluation_run_label,
            key="evaluation_run",
        )
        try:
            report = _load_evaluation_report(selected_run, package_path, config)
        except EvaluationReportError as exc:
            st.error(f"The evaluation report was not loaded: {exc}", icon=":material/error:")
            return
        _render_evaluation_report(report)


def _render_result(result) -> None:
    prediction = result.prediction
    st.header("Result", icon=":material/analytics:")
    with st.container(border=True):
        first, second, third, fourth = st.columns(4)
        first.metric("Display label", prediction.display_label)
        second.metric("Score", f"{prediction.score:.4f}")
        third.metric("Regions", prediction.region_count)
        fourth.metric("No detection", "Yes" if prediction.no_detection else "No")
        st.caption(f"Score type: {prediction.score_type} · threshold: {prediction.threshold:.4f} · model: {prediction.model_id}")
        st.warning("Research prototype result; this is not proof of authenticity.", icon=":material/warning:")
    relation = "above" if prediction.score >= prediction.threshold else "below"
    with st.container(border=True):
        st.subheader("How to read this result", icon=":material/help:")
        st.write(
            f"The model score ({prediction.score:.4f}) is {relation} the decision threshold "
            f"({prediction.threshold:.4f}), so the system returned **{prediction.display_label}**."
        )
        st.caption(
            "This is a visual-feature classification under the evaluated receipt-dataset rules. "
            "It is not proof of authenticity and does not use recognized receipt text or meaning."
        )
        if prediction.no_detection:
            st.warning("No text-like regions were detected; the trained missing-value policy was used.", icon=":material/visibility_off:")
        elif prediction.quality_flags:
            st.info(f"The detector found {prediction.region_count} text-like regions; quality flags are listed below.", icon=":material/info:")
    explanation = result.explanation
    if explanation is not None:
        st.subheader("Why this label?", icon=":material/lightbulb:")
        if explanation.status == "available":
            if explanation.margin_over_threshold is not None and explanation.baseline_to_threshold is not None:
                st.write(
                    f"Baseline relative to the decision threshold ({explanation.baseline_to_threshold:+.5f}) "
                    f"plus all visual-feature contributions ({explanation.feature_contribution_sum:+.5f}) "
                    f"equals the model margin relative to that threshold ({explanation.margin_over_threshold:+.5f})."
                )
            else:
                st.write("This fitted model returned the shown label; its threshold is at an extreme score boundary, so a finite distance from the threshold is unavailable.")
            contribution_rows = [
                {**asdict(item), "direction": "Toward AI-generated"}
                for item in explanation.top_toward_ai
            ] + [
                {**asdict(item), "direction": "Toward non-AI-generated"}
                for item in explanation.top_toward_non_ai
            ]
            if explanation.remaining_contribution is not None:
                contribution_rows.append({
                    "feature": "Remaining feature contributions",
                    "group": "Multiple groups",
                    "component": "remainder",
                    "raw_value": None,
                    "transformed_value": None,
                    "coefficient": None,
                    "contribution": explanation.remaining_contribution,
                    "direction": "Combined remainder",
                })
            if contribution_rows:
                contribution_frame = pd.DataFrame(contribution_rows)
                contribution_chart = alt.Chart(contribution_frame[contribution_frame["component"] != "remainder"]).mark_bar().encode(
                    x=alt.X("contribution:Q", title=f"Contribution ({explanation.decision_scale})"),
                    y=alt.Y("feature:N", sort=alt.SortField(field="contribution", order="descending"), title="Feature"),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["Toward AI-generated", "Toward non-AI-generated"], range=["#a23b72", "#2878b5"]), title="Direction"),
                    tooltip=["feature:N", "group:N", "component:N", "raw_value:Q", "contribution:Q", "direction:N"],
                )
                st.altair_chart(contribution_chart)
                st.dataframe(contribution_frame, hide_index=True, width="stretch")
            if explanation.group_contributions:
                group_contribution_frame = pd.DataFrame(explanation.group_contributions)
                group_chart = alt.Chart(group_contribution_frame).mark_bar().encode(
                    x=alt.X("contribution:Q", title=f"Summed contribution ({explanation.decision_scale})"),
                    y=alt.Y("group:N", sort=alt.SortField(field="contribution", order="descending"), title="Feature group"),
                    color=alt.Color("direction:N", title="Direction"),
                    tooltip=["group:N", "contribution:Q", "direction:N"],
                )
                st.altair_chart(group_chart)
        else:
            st.info(explanation.reason or "This model does not provide an additive local explanation.", icon=":material/info:")
            st.caption("Use the measured feature values below and the selected package's global feature diagnostics in the evaluation report.")
        if explanation.feature_values:
            with st.expander("Measured feature values used by this model", icon=":material/data_object:"):
                st.dataframe(pd.DataFrame(explanation.feature_values), hide_index=True, width="stretch")
        if explanation.status != "available":
            st.caption(explanation.interpretation_note)
        explanation_payload = json.dumps(explanation.to_dict(), indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        st.download_button(
            "Download prediction explanation JSON",
            explanation_payload,
            "prediction-explanation-2.json",
            "application/json",
            icon=":material/download:",
        )
    overlay_col, details_col = st.columns(2)
    with overlay_col:
        st.subheader("Region overlay", icon=":material/crop_free:")
        st.image(result.overlay, width="stretch")
    with details_col:
        st.subheader("Traceability", icon=":material/fingerprint:")
        st.json({
            "input_sha256": prediction.input_sha256,
            "model_id": prediction.model_id,
            "feature_schema_version": prediction.feature_schema_version,
            "feature_schema_sha256": prediction.feature_schema_sha256,
            "aggregation_version": prediction.aggregation_version,
            "detector_config_sha256": prediction.detector_config_sha256,
            "feature_vector_sha256": prediction.feature_vector_sha256,
            "quality_flags": prediction.quality_flags,
        })
    if prediction.warnings:
        st.warning("\n".join(prediction.warnings), icon=":material/priority_high:")
    with st.expander("Region measurements", icon=":material/table_chart:"):
        rows = []
        for region_id, features in result.extraction.region_features.items():
            for name, feature in sorted(features.items()):
                rows.append({"region_id": region_id, "feature": name, "group": feature.group, "value": feature.value, "status": feature.status, "reason": feature.reason or ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Image-level feature record", icon=":material/data_object:"):
        st.dataframe(pd.DataFrame([result.extraction.image_feature_record.values]), hide_index=True, width="stretch")
    prediction_json = json.dumps(prediction.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    overlay_buffer = io.BytesIO()
    result.overlay.save(overlay_buffer, format="PNG")
    localization_payload = result.extraction.detection.to_dict()
    localization_payload.update({"image_sha256": prediction.input_sha256, "model_id": prediction.model_id})
    localization_json = json.dumps(localization_payload, indent=2, sort_keys=True).encode("utf-8")
    with st.container(horizontal=True):
        st.download_button("Download prediction JSON", prediction_json, "prediction.json", "application/json", icon=":material/download:")
        st.download_button("Download overlay", overlay_buffer.getvalue(), "localization_overlay.png", "image/png", icon=":material/download:")
        st.download_button("Download localization JSON", localization_json, "localization.json", "application/json", icon=":material/data_object:")


_init_state()
config = _config()
st.title("Receipt text-region prototype", icon=":material/receipt_long:")
st.caption("Visual text-region research pipeline · geometry and pixel features only")
st.info("The system localizes visible text-like regions and measures pixels and geometry. It does not use recognized words, language, merchant names, amounts, or semantics.", icon=":material/info:")

packages = _model_packages()
with st.container(border=True):
    st.header("Model and image input", icon=":material/upload:")
    with st.form("inference_form", border=False):
        if packages:
            package_labels = [item[0] for item in packages]
            package_roles = {model_id: _package_role(_package_manifest(path)) for model_id, path in packages}
            selected_model = st.selectbox(
                "Compatible model package",
                package_labels,
                format_func=lambda model_id: f"{model_id} · {package_roles.get(model_id, 'study package')}",
                key="model_package",
            )
            package_path = dict(packages)[selected_model]
            package_manifest = _package_manifest(package_path)
            st.caption(
                f"Family: {package_manifest.get('estimator_family', 'unknown')} · "
                f"schema: {package_manifest.get('feature_schema_version', 'unknown')} · "
                f"created: {package_manifest.get('created_at_utc', 'unknown')} · "
                f"role: {_package_role(package_manifest)}"
            )
            if package_manifest.get("study_warning"):
                st.info(package_manifest["study_warning"], icon=":material/science:")
        else:
            st.error("No compatible model package is available. Train a model package with the CLI before running inference.", icon=":material/error:")
            package_path = None
        uploaded = st.file_uploader("Input image", type=["png", "jpg", "jpeg", "webp"], key="image_upload")
        category = st.segmented_control("Category confirmation", ["Receipt", "Unknown"], default="Unknown", key="category_confirmation")
        submitted = st.form_submit_button("Analyze image", type="primary", icon=":material/play_arrow:")

if uploaded is not None:
    try:
        with Image.open(io.BytesIO(uploaded.getvalue())) as preview:
            preview.load()
            st.caption(f"Input: {preview.width}×{preview.height} · {preview.format or 'decoded image'} · SHA-256: {sha256_bytes(uploaded.getvalue())}")
    except Exception as exc:
        st.error(f"The selected image could not be decoded: {exc}", icon=":material/error:")

if submitted:
    if package_path is None:
        st.error("Load a compatible model package before running inference.")
    elif uploaded is None:
        st.error("Choose a supported image before running inference.")
    else:
        stages: list[dict[str, str]] = []
        with st.status("Running the locked pipeline…", expanded=True) as status:
            def progress(name: str, state: str) -> None:
                stages.append({"stage": name, "status": state})
                st.write(f"{name}: {state}")

            try:
                st.write("Loading the detection-only text-region model (cached after first use)…")
                detector = _production_detector()
                result = predict_image_bytes(
                    uploaded.getvalue(),
                    package_path,
                    config,
                    category="receipt" if category == "Receipt" else None,
                    detector=detector,
                    progress=progress,
                )
                st.session_state.inference_result = result
                status.update(label="Inference completed", state="complete")
            except Exception as exc:
                status.update(label="Inference stopped", state="error")
                st.error(str(exc), icon=":material/error:")

if st.session_state.inference_result is not None:
    _render_result(st.session_state.inference_result)

_render_evaluation_panel(package_path, config)
