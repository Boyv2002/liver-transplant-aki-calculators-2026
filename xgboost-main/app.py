"""XGBoost AKI risk calculator for the primary analysis."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt  # noqa: E402


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "model_bundle.joblib"
PLOT_LOCK = Lock()
ANALYSIS_LABEL = "Primary analysis"
INPUT_LABELS = {
    "Neohepatic-phase Urine Flow Rate": "Neohepatic-phase urine flow rate (NHP-UFR)",
    "VIS": "Vasoactive-inotropic score (VIS)",
    "DBIL": "Direct bilirubin (DBIL)",
    "MELD": "Model for End-stage Liver Disease (MELD) score",
    "sCr": "Serum creatinine (sCr)",
    "Plasma transfusion": "Plasma transfusion",
    "Operation Time": "Operation time",
    "Hb": "Hemoglobin (Hb)",
    "Hypertension": "Hypertension",
}


@st.cache_resource(show_spinner="Loading the locked XGBoost model...")
def load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


def make_probability_shap_figure(bundle: dict, raw_input: pd.DataFrame, scaled_input: pd.DataFrame):
    """Create an uncluttered probability-scale XGBoost SHAP force plot in memory."""
    features = bundle["features"]
    baseline = pd.DataFrame([bundle["shap_baseline"]], columns=features)
    scaled_baseline = bundle["scaler"].transform(baseline)
    with PLOT_LOCK:
        explainer = shap.TreeExplainer(
            bundle["model"], data=scaled_baseline, model_output="probability",
            feature_perturbation="interventional",
        )
        source = explainer(scaled_input)
        values, base_values = np.asarray(source.values), np.asarray(source.base_values)
        class_index = int(np.flatnonzero(np.asarray(bundle["model"].classes_) == 1)[0])
        if values.ndim == 3:
            values = values[:, :, class_index]
        if base_values.ndim == 2:
            base_values = base_values[:, class_index]
        # The default form values equal the stored median baseline. SHAP's
        # waterfall renderer cannot lay out an all-zero contribution vector.
        if np.allclose(values[0], 0.0, atol=1e-12, rtol=0):
            return None
        explanation = shap.Explanation(
            values=values[0], base_values=float(base_values[0]), data=raw_input.iloc[0].to_numpy(),
            # Compact labels match the validated manuscript force-plot layout.
            feature_names=["NHP-UFR", "VIS", "DBIL", "MELD", "sCr", "Plasma", "Operation Time", "Hb", "Hypertension"],
        )
        # SHAP may emit a harmless divide-by-zero warning for a feature whose
        # contribution is exactly zero while it lays out the force plot.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            shap.plots.force(
                explanation, matplotlib=True, show=False, figsize=(16, 4.2),
                contribution_threshold=0.05,
            )
        figure = plt.gcf()
        axis = plt.gca()
        axis.set_facecolor("white")
        # Keep f(x) and the direction legend clear of the probability label.
        risk = float(explanation.base_values + np.sum(explanation.values))
        ticks = np.asarray(axis.get_xticks())
        axis.set_xticks(ticks[np.abs(ticks - risk) > 0.025])
        for text in axis.texts:
            if "=" in text.get_text():
                key, value = text.get_text().rsplit("=", 1)
                try:
                    text.set_text(f"{key}={float(value):.3f}")
                except ValueError:
                    pass
                text.set_rotation(45)
                text.set_rotation_mode("anchor")
                text.set_horizontalalignment("right")
            label = text.get_text()
            if label == "f(x)":
                x, y = text.get_position()
                text.set_position((x, y + 0.07))
            elif label in {"higher", "lower", r"$\leftarrow$", r"$\rightarrow$"}:
                x, y = text.get_position()
                text.set_position((x, y + 0.08))
        for image in axis.images:
            image.set_interpolation("nearest")
            image.set_resample(False)
        plt.subplots_adjust(bottom=0.54, top=0.88)
    return figure


def values_outside_observed_range(bundle: dict, values: dict[str, float]) -> list[str]:
    return [
        INPUT_LABELS[feature] for feature in bundle["features"]
        if values[feature] < bundle["input_config"][feature]["observed_min"]
        or values[feature] > bundle["input_config"][feature]["observed_max"]
    ]


def main() -> None:
    st.set_page_config(page_title="AKI Risk Calculator | XGBoost", page_icon="🩺", layout="centered")
    bundle = load_bundle()
    metadata = bundle["metadata"]
    st.title("Postoperative Acute Kidney Injury (AKI) Risk Calculator")
    st.caption(f"{ANALYSIS_LABEL} | {metadata['model_name']} | Training cohort: {metadata['training_n']} patients; independent test cohort: {metadata['test_n']} patients (AUC {metadata['test_auc']:.2f})")
    st.info("For research and educational use only. This calculator must not be used as an independent basis for clinical decision-making.")

    values: dict[str, float] = {}
    with st.form("risk_calculator"):
        for feature in bundle["features"]:
            config = bundle["input_config"][feature]
            if config["kind"] == "binary":
                values[feature] = float(st.selectbox(INPUT_LABELS[feature], [0, 1], index=int(config["default"]), format_func=lambda value: "Yes" if value else "No"))
            else:
                values[feature] = float(st.number_input(INPUT_LABELS[feature], min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["default"]), step=float(config["step"])))
        submitted = st.form_submit_button("Calculate AKI Risk", type="primary")
    if not submitted:
        return

    raw_input = pd.DataFrame([values], columns=bundle["features"])
    scaled_input = bundle["scaler"].transform(raw_input)
    probability = float(bundle["model"].predict_proba(scaled_input)[0, 1])
    threshold = float(bundle["threshold"])
    st.metric("Predicted AKI Probability", f"{probability:.1%}")
    if probability >= threshold:
        st.warning(f"Using the training out-of-fold threshold of {threshold:.3f}, this case is classified as high risk by the model.")
    else:
        st.success(f"Using the training out-of-fold threshold of {threshold:.3f}, this case is not classified as high risk by the model.")
    outside = values_outside_observed_range(bundle, values)
    if outside:
        st.warning("The following input(s) are within the calculator limits but outside the observed development-cohort range: " + ", ".join(outside) + ". The result is an extrapolation and should be interpreted cautiously.")

    with st.expander("View SHAP Probability Force Plot for This Case", expanded=False):
        with st.spinner("Generating the SHAP explanation for this case..."):
            figure = make_probability_shap_figure(bundle, raw_input, scaled_input)
            if figure is None:
                st.info("The current input equals the training-median baseline; all SHAP contributions relative to this baseline are zero.")
            else:
                # Fixed in-memory canvas: do not let Streamlit's tight bounding-box
                # calculation enlarge a SHAP figure because of off-canvas labels.
                st.pyplot(figure, clear_figure=True, use_container_width=True, dpi=150, bbox_inches=None)
                plt.close(figure)
        st.caption("The baseline is the training-set median for each variable. Contributions reflect the fitted model's associations with predicted AKI probability and do not imply causality.")


if __name__ == "__main__":
    main()
