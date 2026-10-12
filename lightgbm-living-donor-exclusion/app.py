"""LightGBM AKI risk calculator for the living-donor-exclusion sensitivity analysis."""

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
ANALYSIS_LABEL = "Living-donor-exclusion sensitivity analysis"
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


@st.cache_resource(show_spinner="Loading the locked LightGBM model...")
def load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


def make_probability_shap_figure(bundle: dict, raw_input: pd.DataFrame, scaled_input: pd.DataFrame):
    """Render a per-request, uncluttered probability-scale SHAP force plot in memory."""
    features = bundle["features"]
    model = bundle["model"]
    baseline = pd.DataFrame([bundle["shap_baseline"]], columns=features)
    scaled_baseline = bundle["scaler"].transform(baseline)

    # SHAP uses matplotlib's global state. Serializing this short operation avoids
    # cross-session figure interference while still keeping every image in memory.
    with PLOT_LOCK:
        explainer = shap.KernelExplainer(
            lambda values: model.predict_proba(pd.DataFrame(values, columns=features))[:, 1],
            scaled_baseline,
            seed=2025,
        )
        values = np.asarray(explainer.shap_values(scaled_input, nsamples="auto"))[0]
        # The default form values equal the stored median baseline. SHAP's
        # force renderer cannot lay out an all-zero contribution vector.
        if np.allclose(values, 0.0, atol=1e-12, rtol=0):
            return None
        explanation = shap.Explanation(
            values=values,
            base_values=float(explainer.expected_value),
            data=raw_input.iloc[0].to_numpy(),
            feature_names=["NHP-UFR", "VIS", "DBIL", "MELD", "sCr", "Plasma", "Operation Time", "Hb", "Hypertension"],
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            shap.plots.force(
                explanation, matplotlib=True, show=False, figsize=(16, 4.2),
                contribution_threshold=0.05,
            )
        figure = plt.gcf()
        axis = plt.gca()
        axis.set_facecolor("white")
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
    outside: list[str] = []
    for feature in bundle["features"]:
        config = bundle["input_config"][feature]
        if values[feature] < config["observed_min"] or values[feature] > config["observed_max"]:
            outside.append(INPUT_LABELS[feature])
    return outside


def main() -> None:
    st.set_page_config(page_title="AKI Risk Calculator | LightGBM", page_icon="🩺", layout="centered")
    bundle = load_bundle()
    metadata = bundle["metadata"]
    st.title("Postoperative Acute Kidney Injury (AKI) Risk Calculator")
    st.caption(
        f"{ANALYSIS_LABEL} | {metadata['model_name']} | Training cohort: {metadata['training_n']} patients; "
        f"independent test cohort: {metadata['test_n']} patients (AUC {metadata['test_auc']:.2f})"
    )
    st.info("For research and educational use only. This calculator must not be used as an independent basis for clinical decision-making.")

    values: dict[str, float] = {}
    with st.form("risk_calculator"):
        for feature in bundle["features"]:
            config = bundle["input_config"][feature]
            if config["kind"] == "binary":
                values[feature] = float(
                    st.selectbox(INPUT_LABELS[feature], options=[0, 1], index=int(config["default"]),
                                 format_func=lambda value: "Yes" if value else "No")
                )
            else:
                values[feature] = float(
                    st.number_input(
                        INPUT_LABELS[feature],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config["step"]),
                    )
                )
        submitted = st.form_submit_button("Calculate AKI Risk", type="primary")

    if not submitted:
        return

    raw_input = pd.DataFrame([values], columns=bundle["features"])
    scaled_input = bundle["scaler"].transform(raw_input)
    probability = float(bundle["model"].predict_proba(scaled_input)[0, 1])
    threshold = float(bundle["threshold"])
    predicted_high_risk = probability >= threshold

    st.metric("Predicted AKI Probability", f"{probability:.1%}")
    if predicted_high_risk:
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
                st.pyplot(figure, clear_figure=True, use_container_width=True, dpi=150, bbox_inches=None)
                plt.close(figure)
        st.caption("The baseline is the training-set median for each variable. Contributions reflect the fitted model's associations with predicted AKI probability and do not imply causality.")


if __name__ == "__main__":
    main()
