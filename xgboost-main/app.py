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


@st.cache_resource(show_spinner="正在加载已锁定的 XGBoost 模型…")
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
        bundle["input_config"][feature]["label"] for feature in bundle["features"]
        if values[feature] < bundle["input_config"][feature]["observed_min"]
        or values[feature] > bundle["input_config"][feature]["observed_max"]
    ]


def main() -> None:
    st.set_page_config(page_title="AKI 风险计算器｜XGBoost", page_icon="🩺", layout="centered")
    bundle = load_bundle()
    metadata = bundle["metadata"]
    st.title("术后急性肾损伤（AKI）风险计算器")
    st.caption(f"{metadata['analysis']}｜{metadata['model_name']}｜训练集 {metadata['training_n']} 例，独立测试集 {metadata['test_n']} 例（AUC {metadata['test_auc']:.2f}）")
    st.info("仅供科研展示与辅助研究使用，不应用作独立临床诊疗决策。")

    values: dict[str, float] = {}
    with st.form("risk_calculator"):
        for feature in bundle["features"]:
            config = bundle["input_config"][feature]
            if config["kind"] == "binary":
                values[feature] = float(st.selectbox(config["label"], [0, 1], index=int(config["default"]), format_func=lambda value: "Yes" if value else "No"))
            else:
                values[feature] = float(st.number_input(config["label"], min_value=float(config["min"]), max_value=float(config["max"]), value=float(config["default"]), step=float(config["step"])))
        submitted = st.form_submit_button("计算 AKI 风险", type="primary")
    if not submitted:
        return

    raw_input = pd.DataFrame([values], columns=bundle["features"])
    scaled_input = bundle["scaler"].transform(raw_input)
    probability = float(bundle["model"].predict_proba(scaled_input)[0, 1])
    threshold = float(bundle["threshold"])
    st.metric("预测 AKI 概率", f"{probability:.1%}")
    if probability >= threshold:
        st.warning(f"按训练折外锁定阈值 {threshold:.3f}，该输入被归为模型高风险组。")
    else:
        st.success(f"按训练折外锁定阈值 {threshold:.3f}，该输入未归为模型高风险组。")
    outside = values_outside_observed_range(bundle, values)
    if outside:
        st.warning("以下输入虽在网页允许范围内，但已超出研究队列观察范围：" + "、".join(outside) + "。结果属于外推，应谨慎解读。")

    with st.expander("查看本例的 SHAP 概率力图", expanded=False):
        with st.spinner("正在生成本例的 SHAP 解释…"):
            figure = make_probability_shap_figure(bundle, raw_input, scaled_input)
            if figure is None:
                st.info("当前输入与训练集各变量中位数基线一致；相对该基线的各项 SHAP 贡献均为 0。")
            else:
                # Fixed in-memory canvas: do not let Streamlit's tight bounding-box
                # calculation enlarge a SHAP figure because of off-canvas labels.
                st.pyplot(figure, clear_figure=True, use_container_width=True, dpi=150, bbox_inches=None)
                plt.close(figure)
        st.caption("基线为训练集各变量中位数；图中贡献值解释的是 AKI 概率，而非因果效应。")


if __name__ == "__main__":
    main()
