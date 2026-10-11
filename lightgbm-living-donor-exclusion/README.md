# AKI risk calculator — LightGBM

This Streamlit app deploys the locked LightGBM model from the living-donor-exclusion sensitivity analysis. It estimates postoperative AKI probability from nine variables and provides a case-level, probability-scale SHAP explanation.

## Public repository contents

Commit only the following deployment materials:

- `app.py`
- `model_bundle.joblib`
- `requirements.txt`
- `runtime.txt`
- `README.md`

Do **not** commit the source CSV files, the `内部复核_不直接公开` directory, or any row-level predictions. `model_bundle.joblib` contains only the fitted model, fitted scaler, aggregate input ranges/medians, and model metadata; it contains no participant rows or outcomes.

## Local run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deployment and stable link

GitHub hosts the versioned source, but it does not execute Streamlit. Push this directory to a public GitHub repository, then create an app from that repository on a Streamlit-compatible hosting service (for example, Streamlit Community Cloud) with `app.py` as the entry point. The service-issued HTTPS app URL is the link to cite in the manuscript. Keep the repository and deployed app active; use tagged GitHub releases for each manuscript version and do not overwrite a published model bundle.

## Rebuilding the bundle internally

`build_bundle.py` is intentionally for the private research workspace only. It reconstructs the locked model from the final imputed training set and verifies every test-set LightGBM probability against the private audit file before writing `model_bundle.joblib`.

## Interpretation

The calculator is for research display and does not replace clinical judgment. Inputs outside the development-cohort observed range are allowed only within modestly expanded bounds and are explicitly flagged as extrapolations. SHAP values are local associations with the fitted model and are not causal effects.
