# AKI risk calculator — XGBoost

This Streamlit app deploys the locked XGBoost model from the primary analysis. It estimates postoperative AKI probability from nine variables and provides a case-level, probability-scale SHAP explanation.

## Public repository contents

Commit only `app.py`, `model_bundle.joblib`, `requirements.txt`, `runtime.txt`, and `README.md`. Do **not** commit source CSV files, `内部复核_不直接公开`, or row-level predictions. The bundle contains the fitted model/scaler plus aggregate ranges and medians only.

## Local run

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deployment

GitHub versions the source but does not execute Streamlit. Deploy the repository on a Streamlit-compatible hosting service with `app.py` as entry point; cite the service-issued HTTPS URL in the manuscript. Preserve each published bundle with a tagged GitHub release rather than overwriting it.

## Internal rebuilding

`build_bundle.py` is local-only. It rebuilds the locked XGBoost model from the final imputed training set and compares every test-set probability with the private audit record before exporting the bundle.

## Interpretation

This calculator supports research display only and does not replace clinical judgment. Modestly expanded input bounds are allowed, but values outside the observed development-cohort range are flagged as extrapolations. SHAP values describe the fitted model, not causal effects.
