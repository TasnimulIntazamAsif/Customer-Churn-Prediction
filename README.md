# ACCURATE-Churn Studio

ACCURATE-Churn Studio is a Streamlit application for running the deployed inference stack from the churn prediction project:

**ACCURATE-Churn: An Actionable, Calibrated, and Uncertainty-Aware Ensemble Framework for Telecom Customer Churn Prediction and Retention Decision Support**

## What the app does

- Predicts final churn probability with the calibrated stacked ensemble
- Applies the final operational threshold to produce a decision label
- Shows uncertainty tiers: `confident_churn`, `uncertain`, `confident_non_churn`
- Displays base-model probabilities
- Shows local SHAP driver summaries
- Simulates retention interventions and ranks them by estimated probability reduction

## Project layout

```text
accurateChurn/
|-- app.py
|-- requirements.txt
|-- README.md
|-- artifacts/
|   |-- metadata.json
|   |-- saved_assets.json
|   |-- models/
|   |-- reports/
|   `-- figures/
`-- src/
    |-- __init__.py
    |-- feature_engineering.py
    `-- predictor.py
```

## Requirements

- Python 3.13 was verified locally
- The repository already includes the required `artifacts/` directory
- Dependencies are pinned in [`requirements.txt`](/e:/myWebsites/accurateChurn/requirements.txt) to match the shipped model files

## Local setup

From the project root in PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Common startup issues

### `streamllit` is not recognized

The command is spelled `streamlit`, not `streamllit`.

If the direct command still is not found, use:

```powershell
python -m streamlit run app.py
```

### The app says the model artifacts or dependencies cannot be loaded

Recreate the environment from scratch:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes

- The final decision threshold is `0.23`
- The uncertainty thresholds are:
  - `confident_churn`: probability `>= 0.70`
  - `confident_non_churn`: probability `<= 0.10`
- The app resolves the `artifacts/` folder relative to `app.py`, so it is less sensitive to where the command is launched from
