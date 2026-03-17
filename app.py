from pathlib import Path

import pandas as pd
import streamlit as st

from src.predictor import AccurateChurnPredictor, PredictorSetupError

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(
    page_title="Cynthia's ACCURATE-Churn Prediction System",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> AccurateChurnPredictor:
    return AccurateChurnPredictor(ARTIFACTS_DIR)


try:
    predictor = load_predictor()
except PredictorSetupError as exc:
    st.title("Cynthia's ACCURATE-Churn Prediction System")
    st.error("The app could not load its runtime dependencies or model artifacts.")
    st.code(str(exc))
    st.markdown(
        """
Use these commands from the project root:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
"""
    )
    st.stop()

meta = predictor.metadata

st.title(meta["app_title"])
st.caption(meta["project_title"])

st.markdown("Predict churn risk, view uncertainty tier, inspect top drivers, and simulate retention interventions.")

with st.sidebar:
    st.header("Customer Input")

    gender = st.selectbox("Gender", meta["categorical_options"]["gender"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], index=0)
    partner = st.selectbox("Partner", meta["categorical_options"]["Partner"])
    dependents = st.selectbox("Dependents", meta["categorical_options"]["Dependents"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

    phone_service = st.selectbox("Phone Service", meta["categorical_options"]["PhoneService"])
    multiple_lines = st.selectbox("Multiple Lines", meta["categorical_options"]["MultipleLines"])
    internet_service = st.selectbox("Internet Service", meta["categorical_options"]["InternetService"])
    online_security = st.selectbox("Online Security", meta["categorical_options"]["OnlineSecurity"])
    online_backup = st.selectbox("Online Backup", meta["categorical_options"]["OnlineBackup"])
    device_protection = st.selectbox("Device Protection", meta["categorical_options"]["DeviceProtection"])
    tech_support = st.selectbox("Tech Support", meta["categorical_options"]["TechSupport"])
    streaming_tv = st.selectbox("Streaming TV", meta["categorical_options"]["StreamingTV"])
    streaming_movies = st.selectbox("Streaming Movies", meta["categorical_options"]["StreamingMovies"])

    contract = st.selectbox("Contract", meta["categorical_options"]["Contract"])
    paperless_billing = st.selectbox("Paperless Billing", meta["categorical_options"]["PaperlessBilling"])
    payment_method = st.selectbox("Payment Method", meta["categorical_options"]["PaymentMethod"])

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=0.1)
    auto_compute_total = st.checkbox("Auto-compute Total Charges from tenure x monthly charges", value=True)

    if auto_compute_total:
        total_charges = float(tenure * monthly_charges)
        st.caption(f"Computed Total Charges: {total_charges:.2f}")
    else:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(tenure * monthly_charges),
            step=1.0,
        )

    predict_btn = st.button("Run Prediction", type="primary")

if predict_btn:
    user_input = {
        "customerID": "WEB-USER-0001",
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    result = predictor.predict(user_input)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Churn Probability", f"{result['final_probability']:.4f}")
    col2.metric(f"Decision @ Threshold {meta['final_threshold']}", result["decision_label"])
    col3.metric("Uncertainty Tier", result["uncertainty_tier"])

    st.subheader("Base Model Probabilities")
    st.dataframe(pd.DataFrame([result["base_model_probabilities"]]), use_container_width=True)

    st.subheader("Top SHAP Drivers")
    st.dataframe(result["top_drivers"], use_container_width=True)

    st.subheader("Recommended Retention Interventions")
    st.dataframe(result["counterfactuals"], use_container_width=True)

    st.subheader("Model Notes")
    st.markdown(
        f"""
- Final calibrated decision threshold: **{meta['final_threshold']}**
- Confident churn if probability >= **{meta['uncertainty_thresholds']['confident_churn_min']}**
- Confident non-churn if probability <= **{meta['uncertainty_thresholds']['confident_non_churn_max']}**
"""
    )
else:
    st.info("Fill in the customer information from the sidebar and click Run Prediction.")
