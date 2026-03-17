import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" not in df.columns:
        df["customerID"] = "WEB-USER-0001"

    tenure_safe = df["tenure"].replace(0, 1)

    # Monetary and tenure features
    df["avg_charge_per_month"] = df["TotalCharges"] / tenure_safe
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / tenure_safe
    df["high_monthly_charge_flag"] = (df["MonthlyCharges"] > 80).astype(int)
    df["low_tenure_flag"] = (df["tenure"] <= 12).astype(int)
    df["long_tenure_flag"] = (df["tenure"] >= 60).astype(int)

    # Service bundle features
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    yes_like = {
        "PhoneService": ["Yes"],
        "MultipleLines": ["Yes"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["Yes"],
        "TechSupport": ["Yes"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"]
    }

    df["service_count"] = 0
    for col in service_cols:
        df["service_count"] += df[col].isin(yes_like[col]).astype(int)

    support_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["support_service_count"] = sum(df[col].eq("Yes").astype(int) for col in support_cols)

    streaming_cols = ["StreamingTV", "StreamingMovies"]
    df["streaming_service_count"] = sum(df[col].eq("Yes").astype(int) for col in streaming_cols)

    df["internet_dependent_flag"] = df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    df["no_support_bundle_flag"] = (df["support_service_count"] == 0).astype(int)

    # Contract and payment risk features
    contract_risk_map = {
        "Month-to-month": 2,
        "One year": 1,
        "Two year": 0
    }
    payment_risk_map = {
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1
    }

    df["contract_risk_score"] = df["Contract"].map(contract_risk_map)
    df["payment_friction_score"] = df["PaymentMethod"].map(payment_risk_map)

    df["paperless_auto_payment_mismatch"] = (
        (df["PaperlessBilling"] == "Yes") &
        (~df["PaymentMethod"].isin(["Bank transfer (automatic)", "Credit card (automatic)"]))
    ).astype(int)

    df["month_to_month_high_risk_flag"] = (
        (df["Contract"] == "Month-to-month") &
        (df["MonthlyCharges"] > 70)
    ).astype(int)

    # Interactions
    df["tenure_x_monthlycharges"] = df["tenure"] * df["MonthlyCharges"]
    df["internetservice_x_techsupport"] = df["InternetService"].astype(str) + "_" + df["TechSupport"].astype(str)
    df["seniorcitizen_x_dependents"] = df["SeniorCitizen"].astype(str) + "_" + df["Dependents"].astype(str)
    df["contract_x_paymentmethod"] = df["Contract"].astype(str) + "_" + df["PaymentMethod"].astype(str)
    df["support_bundle_x_monthlycharges"] = df["support_service_count"] * df["MonthlyCharges"]

    # Bins
    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    ).astype(str)

    df["monthly_charge_bin"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-np.inf, 35, 70, 90, np.inf],
        labels=["low", "medium", "high", "very_high"]
    ).astype(str)

    df["total_charge_bin"] = pd.cut(
        df["TotalCharges"],
        bins=[-np.inf, 500, 2000, 5000, np.inf],
        labels=["low", "medium", "high", "very_high"]
    ).astype(str)

    return df