import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabular import TabularModel

from .feature_engineering import add_engineered_features

REQUIRED_SKLEARN_VERSION = "1.6.1"


class PredictorSetupError(RuntimeError):
    """Raised when the local runtime cannot safely load the shipped artifacts."""


class AccurateChurnPredictor:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.models_dir = self.artifacts_dir / "models"

        self._validate_runtime()
        self.metadata = self._load_json("metadata.json")
        self.saved_assets = self._load_json("saved_assets.json")
        self._validate_artifacts()

        self.raw_input_columns = self.metadata["raw_input_columns"]
        self.final_categorical_cols = self.metadata["final_categorical_cols"]
        self.final_numeric_cols = self.metadata["final_numeric_cols"]
        self.cluster_features = self.metadata["cluster_features"]
        self.final_threshold = float(self.metadata["final_threshold"])
        self.confident_churn_min = float(self.metadata["uncertainty_thresholds"]["confident_churn_min"])
        self.confident_non_churn_max = float(self.metadata["uncertainty_thresholds"]["confident_non_churn_max"])

        # Core models
        self.xgb_pipeline = joblib.load(self._asset_path("xgb_pipeline"))
        self.lgb_pipeline = joblib.load(self._asset_path("lgb_pipeline"))
        self.stack_meta = joblib.load(self._asset_path("stack_meta"))
        self.iso_calibrator = joblib.load(self._asset_path("iso_calibrator"))

        # Preprocessing assets
        self.cluster_imputer = joblib.load(self._asset_path("cluster_imputer"))
        self.cluster_scaler = joblib.load(self._asset_path("cluster_scaler"))
        self.kmeans = joblib.load(self._asset_path("kmeans"))

        self.num_imputer_tabnet = joblib.load(self._asset_path("num_imputer_tabnet"))
        self.cat_imputer_tabnet = joblib.load(self._asset_path("cat_imputer_tabnet"))
        self.label_encoders_tabnet = joblib.load(self._asset_path("label_encoders_tabnet"))
        self.ft_numeric_imputer = joblib.load(self._asset_path("ft_numeric_imputer"))

        # CatBoost
        self.cat_model = CatBoostClassifier()
        self.cat_model.load_model(str(self._asset_path("catboost_explainer")))
        self.shap_explainer = shap.TreeExplainer(self.cat_model)

        # TabNet
        self.tabnet_model = TabNetClassifier()
        self.tabnet_model.load_model(str(self._asset_path("tabnet_model")))

        # FT-Transformer
        self.ft_model = TabularModel.load_model(str(self._asset_path("ft_transformer_model")))

    def _validate_runtime(self) -> None:
        if sklearn.__version__ != REQUIRED_SKLEARN_VERSION:
            raise PredictorSetupError(
                "This project ships scikit-learn model artifacts that require "
                f"scikit-learn {REQUIRED_SKLEARN_VERSION}, but found {sklearn.__version__}. "
                "Reinstall dependencies with `pip install -r requirements.txt`."
            )

    def _load_json(self, filename: str) -> dict:
        path = self.artifacts_dir / filename
        if not path.exists():
            raise PredictorSetupError(
                f"Missing required artifact file: {path}. "
                "Make sure the repository includes the full `artifacts/` directory."
            )

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _asset_path(self, asset_key: str) -> Path:
        try:
            relative_path = self.saved_assets[asset_key]
        except KeyError as exc:
            raise PredictorSetupError(
                f"`saved_assets.json` is missing the `{asset_key}` entry."
            ) from exc

        path = self.models_dir / relative_path
        if not path.exists():
            raise PredictorSetupError(
                f"Missing required model asset: {path}. "
                "Reinstall or restore the full export bundle before running the app."
            )
        return path

    def _validate_artifacts(self) -> None:
        if not self.artifacts_dir.exists():
            raise PredictorSetupError(
                f"Artifacts directory not found: {self.artifacts_dir}"
            )

        if not self.models_dir.exists():
            raise PredictorSetupError(
                f"Models directory not found: {self.models_dir}"
            )

        required_metadata_keys = [
            "raw_input_columns",
            "final_categorical_cols",
            "final_numeric_cols",
            "cluster_features",
            "final_threshold",
            "uncertainty_thresholds",
            "interventions",
        ]
        missing_keys = [key for key in required_metadata_keys if key not in self.metadata]
        if missing_keys:
            raise PredictorSetupError(
                "metadata.json is missing required keys: " + ", ".join(missing_keys)
            )

        required_assets = [
            "xgb_pipeline",
            "lgb_pipeline",
            "stack_meta",
            "iso_calibrator",
            "cluster_imputer",
            "cluster_scaler",
            "kmeans",
            "num_imputer_tabnet",
            "cat_imputer_tabnet",
            "label_encoders_tabnet",
            "ft_numeric_imputer",
            "catboost_explainer",
            "tabnet_model",
            "ft_transformer_model",
        ]
        for asset_key in required_assets:
            self._asset_path(asset_key)

    def _build_raw_df(self, user_input: dict) -> pd.DataFrame:
        row = {col: user_input.get(col) for col in self.raw_input_columns}
        if row["customerID"] is None:
            row["customerID"] = "WEB-USER-0001"
        return pd.DataFrame([row], columns=self.raw_input_columns)

    def _assign_segment(self, df_fe: pd.DataFrame) -> pd.DataFrame:
        df_fe = df_fe.copy()
        cluster_num = self.cluster_imputer.transform(df_fe[self.cluster_features])
        cluster_scaled = self.cluster_scaler.transform(cluster_num)
        df_fe["customer_segment_kmeans"] = self.kmeans.predict(cluster_scaled)
        return df_fe

    def _prepare_model_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df_fe = add_engineered_features(raw_df)
        df_fe = self._assign_segment(df_fe)
        model_df = df_fe.drop(columns=["customerID"]).copy()
        return model_df

    def _catboost_prob(self, model_df: pd.DataFrame) -> np.ndarray:
        df = model_df.copy()
        for col in self.final_categorical_cols:
            df[col] = df[col].astype(str)
        for col in self.final_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return self.cat_model.predict_proba(df)[:, 1]

    def _tabnet_prob(self, model_df: pd.DataFrame) -> np.ndarray:
        case_num = pd.DataFrame(
            self.num_imputer_tabnet.transform(model_df[self.final_numeric_cols]),
            columns=self.final_numeric_cols,
            index=model_df.index,
        )
        case_cat = pd.DataFrame(
            self.cat_imputer_tabnet.transform(model_df[self.final_categorical_cols]),
            columns=self.final_categorical_cols,
            index=model_df.index,
        )

        for col in self.final_categorical_cols:
            le = self.label_encoders_tabnet[col]
            case_cat[col] = le.transform(case_cat[col].astype(str))

        case_tab = pd.concat([case_num, case_cat], axis=1)
        case_tab = case_tab.astype(np.float32)
        return self.tabnet_model.predict_proba(case_tab.values)[:, 1]

    def _ft_prob(self, model_df: pd.DataFrame) -> np.ndarray:
        df = model_df.copy()
        df["Churn_Binary"] = 0
        df[self.final_numeric_cols] = self.ft_numeric_imputer.transform(df[self.final_numeric_cols])

        pred_df = self.ft_model.predict(df)

        prob_col_candidates = [
            "Churn_Binary_1_probability",
            "1_probability",
            "probability_1",
            "prediction_probability",
            "Churn_Binary_probability",
            "Churn_Binary_1_probs",
            "1_probabilities",
        ]

        for candidate in prob_col_candidates:
            if candidate in pred_df.columns:
                return pred_df[candidate].values.astype(float)

        numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_numeric = [
            col for col in numeric_cols if col.lower() not in ["prediction", "predictions", "churn_binary"]
        ]
        if not candidate_numeric:
            raise ValueError("Could not locate FT-Transformer probability column.")
        return pred_df[candidate_numeric[-1]].values.astype(float)

    def base_model_probabilities(self, user_input: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raw_df = self._build_raw_df(user_input)
        model_df = self._prepare_model_df(raw_df)

        probs = pd.DataFrame(
            {
                "catboost": self._catboost_prob(model_df),
                "xgboost": self.xgb_pipeline.predict_proba(model_df)[:, 1],
                "lightgbm": self.lgb_pipeline.predict_proba(model_df)[:, 1],
                "tabnet": self._tabnet_prob(model_df),
                "ft_transformer": self._ft_prob(model_df),
            }
        )

        return raw_df, model_df, probs

    def _final_probability_from_stack(self, stack_df: pd.DataFrame) -> float:
        return float(self.iso_calibrator.predict_proba(stack_df)[:, 1][0])

    def _uncertainty_tier(self, probability: float) -> str:
        if probability >= self.confident_churn_min:
            return "confident_churn"
        if probability <= self.confident_non_churn_max:
            return "confident_non_churn"
        return "uncertain"

    def _decision_label(self, probability: float) -> str:
        return "Likely Churn" if probability >= self.final_threshold else "Likely Non-Churn"

    def _local_shap_table(self, model_df: pd.DataFrame) -> pd.DataFrame:
        case_df = model_df.copy()
        for col in self.final_categorical_cols:
            case_df[col] = case_df[col].astype(str)
        shap_vals = self.shap_explainer.shap_values(case_df)
        shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

        row = pd.DataFrame(
            {
                "feature": case_df.columns,
                "value": case_df.iloc[0].values,
                "shap_value": shap_vals[0],
            }
        ).sort_values("shap_value", ascending=False)

        return row.head(10).reset_index(drop=True)

    def _apply_intervention(self, raw_df: pd.DataFrame, intervention_name: str) -> pd.DataFrame:
        mod = raw_df.copy()

        if intervention_name == "contract_to_one_year":
            mod["Contract"] = "One year"
        elif intervention_name == "contract_to_two_year":
            mod["Contract"] = "Two year"
        elif intervention_name == "add_tech_support":
            if mod.iloc[0]["InternetService"] in ["DSL", "Fiber optic"]:
                mod["TechSupport"] = "Yes"
        elif intervention_name == "add_online_security":
            if mod.iloc[0]["InternetService"] in ["DSL", "Fiber optic"]:
                mod["OnlineSecurity"] = "Yes"
        elif intervention_name == "switch_payment_auto":
            mod["PaymentMethod"] = "Bank transfer (automatic)"
        elif intervention_name == "reduce_monthly_charge_10pct":
            mod["MonthlyCharges"] = mod["MonthlyCharges"] * 0.90
            mod["TotalCharges"] = mod["TotalCharges"] * 0.90
        elif intervention_name == "bundle_support_security":
            if mod.iloc[0]["InternetService"] in ["DSL", "Fiber optic"]:
                mod["TechSupport"] = "Yes"
                mod["OnlineSecurity"] = "Yes"
        else:
            raise ValueError(f"Unknown intervention: {intervention_name}")

        return mod

    def _recommend_interventions(self, raw_df: pd.DataFrame, baseline_prob: float) -> pd.DataFrame:
        rows = []
        for intervention in self.metadata["interventions"]:
            mod_raw = self._apply_intervention(raw_df, intervention)
            _, _, stack_df = self.base_model_probabilities(mod_raw.iloc[0].to_dict())
            new_prob = self._final_probability_from_stack(stack_df)
            rows.append(
                {
                    "intervention": intervention,
                    "baseline_probability": baseline_prob,
                    "new_probability": new_prob,
                    "probability_reduction": baseline_prob - new_prob,
                }
            )

        return pd.DataFrame(rows).sort_values("probability_reduction", ascending=False).reset_index(drop=True)

    def predict(self, user_input: dict) -> dict:
        raw_df, model_df, stack_df = self.base_model_probabilities(user_input)
        final_prob = self._final_probability_from_stack(stack_df)

        return {
            "final_probability": final_prob,
            "decision_label": self._decision_label(final_prob),
            "uncertainty_tier": self._uncertainty_tier(final_prob),
            "base_model_probabilities": stack_df.iloc[0].to_dict(),
            "top_drivers": self._local_shap_table(model_df),
            "counterfactuals": self._recommend_interventions(raw_df, final_prob),
        }
