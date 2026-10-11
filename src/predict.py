"""
Inference and Prediction Engine Module.
Loads trained artifacts and provides comprehensive predictions, risk scores,
and human-readable decision factors for loan applicants.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.preprocess import clean_raw_data, add_engineered_features


class LoanPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        self.best_model_path = os.path.join(models_dir, "best_model.joblib")
        self.tf_model_path = os.path.join(models_dir, "tf_linear_model.keras")

        self.preprocessor = None
        self.best_model = None
        self.tf_model = None

        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = joblib.load(self.preprocessor_path)
        if os.path.exists(self.best_model_path):
            self.best_model = joblib.load(self.best_model_path)

        if os.path.exists(self.tf_model_path):
            try:
                import tensorflow as tf
                self.tf_model = tf.keras.models.load_model(self.tf_model_path)
            except Exception:
                self.tf_model = None

    def predict(self, applicant_data: Dict[str, Any], engine: str = "best") -> Dict[str, Any]:
        """
        Predicts loan approval for an applicant dictionary.
        Returns prediction status, probability, risk level, and explanation breakdown.
        """
        if self.preprocessor is None or self.best_model is None:
            raise RuntimeError("Models are not trained yet. Run `python run_pipeline.py` first.")

        # Prepare DataFrame
        df_raw = pd.DataFrame([applicant_data])
        df_cleaned = clean_raw_data(df_raw)
        df_featured = add_engineered_features(df_cleaned)

        # Transform features
        X_proc = self.preprocessor.transform(df_featured)

        # Make prediction based on chosen engine
        if engine == "tensorflow" and self.tf_model is not None:
            raw_prob = float(self.tf_model.predict(X_proc, verbose=0).flatten()[0])
            pred_class = int(raw_prob >= 0.5)
            approval_prob = round(raw_prob * 100, 1)
            model_used = "TensorFlow Linear Classifier"
        else:
            if hasattr(self.best_model, "predict_proba"):
                probs = self.best_model.predict_proba(X_proc)[0]
                pred_class = int(np.argmax(probs))
                approval_prob = round(float(probs[1]) * 100, 1)
            else:
                pred_class = int(self.best_model.predict(X_proc)[0])
                approval_prob = 100.0 if pred_class == 1 else 0.0
            model_used = type(self.best_model).__name__

        # Financial metrics for explainability
        applicant_income = float(applicant_data.get("ApplicantIncome", 0))
        coapplicant_income = float(applicant_data.get("CoapplicantIncome", 0))
        total_income = applicant_income + coapplicant_income
        loan_amount_k = float(applicant_data.get("LoanAmount", 0))
        loan_amount_dollars = loan_amount_k * 1000.0
        term_months = float(applicant_data.get("Loan_Amount_Term", 360))
        credit_history = float(applicant_data.get("Credit_History", 1))

        monthly_emi = round(loan_amount_dollars / max(term_months, 1), 2)
        dti_pct = round((monthly_emi / max(total_income, 1)) * 100, 1)

        # Risk categorization
        if approval_prob >= 75.0:
            risk_level = "Low Risk"
            risk_color = "#10b981"  # Emerald green
        elif approval_prob >= 50.0:
            risk_level = "Moderate Risk"
            risk_color = "#f59e0b"  # Amber
        else:
            risk_level = "High Risk"
            risk_color = "#ef4444"  # Red

        # Decision breakdown factors
        factors = []
        if credit_history == 1.0 or str(credit_history).strip() in ['1', '1.0', 'Good']:
            factors.append({
                "factor": "Credit History",
                "status": "Positive",
                "detail": "Meets credit guidelines (healthy repayment track record)."
            })
        else:
            factors.append({
                "factor": "Credit History",
                "status": "Negative",
                "detail": "Credit guidelines not met or adverse credit history detected."
            })

        if dti_pct <= 25.0:
            factors.append({
                "factor": "Debt-to-Income (DTI)",
                "status": "Positive",
                "detail": f"Estimated EMI is {dti_pct}% of monthly income (favorable buffer)."
            })
        elif dti_pct <= 45.0:
            factors.append({
                "factor": "Debt-to-Income (DTI)",
                "status": "Neutral",
                "detail": f"Estimated EMI is {dti_pct}% of monthly income (acceptable range)."
            })
        else:
            factors.append({
                "factor": "Debt-to-Income (DTI)",
                "status": "Negative",
                "detail": f"Estimated EMI is {dti_pct}% of monthly income (strained cash flow)."
            })

        if coapplicant_income > 0:
            factors.append({
                "factor": "Co-Applicant Support",
                "status": "Positive",
                "detail": f"Secondary income of ${coapplicant_income:,.0f}/mo boosts household solvency."
            })
        else:
            factors.append({
                "factor": "Co-Applicant Support",
                "status": "Neutral",
                "detail": "Sole applicant responsible for loan obligations."
            })

        return {
            "loan_status": "Approved" if pred_class == 1 else "Rejected",
            "is_approved": bool(pred_class == 1),
            "approval_probability": approval_prob,
            "risk_score": round(100.0 - approval_prob, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "model_used": model_used,
            "financial_summary": {
                "total_income": total_income,
                "loan_amount": loan_amount_dollars,
                "monthly_emi": monthly_emi,
                "dti_ratio": dti_pct,
                "term_months": int(term_months)
            },
            "decision_factors": factors
        }
