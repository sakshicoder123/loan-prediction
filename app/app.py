"""
Loan Prediction Web Application & REST API.
Built with Flask, Scikit-Learn, NumPy, Pandas, and TensorFlow.
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import LoanPredictor

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)

# Initialize predictor singleton
predictor = LoanPredictor(models_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

# Load cached metrics
METRICS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json"))


def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


@app.route("/")
def index():
    metrics_data = load_metrics()
    best_model_name = metrics_data.get("best_model_name", "Stacking Ensemble")
    return render_template(
        "index.html",
        best_model=best_model_name,
        models_count=len(metrics_data.get("models_benchmarked", {}))
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No input payload provided."}), 400

        applicant_data = {
            "Gender": data.get("gender", "Male"),
            "Married": data.get("married", "Yes"),
            "Dependents": str(data.get("dependents", "0")),
            "Education": data.get("education", "Graduate"),
            "Self_Employed": data.get("self_employed", "No"),
            "ApplicantIncome": float(data.get("applicant_income", 4500)),
            "CoapplicantIncome": float(data.get("coapplicant_income", 1500)),
            "LoanAmount": float(data.get("loan_amount", 120)),  # in K
            "Loan_Amount_Term": float(data.get("loan_term", 360)),
            "Credit_History": float(data.get("credit_history", 1)),
            "Property_Area": data.get("property_area", "Semiurban")
        }

        engine = data.get("engine", "best")  # 'best' (sklearn) or 'tensorflow'
        result = predictor.predict(applicant_data, engine=engine)

        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics_data = load_metrics()
    return jsonify({
        "status": "success",
        "data": metrics_data
    })


@app.route("/api/samples", methods=["GET"])
def api_samples():
    samples = [
        {
            "id": "prime",
            "title": "Prime Applicant (Low Risk)",
            "description": "High income, co-applicant support, stellar credit history.",
            "data": {
                "gender": "Male",
                "married": "Yes",
                "dependents": "1",
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 6500,
                "coapplicant_income": 2500,
                "loan_amount": 140,
                "loan_term": 360,
                "credit_history": 1,
                "property_area": "Semiurban"
            }
        },
        {
            "id": "starter",
            "title": "First-Time Buyer (Moderate Risk)",
            "description": "Moderate earnings, good credit, standard starter loan.",
            "data": {
                "gender": "Female",
                "married": "No",
                "dependents": "0",
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 4200,
                "coapplicant_income": 0,
                "loan_amount": 110,
                "loan_term": 360,
                "credit_history": 1,
                "property_area": "Urban"
            }
        },
        {
            "id": "overleveraged",
            "title": "High Debt Burden (High Risk)",
            "description": "Low income requesting a heavy loan with no coapplicant.",
            "data": {
                "gender": "Male",
                "married": "Yes",
                "dependents": "3+",
                "education": "Not Graduate",
                "self_employed": "Yes",
                "applicant_income": 2200,
                "coapplicant_income": 0,
                "loan_amount": 250,
                "loan_term": 180,
                "credit_history": 1,
                "property_area": "Rural"
            }
        },
        {
            "id": "adverse_credit",
            "title": "Adverse Credit History (Rejected)",
            "description": "Decent income but non-compliant past credit track record.",
            "data": {
                "gender": "Male",
                "married": "No",
                "dependents": "0",
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 5800,
                "coapplicant_income": 1200,
                "loan_amount": 130,
                "loan_term": 360,
                "credit_history": 0,
                "property_area": "Urban"
            }
        }
    ]
    return jsonify({
        "status": "success",
        "samples": samples
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Loan Prediction Web Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
