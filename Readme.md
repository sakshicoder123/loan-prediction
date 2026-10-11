# 🏦 LoanPulse AI — Loan Eligibility & Risk Prediction System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20(Linear)-FF6F00.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, explainable Machine Learning system for automated loan eligibility assessment and credit risk categorization. Built strictly using classical ML algorithms (**Scikit-Learn**) and a lightweight single-layer **TensorFlow** linear classifier with **zero deep learning**.

---

## 🌟 Key Highlights

- **Classical Machine Learning Architecture**: Powered by a robust **Stacking Ensemble** combining Random Forest, Gradient Boosting, and Logistic Regression.
- **Strictly No Deep Learning**: Focuses on interpretable classical ML alongside a lightweight single-layer TensorFlow Generalized Linear Model / Perceptron.
- **Feature Engineering Pipeline**:
  - `TotalIncome`: Combined primary and co-applicant income.
  - `EMI`: Estimated monthly repayment obligation.
  - `Debt_to_Income_Ratio (DTI)`: Solvency & financial strain index.
  - `Income_to_Loan_Ratio`: Repayment solvency buffer.
- **Real-Time Interactive Web Dashboard**:
  - Dynamic risk probability gauge (Approval % vs. Risk Tier).
  - 1-Click applicant presets (Prime, First-Time Buyer, Overleveraged, Adverse Credit).
  - Human-interpretable decision factors and financial breakdowns.
  - Model benchmarking modal with live metrics.
- **REST API Endpoints**: Production-ready endpoints for automated credit evaluations and batch inferences.

---

## 📊 Model Benchmark & Performance

All models were evaluated using 5-Fold Stratified Cross-Validation and an identical holdout test set (20%):

| Model | Framework | Test Accuracy | F1-Score | ROC-AUC | 5-Fold CV Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** ⭐ *(Active)* | **Scikit-Learn** | **86.99%** | **0.9130** | **0.8802** | **78.82%** |
| Random Forest Classifier | Scikit-Learn | 86.99% | 0.9121 | 0.8706 | 78.82% |
| Logistic Regression | Scikit-Learn | 86.18% | 0.9081 | 0.8768 | 79.02% |
| TensorFlow Linear Classifier | TensorFlow (Linear) | 86.18% | 0.9081 | 0.8554 | 86.18% |
| Support Vector Machine (RBF) | Scikit-Learn | 84.55% | 0.8984 | 0.8251 | 79.84% |
| Gradient Boosting | Scikit-Learn | 84.55% | 0.8950 | 0.8554 | 77.80% |
| Decision Tree Classifier | Scikit-Learn | 79.67% | 0.8634 | 0.8186 | 78.00% |
| K-Nearest Neighbors (KNN) | Scikit-Learn | 76.42% | 0.8466 | 0.7983 | 72.92% |

---

## 🗂️ Project Structure

```
loan-prediction/
│
├── dataset/
│   └── Loan dataset.csv              # Raw loan applicants dataset
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── preprocess.py                 # Cleaning, feature engineering & ColumnTransformer
│   ├── models.py                     # Scikit-learn classical models & TF Linear model
│   ├── train.py                      # Training, cross-validation & artifact persistence
│   └── predict.py                    # Inference engine, probability & decision attribution
│
├── models/
│   ├── preprocessor.joblib           # Fitted preprocessor pipeline
│   ├── best_model.joblib             # Serialized best classifier (Stacking Ensemble)
│   ├── tf_linear_model.keras         # Single-layer TensorFlow linear model
│   ├── metrics.json                  # Benchmark evaluation results
│   └── feature_names.json            # Final engineered feature schema
│
├── app/
│   ├── app.py                        # Production Flask Web Server & REST API
│   ├── templates/
│   │   └── index.html                # Modern single-page loan assessment interface
│   └── static/
│       ├── css/style.css             # Glassmorphism styling & light/dark theme system
│       └── js/script.js              # Gauge animation, presets & API integration
│
├── run_pipeline.py                   # One-click execution to retrain models & export artifacts
├── run_app.py                        # One-click script to launch web application locally
├── requirements.txt                  # Minimal clean dependencies
└── README.md                         # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/sakshicoder123/loan-prediction.git
cd loan-prediction
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Libraries:** This project strictly uses `numpy`, `pandas`, `scikit-learn`, `tensorflow`, and `Flask`. No bulky or unstable dependencies are included.

### 3. Run the ML Pipeline (Train & Benchmark)
```bash
python run_pipeline.py
```
This trains all classical Scikit-Learn models and the TensorFlow linear baseline, evaluates metrics via 5-fold CV and test split, and updates the `models/` directory.

### 4. Start the Interactive Web Application
```bash
python run_app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 REST API Specification

### Predict Loan Eligibility
- **Endpoint**: `POST /api/predict`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "applicant_income": 6500,
  "coapplicant_income": 2500,
  "loan_amount": 140,
  "loan_term": 360,
  "credit_history": 1,
  "property_area": "Semiurban",
  "education": "Graduate",
  "married": "Yes",
  "dependents": "1",
  "self_employed": "No",
  "engine": "best"
}
```

- **Response**:
```json
{
  "status": "success",
  "data": {
    "loan_status": "Approved",
    "is_approved": true,
    "approval_probability": 83.1,
    "risk_score": 16.9,
    "risk_level": "Low Risk",
    "model_used": "StackingClassifier",
    "financial_summary": {
      "total_income": 9000.0,
      "loan_amount": 140000.0,
      "monthly_emi": 388.89,
      "dti_ratio": 4.3,
      "term_months": 360
    },
    "decision_factors": [
      {
        "factor": "Credit History",
        "status": "Positive",
        "detail": "Meets credit guidelines (healthy repayment track record)."
      },
      {
        "factor": "Debt-to-Income (DTI)",
        "status": "Positive",
        "detail": "Estimated EMI is 4.3% of monthly income (favorable buffer)."
      }
    ]
  }
}
```

### Retrieve Model Metrics
- **Endpoint**: `GET /api/metrics`
- **Response**: Returns comparative accuracy, F1-scores, ROC-AUC, and cross-validation statistics for all evaluated models.

---

## 🛠️ Tech Stack
- **Data & Computing**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Linear Modeling**: [TensorFlow](https://tensorflow.org/) (Strictly single-layer Generalized Linear Model / Perceptron)
- **Serving & Web UI**: [Flask](https://flask.palletsprojects.com/), Modern Vanilla CSS (Glassmorphism), Vanilla JavaScript

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
