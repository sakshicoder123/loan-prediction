"""
Model Training and Evaluation Pipeline.
Evaluates classical Scikit-Learn classifiers and a lightweight TensorFlow Linear Classifier.
Saves model artifacts, preprocessing pipeline, and benchmark metrics.
"""

import os
import sys
import json
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

from src.preprocess import prepare_dataset
from src.models import get_sklearn_models, build_tf_linear_model


def evaluate_model(model, X_test, y_test, is_tf: bool = False):
    """
    Computes standard classification evaluation metrics.
    """
    if is_tf:
        y_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = float(acc)

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm
    }


def train_and_evaluate(dataset_path: str = "dataset/Loan dataset.csv", models_dir: str = "models"):
    """
    Runs the complete training and benchmarking suite.
    """
    os.makedirs(models_dir, exist_ok=True)
    print("=" * 70)
    print("  LOAN PREDICTION: PIPELINE TRAINING & BENCHMARKING")
    print("=" * 70)

    # 1. Prepare data
    print("\n[1/5] Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names, X_train_raw, X_test_raw = prepare_dataset(dataset_path)
    print(f"      - Training samples : {X_train.shape[0]}")
    print(f"      - Testing samples  : {X_test.shape[0]}")
    print(f"      - Feature count    : {X_train.shape[1]}")

    results = {}
    fitted_models = {}

    # 2. Train and evaluate Scikit-Learn Classical ML models
    print("\n[2/5] Training Scikit-Learn Classical ML Models...")
    sklearn_models = get_sklearn_models()

    for name, model in sklearn_models.items():
        print(f"      -> Fitting {name}...")
        # 5-fold cross validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Evaluate on holdout test set
        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_accuracy_mean"] = round(cv_mean, 4)
        metrics["cv_accuracy_std"] = round(cv_std, 4)
        metrics["type"] = "Scikit-Learn (Classical ML)"
        results[name] = metrics

    # 3. Train TensorFlow Linear Classifier (strictly single-layer linear model)
    print("\n[3/5] Training TensorFlow Linear Classifier (No Deep Learning)...")
    try:
        import tensorflow as tf
        tf_linear = build_tf_linear_model(input_dim=X_train.shape[1])
        if tf_linear is not None:
            tf_linear.fit(
                X_train, y_train,
                epochs=60,
                batch_size=16,
                validation_split=0.15,
                verbose=0
            )
            tf_metrics = evaluate_model(tf_linear, X_test, y_test, is_tf=True)
            tf_metrics["cv_accuracy_mean"] = tf_metrics["accuracy"]
            tf_metrics["cv_accuracy_std"] = 0.0
            tf_metrics["type"] = "TensorFlow (Linear / Logistic Regressor)"
            results["TensorFlow Linear Classifier"] = tf_metrics

            tf_model_path = os.path.join(models_dir, "tf_linear_model.keras")
            tf_linear.save(tf_model_path)
            print(f"      Saved TensorFlow Linear model -> {tf_model_path}")
    except Exception as e:
        print(f"      TensorFlow linear training skipped or error: {e}")

    # 4. Model Benchmarking Summary Table
    print("\n[4/5] Model Benchmark Performance Table:")
    print("-" * 75)
    print(f"{'Model Name':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'CV Accuracy'}")
    print("-" * 75)
    for model_name, m in results.items():
        print(f"{model_name:<30} | {m['accuracy']:<10.4f} | {m['f1_score']:<10.4f} | {m['roc_auc']:<10.4f} | {m['cv_accuracy_mean']:.4f}")
    print("-" * 75)

    # 5. Select Best Classical Model based on F1-Score & Accuracy
    best_model_name = max(
        [k for k in results.keys() if k in fitted_models],
        key=lambda k: (results[k]['f1_score'] + results[k]['accuracy']) / 2
    )
    best_model = fitted_models[best_model_name]
    print(f"\n[5/5] Best Classical Model: '{best_model_name}' (Accuracy: {results[best_model_name]['accuracy']:.4f}, F1: {results[best_model_name]['f1_score']:.4f})")

    # Serialize artifacts
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    feature_names_path = os.path.join(models_dir, "feature_names.json")

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(best_model, best_model_path)

    metadata = {
        "best_model_name": best_model_name,
        "dataset_rows": int(len(X_train) + len(X_test)),
        "models_benchmarked": results,
        "features": feature_names
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(f"      Saved Preprocessor -> {preprocessor_path}")
    print(f"      Saved Best Model   -> {best_model_path}")
    print(f"      Saved Metrics      -> {metrics_path}")
    print("\nTraining completed successfully!\n")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
