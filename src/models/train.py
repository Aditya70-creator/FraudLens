import json
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.generate_data import FEATURES

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(transformers=[("num", numeric_transformer, FEATURES)])

def tune_cost_sensitive_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    # In fraud detection, missing a fraud (False Negative) is ~5x more costly than a False Positive
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    costs = []
    for p, r, th in zip(precisions[:-1], recalls[:-1], thresholds):
        # Custom cost function: 5 * (False Negatives) + 1 * (False Positives)
        fn = (y_true == 1).sum() - (r * (y_true == 1).sum())
        fp = ((y_true == 0).sum()) * (1 - p) # approximate
        cost = (5 * fn) + (1 * fp)
        costs.append(cost)
    
    if len(costs) > 0:
        best_idx = int(np.argmin(costs))
        return float(thresholds[best_idx])
    return 0.5

def train_and_evaluate():
    df = pd.read_csv("data/raw/transactions_reference.csv").drop_duplicates()
    X = df[FEATURES]
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidate_models = {
        "LogisticRegression_Baseline": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("FraudLens_Risk_Intelligence")

    leaderboard = {}
    best_pr_auc = -1.0
    best_pipeline = None
    best_name = ""
    best_threshold = 0.5

    for name, clf in candidate_models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ])
            pipeline.fit(X_train, y_train)

            y_probs = pipeline.predict_proba(X_test)[:, 1]
            threshold = tune_cost_sensitive_threshold(y_test.values, y_probs)
            y_pred = (y_probs >= threshold).astype(int)

            metrics = {
                "pr_auc": round(float(average_precision_score(y_test, y_probs)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "optimal_threshold": round(threshold, 4),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }

            mlflow.log_param("model_name", name)
            mlflow.log_param("optimal_threshold", metrics["optimal_threshold"])
            for k in ["pr_auc", "f1_score", "precision", "recall"]:
                mlflow.log_metric(k, metrics[k])

            leaderboard[name] = metrics
            print(f"[{name}] PR-AUC: {metrics['pr_auc']} | F1: {metrics['f1_score']} | Recall: {metrics['recall']}")

            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_pipeline = pipeline
                best_name = name
                best_threshold = metrics["optimal_threshold"]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/fraud_pipeline.joblib")

    metadata = {
        "winning_model": best_name,
        "optimal_threshold": best_threshold,
        "features": FEATURES,
        "leaderboard": leaderboard,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[WINNER] {best_name} saved to models/fraud_pipeline.joblib (Threshold: {best_threshold})")

if __name__ == "__main__":
    train_and_evaluate()