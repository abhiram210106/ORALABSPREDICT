"""
model_comparison.py
-------------------
Rigorous model evaluation, metric calculations, and benchmarking across
the 6 mandated algorithms for OralAbsPredict.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Initializes the 6 project-specified classification models with appropriate
    class balancing and preprocessing pipelines.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "Balanced Random Forest": BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                C=1.0,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state
            ))
        ]),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0
        )
    }
    return models


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculates comprehensive classification metrics suited for imbalanced datasets:
    Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix.
    """
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = float(np.mean(y_true))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Trains all candidate models on (X_train, y_train), evaluates them on (X_val, y_val),
    generates a side-by-side comparison table, and selects the best model based on Balanced Accuracy and F1.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], str]
        (comparison_df, all_evaluations_dict, best_model_name)
    """
    models = get_candidate_models(random_state=random_state)
    comparison_rows = []
    evaluations: Dict[str, Any] = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Probabilities for positive class
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_val)
            else:
                y_prob = None

            metrics = evaluate_predictions(y_val, y_pred, y_prob)
            evaluations[name] = {
                "model": model,
                "metrics": metrics
            }

            comparison_rows.append({
                "Model": name,
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "F1-Score": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
                "Recall (Sensitivity)": metrics["recall"],
                "Precision": metrics["precision"],
                "Accuracy": metrics["accuracy"],
            })
        except Exception as e:
            comparison_rows.append({
                "Model": name,
                "Balanced Accuracy": 0.0,
                "F1-Score": 0.0,
                "ROC-AUC": 0.0,
                "PR-AUC": 0.0,
                "Recall (Sensitivity)": 0.0,
                "Precision": 0.0,
                "Accuracy": 0.0,
                "Error": str(e)
            })

    comparison_df = pd.DataFrame(comparison_rows)
    # Sort primarily by Balanced Accuracy, then F1-Score
    comparison_df = comparison_df.sort_values(
        by=["Balanced Accuracy", "F1-Score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    best_model_name = comparison_df.iloc[0]["Model"]
    return comparison_df, evaluations, best_model_name
