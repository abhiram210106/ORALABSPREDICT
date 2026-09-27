"""
test_models.py
--------------
Unit tests for model persistence, metadata structure, and benchmarking functions.
"""

import os
import json
import pytest
import joblib
import numpy as np
from src.model_comparison import get_candidate_models, evaluate_predictions


def test_candidate_models_initialization():
    """Verify that all 6 required project models are initialized properly."""
    models = get_candidate_models(random_state=42)
    expected_names = [
        "Logistic Regression",
        "Random Forest",
        "Balanced Random Forest",
        "Support Vector Machine",
        "LightGBM",
        "CatBoost"
    ]
    for name in expected_names:
        assert name in models, f"Expected candidate model '{name}' missing from candidate pool."


def test_evaluate_predictions_metrics():
    """Verify classification metrics calculation on synthetic predictions."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.95])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_serialized_models_exist():
    """Verify that serialized model files and feature files exist on disk."""
    assert os.path.exists("models/hia_model.pkl"), "hia_model.pkl not found"
    assert os.path.exists("models/hia_feature_names.pkl"), "hia_feature_names.pkl not found"
    assert os.path.exists("models/hob_model.pkl"), "hob_model.pkl not found"
    assert os.path.exists("models/hob_feature_names.pkl"), "hob_feature_names.pkl not found"


def test_metadata_integrity():
    """Verify metadata JSON files exist and have required schema fields."""
    for endpoint in ["hia", "hob"]:
        meta_path = f"models/metadata/{endpoint}_metadata.json"
        assert os.path.exists(meta_path), f"Metadata file {meta_path} missing"

        with open(meta_path, "r") as f:
            data = json.load(f)

        assert "model_name" in data
        assert "endpoint" in data
        assert "feature_count" in data
        assert "classes" in data
        assert "test_metrics" in data
        assert "validation_comparison" in data
        assert data["feature_count"] == 706
