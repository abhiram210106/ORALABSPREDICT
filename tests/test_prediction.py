"""
test_prediction.py
------------------
End-to-end integration tests for OralAbsPredictor, HIA/HOB inference,
SHAP explainability, and error handling.
"""

import pytest
import numpy as np
from src.predictor import OralAbsPredictor
from src.explainability import explain_prediction, get_global_feature_importance


@pytest.fixture(scope="module")
def predictor():
    """Initializes the OralAbsPredictor once for integration testing."""
    p = OralAbsPredictor(models_dir="models")
    return p


def test_predictor_loaded(predictor):
    """Verify that models and feature names load successfully."""
    assert predictor.is_loaded is True
    assert predictor.hia_model is not None
    assert predictor.hob_model is not None
    assert len(predictor.hia_features) == 706
    assert len(predictor.hob_features) == 706


def test_hia_prediction(predictor):
    """Verify HIA prediction outputs valid classes and probabilities."""
    smi = "CC(=O)Nc1ccccc1"  # Paracetamol core
    res = predictor.predict_hia(smi)

    assert "prediction_class" in res
    assert res["prediction_class"] in [0, 1]
    assert "probability_high" in res
    assert "probability_low" in res
    assert 0.0 <= res["probability_high"] <= 1.0
    assert 0.0 <= res["probability_low"] <= 1.0
    assert abs(res["probability_high"] + res["probability_low"] - 1.0) < 1e-4
    assert res["short_label"] in ["High", "Low"]


def test_hob_prediction(predictor):
    """Verify HOB prediction outputs valid classes and probabilities."""
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    res = predictor.predict_hob(smi)

    assert "prediction_class" in res
    assert res["prediction_class"] in [0, 1]
    assert "probability_high" in res
    assert "probability_low" in res
    assert 0.0 <= res["probability_high"] <= 1.0
    assert 0.0 <= res["probability_low"] <= 1.0
    assert abs(res["probability_high"] + res["probability_low"] - 1.0) < 1e-4
    assert res["short_label"] in ["High", "Low"]


def test_predict_all_complete_pipeline(predictor):
    """Verify predict_all returns all components (2D, 3D, descriptors, predictions)."""
    smi = "CCO"  # Ethanol
    res = predictor.predict_all(smi)

    assert res["success"] is True
    assert res["canonical_smiles"] == "CCO"
    assert res["molecular_formula"] == "C2H6O"
    assert "descriptors" in res
    assert "svg_2d" in res
    assert len(res["svg_2d"]) > 50
    assert res["hia"] is not None
    assert res["hob"] is not None


def test_invalid_smiles_handling(predictor):
    """Verify that unparseable SMILES gracefully returns error without crashing."""
    res = predictor.predict_all("NonChemicalGarbage12345!@#$")
    assert res["success"] is False
    assert "error" in res


def test_missing_model_handling():
    """Verify that predictor handles non-existent model directories gracefully."""
    empty_p = OralAbsPredictor(models_dir="non_existent_models_directory")
    assert empty_p.is_loaded is False

    res = empty_p.predict_all("CCO")
    assert res["success"] is True
    assert "warning" in res
    assert res["hia"] is None


def test_shap_explanation(predictor):
    """Verify SHAP local explanation calculation and table output."""
    smi = "CC(=O)Nc1ccccc1"
    vec = predictor.predict_hia(smi)["feature_vector"]

    exp = explain_prediction(
        model=predictor.hia_model,
        feature_names=predictor.hia_features,
        feature_vector=vec,
        top_n=5
    )

    assert "shap_values" in exp
    assert "positive_features" in exp
    assert "negative_features" in exp
    assert "explanation_table" in exp
    assert "plotly_figure" in exp

    assert len(exp["shap_values"]) == 706
    assert not exp["explanation_table"].empty
