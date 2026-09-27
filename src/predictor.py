"""
predictor.py
------------
End-to-end inference engine for OralAbsPredict. Loads serialized models,
validates chemical structures, calculates descriptors, and predicts HIA and HOB.
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import json
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.preprocessing import validate_smiles
from src.descriptors import (
    calculate_rdkit_descriptors,
    generate_features,
    get_descriptor_table_df
)
from src.visualization import mol_to_2d_svg, generate_3d_conformer, get_3dmol_html


class OralAbsPredictor:
    """
    Unified predictor for Human Intestinal Absorption (HIA) and Human Oral Bioavailability (HOB).
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.hia_model = None
        self.hia_features = None
        self.hia_metadata = None
        self.hob_model = None
        self.hob_features = None
        self.hob_metadata = None
        self.is_loaded = False
        self.load_models()

    def load_models(self) -> bool:
        """
        Loads trained models, feature names, and metadata from disk.
        """
        try:
            hia_model_path = os.path.join(self.models_dir, "hia_model.pkl")
            hia_feat_path = os.path.join(self.models_dir, "hia_feature_names.pkl")
            hia_meta_path = os.path.join(self.models_dir, "metadata", "hia_metadata.json")

            hob_model_path = os.path.join(self.models_dir, "hob_model.pkl")
            hob_feat_path = os.path.join(self.models_dir, "hob_feature_names.pkl")
            hob_meta_path = os.path.join(self.models_dir, "metadata", "hob_metadata.json")

            if not (os.path.exists(hia_model_path) and os.path.exists(hob_model_path)):
                self.is_loaded = False
                return False

            self.hia_model = joblib.load(hia_model_path)
            self.hia_features = joblib.load(hia_feat_path)
            if os.path.exists(hia_meta_path):
                with open(hia_meta_path, "r") as f:
                    self.hia_metadata = json.load(f)

            self.hob_model = joblib.load(hob_model_path)
            self.hob_features = joblib.load(hob_feat_path)
            if os.path.exists(hob_meta_path):
                with open(hob_meta_path, "r") as f:
                    self.hob_metadata = json.load(f)

            self.is_loaded = True
            return True
        except Exception as e:
            self.is_loaded = False
            return False

    def predict_hia(self, smiles: str) -> Dict[str, Any]:
        """
        Predicts Human Intestinal Absorption for a SMILES string.
        """
        if not self.is_loaded or self.hia_model is None:
            raise RuntimeError("HIA model is not loaded. Please train models first.")

        is_valid, can_smiles, mol = validate_smiles(smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")

        X = generate_features(can_smiles, feature_names=self.hia_features).reshape(1, -1)
        pred_class = int(self.hia_model.predict(X)[0])

        prob_high = 0.5
        prob_low = 0.5
        if hasattr(self.hia_model, "predict_proba"):
            probs = self.hia_model.predict_proba(X)[0]
            prob_low = float(probs[0])
            prob_high = float(probs[1])

        label = "High Intestinal Absorption (HIA+)" if pred_class == 1 else "Low / Poor Intestinal Absorption (HIA-)"
        short_label = "High" if pred_class == 1 else "Low"

        # Confidence: distance from decision threshold (0.50)
        confidence = abs(prob_high - 0.5) * 2.0

        return {
            "endpoint": "Human Intestinal Absorption (HIA)",
            "prediction_class": pred_class,
            "label": label,
            "short_label": short_label,
            "probability_high": prob_high,
            "probability_low": prob_low,
            "confidence_score": confidence,
            "confidence_percent": f"{confidence * 100:.1f}%",
            "model_name": self.hia_metadata.get("model_name", "CatBoost") if self.hia_metadata else "CatBoost",
            "feature_vector": X[0]
        }

    def predict_hob(self, smiles: str) -> Dict[str, Any]:
        """
        Predicts Human Oral Bioavailability for a SMILES string.
        """
        if not self.is_loaded or self.hob_model is None:
            raise RuntimeError("HOB model is not loaded. Please train models first.")

        is_valid, can_smiles, mol = validate_smiles(smiles)
        if not is_valid or mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")

        X = generate_features(can_smiles, feature_names=self.hob_features).reshape(1, -1)
        pred_class = int(self.hob_model.predict(X)[0])

        prob_high = 0.5
        prob_low = 0.5
        if hasattr(self.hob_model, "predict_proba"):
            probs = self.hob_model.predict_proba(X)[0]
            prob_low = float(probs[0])
            prob_high = float(probs[1])

        label = "High Oral Bioavailability (HOB+ >= 30%)" if pred_class == 1 else "Low / Poor Oral Bioavailability (HOB- < 30%)"
        short_label = "High" if pred_class == 1 else "Low"

        confidence = abs(prob_high - 0.5) * 2.0

        return {
            "endpoint": "Human Oral Bioavailability (HOB)",
            "prediction_class": pred_class,
            "label": label,
            "short_label": short_label,
            "probability_high": prob_high,
            "probability_low": prob_low,
            "confidence_score": confidence,
            "confidence_percent": f"{confidence * 100:.1f}%",
            "model_name": self.hob_metadata.get("model_name", "CatBoost") if self.hob_metadata else "CatBoost",
            "feature_vector": X[0]
        }

    def predict_all(self, smiles: str) -> Dict[str, Any]:
        """
        Performs full chemical analysis: structure validation, 2D/3D visualization,
        descriptor calculation, and dual HIA/HOB prediction.
        """
        is_valid, can_smiles, mol = validate_smiles(smiles)
        if not is_valid or mol is None:
            return {
                "success": False,
                "input_smiles": smiles,
                "error": "The provided SMILES string is invalid or chemically unparseable by RDKit."
            }

        # Molecular formula and basic info
        mol_formula = rdMolDescriptors.CalcMolFormula(mol)
        descriptors = calculate_rdkit_descriptors(mol)
        desc_table = get_descriptor_table_df(descriptors)

        # Visualizations
        svg_2d = mol_to_2d_svg(mol, width=420, height=320)
        has_3d, mol_block_3d, _ = generate_3d_conformer(mol)
        html_3d = get_3dmol_html(mol_block_3d) if has_3d else None

        result: Dict[str, Any] = {
            "success": True,
            "input_smiles": smiles,
            "canonical_smiles": can_smiles,
            "molecular_formula": mol_formula,
            "descriptors": descriptors,
            "descriptor_table": desc_table,
            "svg_2d": svg_2d,
            "has_3d": has_3d,
            "mol_block_3d": mol_block_3d,
            "html_3d": html_3d,
            "hia": None,
            "hob": None,
        }

        if self.is_loaded:
            try:
                result["hia"] = self.predict_hia(can_smiles)
            except Exception as e:
                result["hia_error"] = str(e)

            try:
                result["hob"] = self.predict_hob(can_smiles)
            except Exception as e:
                result["hob_error"] = str(e)
        else:
            result["warning"] = "Pre-trained models are not loaded. Run train_hia.py and train_hob.py to train models."

        return result
