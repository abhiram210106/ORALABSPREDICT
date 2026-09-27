"""
train_hia.py
------------
End-to-end training and evaluation pipeline for Human Intestinal Absorption (HIA).
Trains candidate models, compares metrics on validation set, selects the best model,
evaluates on test set, and serializes the model with complete metadata.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from src.data_loader import load_dataset, split_data
from src.descriptors import extract_all_features, generate_features
from src.model_comparison import compare_models, evaluate_predictions


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    n_bits: int = 512
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts features for all SMILES strings in a DataFrame.
    """
    from typing import List, Tuple
    first_smi = df[smiles_col].iloc[0]
    _, feature_names = extract_all_features(first_smi, n_bits=n_bits)

    feature_list = []
    valid_indices = []

    for idx, smi in enumerate(df[smiles_col]):
        try:
            vec = generate_features(str(smi), feature_names=feature_names, n_bits=n_bits)
            feature_list.append(vec)
            valid_indices.append(idx)
        except Exception:
            continue

    X = np.array(feature_list, dtype=np.float32)
    return X, feature_names


def train_hia_pipeline(
    data_path: str = "data/raw/hia_hou.csv",
    output_dir: str = "models",
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.15,
    n_bits: int = 512
) -> Dict[str, Any]:
    """
    Executes the complete HIA training workflow:
    1. Loads and cleans HIA dataset
    2. Builds molecular feature matrix (Physicochemical + Mordred + Morgan FP)
    3. Stratified splitting (Train / Val / Test)
    4. Evaluates all 6 candidate algorithms
    5. Selects best model based on Balanced Accuracy & F1
    6. Evaluates on unseen Test set
    7. Serializes best model, scaler, feature names, and metadata

    Returns
    -------
    Dict[str, Any]
        Summary containing test metrics and model paths.
    """
    print("=" * 65)
    print("ORALABSPREDICT — HUMAN INTESTINAL ABSORPTION (HIA) TRAINING")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"HIA dataset not found at '{data_path}'. "
            f"Please verify that the dataset exists or place it into data/raw/."
        )

    # 1. Load and clean
    print(f"Loading dataset from: {data_path}")
    df, smiles_col, target_col, meta = load_dataset(data_path, clean=True)
    print(f"Loaded {len(df)} validated compounds.")
    print(f"Class distribution: {meta.get('class_distribution', {})}")

    # 2. Extract features
    print(f"Extracting molecular descriptors and {n_bits}-bit Morgan fingerprints...")
    first_smi = df[smiles_col].iloc[0]
    _, feature_names = extract_all_features(first_smi, n_bits=n_bits)
    print(f"Total features per molecule: {len(feature_names)}")

    X_list = []
    y_list = []
    valid_smiles = []

    for smi, y_val in zip(df[smiles_col], df[target_col]):
        try:
            vec = generate_features(str(smi), feature_names=feature_names, n_bits=n_bits)
            X_list.append(vec)
            y_list.append(int(y_val))
            valid_smiles.append(smi)
        except Exception as e:
            continue

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Feature matrix shape: {X.shape}")

    # Save processed features for quick reuse
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/hia_features.npz", X=X, y=y, smiles=valid_smiles)

    # 3. Stratified Split
    print("Performing stratified Train / Val / Test split...")
    processed_df = pd.DataFrame({"index": np.arange(len(y)), "target": y})
    splits = split_data(
        processed_df,
        target_col="target",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    train_idx = splits["train"]["index"].values
    val_idx = splits["val"]["index"].values
    test_idx = splits["test"]["index"].values

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Compare all 6 candidate models on Validation set
    print("\nComparing 6 candidate machine learning models on Validation set...")
    comparison_df, evaluations, best_model_name = compare_models(
        X_train, y_train, X_val, y_val, random_state=random_state
    )

    print("\nValidation Comparison Results:")
    print(comparison_df.to_string(index=False))
    print(f"\n--> Selected Best Model: {best_model_name}")

    best_model_obj = evaluations[best_model_name]["model"]

    # 5. Evaluate on unseen Test set
    print("\nEvaluating best model on untouched Test set...")
    # Refit best model on Train + Val combined for maximum statistical power
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    best_model_obj.fit(X_train_val, y_train_val)

    y_test_pred = best_model_obj.predict(X_test)
    y_test_prob = None
    if hasattr(best_model_obj, "predict_proba"):
        y_test_prob = best_model_obj.predict_proba(X_test)[:, 1]
    elif hasattr(best_model_obj, "decision_function"):
        y_test_prob = best_model_obj.decision_function(X_test)

    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

    print("\nFinal Test Set Performance:")
    print(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {test_metrics['precision']:.4f}")
    print(f"  Recall:            {test_metrics['recall']:.4f}")
    print(f"  F1-Score:          {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:           {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:            {test_metrics['pr_auc']:.4f}")
    print(f"  Confusion Matrix:  {test_metrics['confusion_matrix']}")

    # 6. Serialization
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    model_path = os.path.join(output_dir, "hia_model.pkl")
    features_path = os.path.join(output_dir, "hia_feature_names.pkl")
    meta_path = os.path.join(output_dir, "metadata", "hia_metadata.json")

    joblib.dump(best_model_obj, model_path)
    joblib.dump(feature_names, features_path)

    metadata = {
        "model_name": best_model_name,
        "endpoint": "Human Intestinal Absorption (HIA)",
        "dataset_source": "Hou et al. (TDC Benchmark HIA_Hou)",
        "training_date": datetime.now().isoformat(),
        "random_state": random_state,
        "feature_count": len(feature_names),
        "target_column": target_col,
        "classes": {
            "0": "Low / Poor Intestinal Absorption (HIA-)",
            "1": "High Intestinal Absorption (HIA+)"
        },
        "class_distribution_raw": meta.get("class_distribution", {}),
        "dataset_size": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": test_metrics,
        "validation_comparison": comparison_df.to_dict(orient="records"),
        "model_file": "hia_model.pkl",
        "feature_names_file": "hia_feature_names.pkl"
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nModel saved to: {model_path}")
    print(f"Feature names saved to: {features_path}")
    print(f"Metadata saved to: {meta_path}")
    print("=" * 65)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HIA classification model")
    parser.add_argument("--data", type=str, default="data/raw/hia_hou.csv", help="Path to HIA dataset CSV")
    parser.add_argument("--output", type=str, default="models", help="Output directory for serialized models")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    train_hia_pipeline(
        data_path=args.data,
        output_dir=args.output,
        random_state=args.seed
    )
