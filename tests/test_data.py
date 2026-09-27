"""
test_data.py
------------
Unit tests for SMILES validation, canonicalization, column detection, and dataset loading.
"""

import pytest
import pandas as pd
from src.preprocessing import validate_smiles, canonicalize_smiles, clean_dataset
from src.data_loader import detect_columns, load_dataset, split_data


def test_valid_smiles():
    """Verify that standard valid SMILES strings parse correctly."""
    valid, can_smi, mol = validate_smiles("CCO")  # Ethanol
    assert valid is True
    assert can_smi == "CCO"
    assert mol is not None

    valid2, can_smi2, mol2 = validate_smiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
    assert valid2 is True
    assert mol2 is not None


def test_invalid_smiles():
    """Verify that invalid chemical strings return False and None."""
    valid, can_smi, mol = validate_smiles("InvalidChemicalString123!@#")
    assert valid is False
    assert can_smi is None
    assert mol is None

    # Empty string
    valid_empty, _, _ = validate_smiles("")
    assert valid_empty is False

    # None input
    valid_none, _, _ = validate_smiles(None)
    assert valid_none is False


def test_smiles_canonicalization():
    """Verify different representations of the same molecule produce identical canonical SMILES."""
    # Ethanol represented in two different ways
    smi1 = "OCC"
    smi2 = "CCO"
    can1 = canonicalize_smiles(smi1)
    can2 = canonicalize_smiles(smi2)
    assert can1 == can2 == "CCO"


def test_clean_dataset():
    """Verify dataset cleaning drops invalid SMILES and handles duplicates."""
    df_raw = pd.DataFrame({
        "SMILES": ["CCO", "InvalidSMILES", "OCC", "CC(=O)Nc1ccccc1"],
        "Y": [1, 0, 1, 1]
    })
    cleaned, stats = clean_dataset(df_raw, smiles_col="SMILES", target_col="Y")

    assert stats["invalid_smiles_dropped"] == 1
    assert stats["duplicates_dropped"] == 1  # CCO and OCC are duplicates
    assert len(cleaned) == 2


def test_detect_columns():
    """Verify automated column detection across varying headers."""
    df1 = pd.DataFrame({"smiles": ["CCO"], "target": [1]})
    s_col, t_col = detect_columns(df1)
    assert s_col == "smiles"
    assert t_col == "target"

    df2 = pd.DataFrame({"canonical_smiles": ["CCO"], "HIA": [1]})
    s_col2, t_col2 = detect_columns(df2)
    assert s_col2 == "canonical_smiles"
    assert t_col2 == "HIA"


def test_split_data():
    """Verify stratified data splitting produces expected subsets."""
    df = pd.DataFrame({
        "SMILES": [f"C{i}" for i in range(100)],
        "target": [0] * 30 + [1] * 70
    })
    splits = split_data(df, target_col="target", test_size=0.2, val_size=0.1)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits
    assert len(splits["test"]) == 20
    assert len(splits["val"]) == 10
    assert len(splits["train"]) == 70
