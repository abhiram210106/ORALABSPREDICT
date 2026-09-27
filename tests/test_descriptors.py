"""
test_descriptors.py
-------------------
Unit tests for descriptor generation, fingerprint calculation, and feature vector alignment.
"""

import pytest
import numpy as np
from src.descriptors import (
    calculate_rdkit_descriptors,
    calculate_mordred_subset,
    extract_all_features,
    generate_features,
    get_descriptor_table_df
)
from src.fingerprints import calculate_morgan_fingerprint, calculate_maccs_keys, get_morgan_bit_info


def test_rdkit_descriptors():
    """Verify standard RDKit descriptors compute properly with expected values."""
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    desc = calculate_rdkit_descriptors(smi)

    assert isinstance(desc, dict)
    assert len(desc) == 18
    assert "MolWt" in desc
    assert "LogP" in desc
    assert "TPSA" in desc
    assert "Ro5_Violations" in desc

    # Scientific sanity checks for Aspirin
    assert 170.0 < desc["MolWt"] < 190.0
    assert 0.5 < desc["LogP"] < 2.0
    assert 50.0 < desc["TPSA"] < 75.0
    assert desc["Ro5_Violations"] == 0.0


def test_morgan_fingerprints():
    """Verify Morgan fingerprints generate expected bit vector shape and values."""
    smi = "CCO"  # Ethanol
    fp = calculate_morgan_fingerprint(smi, radius=2, n_bits=512)

    assert isinstance(fp, np.ndarray)
    assert fp.shape == (512,)
    assert set(np.unique(fp)).issubset({0, 1})
    assert fp.sum() > 0


def test_maccs_keys():
    """Verify MACCS structural keys produce 166-bit binary array."""
    smi = "CC(=O)Nc1ccccc1"
    maccs = calculate_maccs_keys(smi)

    assert isinstance(maccs, np.ndarray)
    assert maccs.shape == (166,)
    assert set(np.unique(maccs)).issubset({0, 1})


def test_morgan_bit_info():
    """Verify bit info dictionary maps bits to atom indices and radii."""
    smi = "c1ccccc1"  # Benzene
    fp, info = get_morgan_bit_info(smi, radius=2, n_bits=512)
    assert isinstance(info, dict)
    assert len(info) > 0


def test_feature_vector_shape():
    """Verify end-to-end generate_features returns 706-dimensional vector."""
    smi = "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
    vec = generate_features(smi, n_bits=512)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (706,)
    assert np.all(np.isfinite(vec))


def test_descriptor_table():
    """Verify descriptor table formatting contains required display columns."""
    desc = calculate_rdkit_descriptors("CCO")
    df = get_descriptor_table_df(desc)

    assert not df.empty
    assert "Property" in df.columns
    assert "Value" in df.columns
    assert "Unit" in df.columns
    assert "Description" in df.columns
