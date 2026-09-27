# OralAbsPredict

> **OralAbsPredict: A Data-Driven Framework to Predict Human Intestinal Absorption (HIA) and Human Oral Bioavailability (HOB) from Chemical Structures**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44+-red.svg)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-orange.svg)](https://www.rdkit.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow.svg)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple.svg)](https://github.com/slundberg/shap)
[![Tests Passing](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)](tests/)

---

## 1. Overview

**OralAbsPredict** is an end-to-end, research-grade machine learning and explainable AI framework designed to accelerate early-stage oral drug discovery. Given an arbitrary chemical structure represented as a SMILES string, OralAbsPredict calculates 706 molecular descriptors (RDKit 2D physicochemical properties, Mordred graph-theoretical indices, and Morgan circular fingerprints), predicts both **Human Intestinal Absorption (HIA)** and **Human Oral Bioavailability (HOB)**, evaluates prediction confidence, and transparently decodes the underlying structural and physicochemical drivers using SHAP (SHapley Additive exPlanations).

---

## 2. Problem Statement

> *"Poor oral absorption is one of the primary reasons for drug failure during the early stages of drug discovery, leading to increased development time and cost. Experimental evaluation of Human Intestinal Absorption (HIA) and Human Oral Bioavailability (HOB) is expensive, time-consuming, and resource-intensive. Existing computational methods often suffer from limited datasets, class imbalance, and poor interpretability. This project aims to develop an AI-driven framework that predicts HIA and HOB directly from molecular structures represented as SMILES strings. The system employs machine learning and molecular descriptors to accurately classify orally active compounds before synthesis. It also provides feature interpretation using explainable AI techniques and identifies important molecular substructures responsible for oral absorption. This approach enables rapid screening of drug candidates, reducing experimental costs and accelerating drug discovery."*

---

## 3. Objectives

1. **Chemical Structure Parsing & Validation**: Validate, sanitize, and canonicalize SMILES strings via RDKit with automated salt removal and structure checking.
2. **Comprehensive Molecular Feature Extraction**: Deterministically compute 706 molecular features per compound, encompassing Lipinski's Rule of 5 parameters, Veber bioavailability criteria, 2D Mordred graph invariants, and 512-bit Morgan circular fingerprints (ECFP4).
3. **Imbalance-Aware Model Benchmarking**: Systematically benchmark six candidate algorithms (Logistic Regression, Random Forest, Balanced Random Forest, SVM, LightGBM, and CatBoost) under stratified splits, ranking models strictly by **Balanced Accuracy** and **F1-Score** rather than biased raw accuracy.
4. **Transparent Explainable AI (SHAP)**: Compute local and global feature attribution via cooperative game theory to expose exactly which functional groups and physicochemical properties govern the prediction.
5. **Interactive Scientific Visualization**: Generate crisp 2D SVG molecular diagrams alongside interactive 3D conformers (ETKDG + MMFF94 force field optimization) rendered via embedded 3Dmol.js.
6. **Full Reproducibility & Deployment Readiness**: Provide a modular, production-ready codebase, complete test suite, automated CLI launcher, and an intuitive Streamlit dashboard.

---

## 4. System Architecture

```
SMILES Input
      ↓
Molecular Validation (RDKit)
      ↓
RDKit Molecular Processing
      ↓
Descriptors + Fingerprints (706 Features)
      ↓
Feature Preprocessing & Scaling
      ↓
ML Models
      ↓
 ┌───────────────────────────────┬───────────────────────────────┐
 ↓                               ↓
HIA Model                       HOB Model
(Hou et al., CatBoost)          (Ma et al., CatBoost)
 ↓                               ↓
HIA Prediction: High / Low      HOB Prediction: High / Low
 └───────────────────────────────┴───────────────────────────────┘
                                 ↓
                        SHAP Explainability
                     (Local & Global Drivers)
                                 ↓
                      Molecular Visualization
                       (2D SVG + 3D Conformer)
                                 ↓
                        Streamlit Dashboard
```

---

## 5. Methodology

1. **Data Ingestion & Hygiene**:
   - Clean SMILES strings using `Chem.MolFromSmiles`, convert to canonical isomeric SMILES, drop invalid syntax.
   - Detect and resolve identical molecules with conflicting experimental bioactivity labels.
2. **Feature Extraction Pipeline**:
   - **RDKit 2D Physicochemical Descriptors (18)**: Molecular Weight (MW), Wildman-Crippen LogP, Topological Polar Surface Area (TPSA), H-Bond Donors (HBD), H-Bond Acceptors (HBA), Rotatable Bonds, Total Rings, Aromatic Rings, Aliphatic Rings, Fraction Csp3, Heavy Atom Count, Heteroatoms, Formal Charge, Molar Refractivity (MolMR), Labute ASA, Bertz Complexity, Lipinski Rule of 5 Violations, Veber Rule Compliance.
   - **Mordred 2D Graph Indices (176)**: Topological index, atom count, ring count, carbon types, hydrogen bonding, polarizability, aromaticity.
   - **Morgan Circular Fingerprints (512)**: Radius 2 (ECFP4 equivalent) circular bit vectors capturing atomic microenvironments.
3. **Data Splitting & Cross-Validation**:
   - Stratified Train (65%), Validation (15%), and Test (20%) partitioning maintaining minority class ratios.
4. **Algorithmic Benchmarking**:
   - Evaluated 6 diverse model architectures with class balancing (`class_weight='balanced'` / `auto_class_weights='Balanced'`).
5. **Serialization & Inference**:
   - Serialized winning models, scalers, exact feature orderings, and comprehensive JSON metadata for zero-drift inference.

---

## 6. Datasets

| Dataset | Primary Source | Samples | Target Class 0 (Low) | Target Class 1 (High) | Imbalance Ratio | Endpoint |
|---|---|---|---|---|---|---|
| **HIA** | Hou et al. (TDC `HIA_Hou`) | 578 | 78 (13.5%) | 500 (86.5%) | ~1 : 6.4 | Human Intestinal Absorption ($\ge 30\%$ absorption = 1) |
| **HOB** | Ma et al. (TDC `Bioavailability_Ma`) | 640 | 148 (23.1%) | 492 (76.9%) | ~1 : 3.3 | Human Oral Bioavailability ($F \ge 30\% = 1$) |
| **Caco-2** | Wang et al. (TDC `Caco2_Wang`) | 910 | Continuous | Continuous | N/A | Apparent Permeability ($\log P_{app}$ in cm/s) |

---

## 7. Machine Learning Models & Evaluation

The six algorithms specified in the project requirements were evaluated on the validation set:

### HIA Validation Benchmark Comparison
| Architecture | Balanced Accuracy | F1-Score | ROC-AUC | PR-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|---|
| **CatBoost (Selected)** | **0.8967** | **0.9664** | **0.9656** | **0.9946** | 0.9600 | 0.9730 | 0.9425 |
| Random Forest | 0.8550 | 0.9600 | 0.9600 | 0.9936 | 0.9600 | 0.9600 | 0.9310 |
| Balanced Random Forest | 0.8417 | 0.9459 | 0.9422 | 0.9904 | 0.9333 | 0.9589 | 0.9080 |
| Logistic Regression (StandardScaler) | 0.7783 | 0.9542 | 0.9511 | 0.9919 | 0.9733 | 0.9359 | 0.9195 |
| LightGBM | 0.7650 | 0.9404 | 0.9411 | 0.9907 | 0.9467 | 0.9342 | 0.8966 |
| Support Vector Machine (RBF) | 0.7017 | 0.9487 | 0.9389 | 0.9895 | 0.9867 | 0.9136 | 0.9080 |

### HOB Validation Benchmark Comparison
| Architecture | Balanced Accuracy | F1-Score | ROC-AUC | PR-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|---|
| **CatBoost (Selected)** | **0.6665** | **0.8609** | **0.6892** | **0.8753** | 0.8784 | 0.8442 | 0.7813 |
| Logistic Regression | 0.6579 | 0.8085 | 0.7285 | 0.9029 | 0.7703 | 0.8507 | 0.7188 |
| Random Forest | 0.6302 | 0.8400 | 0.7039 | 0.8738 | 0.8514 | 0.8289 | 0.7500 |
| LightGBM | 0.6210 | 0.8497 | 0.6695 | 0.8582 | 0.8784 | 0.8228 | 0.7604 |
| Balanced Random Forest | 0.6192 | 0.8056 | 0.6517 | 0.8539 | 0.7838 | 0.8286 | 0.7083 |
| Support Vector Machine (RBF) | 0.5663 | 0.8481 | 0.6732 | 0.8606 | 0.9054 | 0.7976 | 0.7500 |

---

## 8. Final Test Set Results

The winning **CatBoost** models were trained on combined training and validation sets and evaluated on the untouched test sets:

| Evaluation Metric | HIA Model (Test Set, N=116) | HOB Model (Test Set, N=128) |
|---|---|---|
| **Accuracy** | **93.97%** | **74.22%** |
| **Balanced Accuracy** | **91.25%** | **57.72%** |
| **Precision** | **97.94%** | **79.82%** |
| **Recall (Sensitivity)** | **95.00%** | **88.78%** |
| **F1-Score** | **0.9645** | **0.8406** |
| **ROC-AUC** | **0.9844** | **0.7425** |
| **PR-AUC** | **0.9976** | **0.9052** |
| **Confusion Matrix** | `[[14, 2], [5, 95]]` | `[[8, 22], [11, 87]]` |

*Note on Biological Context: Intestinal Absorption (HIA) represents passive and carrier-mediated epithelial diffusion, yielding high test ROC-AUC (0.9844). Oral Bioavailability (HOB) involves active hepatic first-pass metabolism, gut-wall CYP3A4 clearance, and biliary elimination, presenting a harder pharmacokinetic endpoint (0.7425 ROC-AUC), reflecting genuine clinical literature.*

---

## 9. Explainable AI (SHAP)

OralAbsPredict uses `shap.TreeExplainer` to provide:
1. **Local Explanations**: For every query compound, SHAP calculates positive features (driving high absorption) and negative features (driving low absorption).
2. **Global Feature Importances**: Across the entire chemical library, the primary governing features are:
   - **Formal Charge / Ionization (32.18% importance)**: Charged species encounter severe desolvation penalties crossing enterocyte lipid bilayers.
   - **Topological Polar Surface Area / TPSA (17.67% importance)**: Directly governs passive membrane permeation (Veber rule $\le 140$ Å²).
   - **H-Bond Donors (4.98% importance)**: High donor counts increase hydrogen-bonding with water, hindering membrane entry.
   - **Wildman-Crippen LogP (3.20% importance)**: Optimal lipophilicity ($0 < \text{LogP} < 5$) ensures membrane partitioning without solubility penalties.
   - **Morgan Circular Substructure Bits**: Specific heterocyclic and alkyl functional groups identifying metabolic liabilities or transporter substrates.

---

## 10. Molecular Visualization

- **2D Depiction**: High-resolution SVG vector rendering generated dynamically by RDKit with clear bond highlights and atomic colorings.
- **3D Conformer Model**: 3D coordinates generated via RDKit's ETKDG v3 algorithm with MMFF94 force-field geometry minimization, rendered interactively using 3Dmol.js (rotate, zoom, stick/sphere representation).

---

## 11. Installation & Setup

### Prerequisites
- Python 3.12 or 3.13 (64-bit)
- Windows, macOS, or Linux

### Step 1: Clone or Open Workspace
```bash
git clone https://github.com/your-username/ORALABSPREDICT.git
cd ORALABSPREDICT
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 12. Running the Application

### Option A: Via Unified Launcher
```bash
python run.py app
```

### Option B: Direct Streamlit Execution
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 13. Training the Models

To retrain the models from scratch or train on updated datasets:
```bash
# Option A: Train both models via CLI launcher
python run.py train

# Option B: Run individual training pipelines
python -m src.train_hia --data data/raw/hia_hou.csv --seed 42
python -m src.train_hob --data data/raw/bioavailability_ma.csv --seed 42
```

---

## 14. Running the Test Suite

Execute the 23-test unit and integration test suite:
```bash
# Option A: Via CLI launcher
python run.py test

# Option B: Via pytest directly
pytest tests/ -v
```

---

## 15. Project Directory Structure

```
ORALABSPREDICT/
│
├── app/                                 # Streamlit Web Application
│   ├── app.py                           # Main application entrypoint
│   ├── components/                      # Reusable UI components
│   │   ├── __init__.py
│   │   ├── descriptor_table.py          # Lipinski & Veber tables
│   │   ├── metric_cards.py              # Glassmorphic KPI prediction cards
│   │   ├── molecular_viewer.py          # 2D SVG & 3Dmol conformer viewers
│   │   └── shap_plots.py                # SHAP attribution charts
│   └── pages/                           # Application views
│       ├── __init__.py
│       ├── about.py                     # Project specifications & references
│       ├── explainability.py            # Global SHAP feature analysis
│       ├── home.py                      # Hero landing page
│       ├── performance.py               # 6-Model benchmarking & confusion matrices
│       └── prediction.py                # Dual HIA/HOB inference dashboard
│
├── data/                                # Datasets
│   ├── raw/                             # Original benchmark CSVs
│   │   ├── bioavailability_ma.csv       # HOB dataset (Ma et al.)
│   │   ├── caco2_wang.csv               # Auxiliary permeability (Wang et al.)
│   │   ├── hia_hou.csv                  # HIA dataset (Hou et al.)
│   │   ├── hia_template.csv             # Template for custom HIA data
│   │   └── hob_template.csv             # Template for custom HOB data
│   ├── processed/                       # Cached feature arrays (.npz)
│   └── README.md                        # Dataset documentation & provenance
│
├── models/                              # Serialized Model Artifacts
│   ├── hia_feature_names.pkl            # 706 feature names in exact order
│   ├── hia_model.pkl                    # Trained HIA CatBoost model
│   ├── hob_feature_names.pkl            # 706 feature names in exact order
│   ├── hob_model.pkl                    # Trained HOB CatBoost model
│   └── metadata/                        # JSON metadata records
│       ├── hia_metadata.json            # Metrics, class breakdown, validation table
│       └── hob_metadata.json            # Metrics, class breakdown, validation table
│
├── notebooks/                           # Research Jupyter Notebooks
│   ├── 01_data_exploration.ipynb        # Exploratory Data Analysis (EDA)
│   ├── 02_feature_engineering.ipynb     # Descriptors & Morgan fingerprints
│   ├── 03_model_training.ipynb          # Stratified training pipeline
│   ├── 04_model_comparison.ipynb        # Multi-algorithm benchmark comparison
│   └── 05_shap_analysis.ipynb           # Local & Global SHAP analysis
│
├── src/                                 # Core Python Modules
│   ├── __init__.py
│   ├── data_loader.py                   # Automatic column detection & splitting
│   ├── descriptors.py                   # RDKit & Mordred descriptor calculations
│   ├── explainability.py                # SHAP TreeExplainer & feature attribution
│   ├── fingerprints.py                  # Morgan circular & MACCS fingerprints
│   ├── model_comparison.py              # 6-Algorithm comparison & metrics
│   ├── predictor.py                     # Unified inference engine
│   ├── preprocessing.py                 # SMILES sanitization & deduplication
│   ├── train_hia.py                     # End-to-end HIA training script
│   ├── train_hob.py                     # End-to-end HOB training script
│   └── visualization.py                 # 2D SVG & 3D ETKDG conformer generation
│
├── tests/                               # Comprehensive Test Suite (23 Tests)
│   ├── __init__.py
│   ├── test_data.py                     # SMILES validation & data cleaning tests
│   ├── test_descriptors.py              # Descriptors, fingerprints, shape tests
│   ├── test_models.py                   # Model loading & metric evaluation tests
│   └── test_prediction.py              # Inference, SHAP, and error handling tests
│
├── .gitignore                           # Git ignore rules
├── LICENSE                              # MIT License
├── README.md                            # Comprehensive project documentation
├── requirements.txt                     # Project dependencies
└── run.py                               # CLI launcher (app, train, test)
```

---

## 16. Limitations & Future Scope

### Current Limitations
1. **Dataset Size**: Benchmark clinical oral bioavailability data is constrained to hundreds of molecules due to the cost and complexity of human clinical trials.
2. **First-Pass Hepatic Dynamics**: Purely 2D/topological representations cannot fully simulate individual patient liver CYP3A4 enzyme polymorphism or biliary excretion rates.
3. **Active Transporter Kinetics**: Non-passive influx/efflux (e.g., P-glycoprotein, OATP, PEPT1) may introduce variance for non-small-molecule compounds.

### Future Scope
1. **Geometric Deep Learning**: Incorporating Directed Message Passing Neural Networks (D-MPNN / Chemprop) for end-to-end molecular representation learning.
2. **Conformal Prediction**: Generating rigorously calibrated prediction intervals at guaranteed statistical significance levels.
3. **Multi-Task Transfer Learning**: Co-training across HIA, Caco-2 permeability, plasma protein binding, and renal clearance.

---

## 17. Scientific Disclaimer

> **IMPORTANT NOTICE**: OralAbsPredict is an artificial intelligence research prototype developed strictly for computational screening, cheminformatics exploration, and educational evaluation. It is **NOT** a clinical diagnostic tool, prescribing system, or medical device. Predictions generated by this framework do not guarantee pharmacodynamic efficacy or pharmacokinetic safety in vivo. In vitro assays (Caco-2, PAMPA) and in vivo human clinical trials remain mandatory prior to synthesizing or administering any chemical compound.

---

## 18. Authors & Acknowledgments

- **Project**: OralAbsPredict — College Project School Specification
- **Frameworks Used**: RDKit, Scikit-learn, CatBoost, LightGBM, Imbalanced-learn, Mordred Community, SHAP, Streamlit, Plotly.
- **Datasets**: Therapeutics Data Commons (TDC) benchmarks by Hou et al. (2007) and Ma et al. (2008).
