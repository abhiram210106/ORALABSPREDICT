# OralAbsPredict Datasets

This directory contains the cheminformatics datasets used for training and evaluating machine learning models in **OralAbsPredict**.

## 1. Human Intestinal Absorption (HIA)

- **Source**: Hou et al. / Therapeutics Data Commons (TDC) benchmark `HIA_Hou`.
  - Citation: Hou, T. J., Wang, J. M., Shen, J. Y., Zhang, W., & Xu, X. J. (2007). *ADME evaluation in drug discovery. 4. Prediction of oral absorption in humans based on molecular properties*. Journal of Chemical Information and Modeling, 47(2), 460-463.
- **File**: `data/raw/hia_hou.csv`
- **Rows**: 578 compounds
- **Target (`Y`)**: Binary classification
  - **`1`**: High Intestinal Absorption (Absorbed Fraction $\ge 30\%$, typically classified as HIA+)
  - **`0`**: Poor / Low Intestinal Absorption (Absorbed Fraction $< 30\%$, classified as HIA-)
- **Class Balance**:
  - Class `1`: 500 (~86.5%)
  - Class `0`: 78 (~13.5%)
  - *Significant class imbalance present; requires balanced algorithms (e.g., Balanced Random Forest, class-weighted loss, PR-AUC evaluation).*

## 2. Human Oral Bioavailability (HOB)

- **Source**: Ma et al. / Therapeutics Data Commons (TDC) benchmark `Bioavailability_Ma`.
  - Citation: Ma, C. Y., et al. (2008 / 2020). *Evaluation of machine learning methods for predicting human oral bioavailability*. Journal of Chemical Information and Modeling.
- **File**: `data/raw/bioavailability_ma.csv`
- **Rows**: 640 compounds
- **Target (`Y`)**: Binary classification
  - **`1`**: High Oral Bioavailability ($F \ge 30\%$, compound exhibits substantial systemic exposure upon oral administration)
  - **`0`**: Poor / Low Oral Bioavailability ($F < 30\%$, compound experiences poor absorption or extensive first-pass metabolism)
- **Class Balance**:
  - Class `1`: 492 (~76.9%)
  - Class `0`: 148 (~23.1%)

## 3. Auxiliary Permeability (Caco-2)

- **Source**: Wang et al. / TDC benchmark `Caco2_Wang`.
- **File**: `data/raw/caco2_wang.csv`
- **Rows**: 910 compounds
- **Target (`Y`)**: Continuous (Apparent Permeability coefficient $\log P_{app}$ in $\text{cm/s}$).

## 4. Expected Dataset Formats & Templates

Any custom or updated dataset can be dropped into `data/raw/`. The pipeline automatically identifies SMILES and label columns matching common patterns:
- SMILES columns: `SMILES`, `smiles`, `Smiles`, `canonical_smiles`, `structure`, `mol`
- Target columns: `Y`, `target`, `label`, `HIA`, `HOB`, `Class`, `Activity`, `bioavailability`, `absorption`

Templates are available at:
- `data/raw/hia_template.csv`
- `data/raw/hob_template.csv`

Processed feature matrices and descriptors are stored in `data/processed/`.
