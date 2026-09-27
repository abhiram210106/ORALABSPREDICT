"""
about.py
--------
About Project, Methodology, Scientific References, and Disclaimer page for OralAbsPredict.
"""

import textwrap
import streamlit as st


def render_about_page():
    st.markdown("## About OralAbsPredict")
    st.markdown(
        "**OralAbsPredict: A Data-Driven Framework to Predict Human Intestinal Absorption (HIA) "
        "and Human Oral Bioavailability (HOB) from Chemical Structures** is an academic AI cheminformatics "
        "framework designed for early-stage in silico pharmacokinetic screening."
    )

    st.markdown("""
### System Architecture & Pipeline
```
Chemical Structure (SMILES)
           │
           ▼
Molecular Validation & Sanitization (RDKit)
           │
           ▼
Feature Engineering (706 Dimensions)
┌──────────────────────────────────────────────┐
│ • 18 RDKit Physicochemical Descriptors       │
│ • 176 Mordred 2D Graph/Topological Indices   │
│ • 512-bit Morgan Circular Fingerprints (ECFP4)│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
      Stratified Train / Val / Test Split
                       │
┌──────────────────────┴───────────────────────┐
▼                                              ▼
HIA Classification Pipeline                   HOB Classification Pipeline
(Hou et al. Benchmark, N=578)                 (Ma et al. Benchmark, N=640)
• CatBoost (Selected Winner)                  • CatBoost (Selected Winner)
• Test Acc: 93.97%, Bal Acc: 91.25%           • Test Acc: 74.22%, AUC: 0.7425
└──────────────────────┬───────────────────────┘
                       │
                       ▼
Dual Prediction & Confidence Quantification
                       │
┌──────────────────────┴───────────────────────┐
▼                                              ▼
SHAP Explainability Engine             2D/3D Molecular Visualization
• TreeExplainer Local Contributions    • RDKit 2D SVG Depiction
• Global Pharmacophoric Drivers        • 3Dmol.js Conformer Optimization
└──────────────────────┬───────────────────────┘
                       │
                       ▼
          Interactive Streamlit Dashboard
```
    """)

    st.markdown("---")

    # Methodology Section
    st.markdown("""
### Methodology Overview
1. **Data Curation & Quality Control**:
   - Benchmark datasets from Therapeutics Data Commons (TDC) were processed using RDKit for SMILES sanitization, stereochemical standardization, and deduplication with conflicting target resolution.
2. **High-Dimensional Feature Engineering**:
   - Combines classical physicochemical parameters (Lipinski descriptors, Veber parameters) with 2D graph indices (Mordred) and Morgan circular fingerprints (radius=2, 512 bits) to capture both global solubility/permeability properties and local functional groups.
3. **Imbalance-Aware Model Benchmarking**:
   - Evaluated six diverse algorithms: Logistic Regression, Random Forest, Balanced Random Forest, Support Vector Machine, LightGBM, and CatBoost.
   - Selected final models based on **Balanced Accuracy** and **F1-Score** to prevent majority-class bias.
4. **Explainability & Mechanistic Transparency**:
   - Implemented SHAP (SHapley Additive exPlanations) to isolate exactly which chemical fragments and physical properties promote or impede absorption.
    """)

    st.markdown("---")

    # Scientific References
    st.markdown("""
### Key Scientific Literature References
1. **Hou, T. J., Wang, J. M., Shen, J. Y., Zhang, W., & Xu, X. J. (2007)**. *ADME evaluation in drug discovery. 4. Prediction of oral absorption in humans based on molecular properties*. Journal of Chemical Information and Modeling, 47(2), 460-463.
2. **Ma, C. Y., et al. (2008 / 2020)**. *Evaluation of machine learning methods for predicting human oral bioavailability*. Journal of Chemical Information and Modeling.
3. **Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (2001)**. *Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings*. Advanced Drug Delivery Reviews, 46(1-3), 3-26.
4. **Veber, D. F., et al. (2002)**. *Molecular properties that influence the oral bioavailability of drug candidates*. Journal of Medicinal Chemistry, 45(12), 2615-2623.
5. **Lundberg, S. M., & Lee, S. I. (2017)**. *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS), 30.
6. **Prokhorenkova, L., et al. (2018)**. *CatBoost: unbiased boosting with categorical features*. Advances in Neural Information Processing Systems (NeurIPS), 31.
    """)

    st.markdown("---")

    # Limitations & Future Scope
    col_lim, col_scope = st.columns(2)

    with col_lim:
        st.markdown("""
#### Limitations
- **Dataset Size**: Public clinical oral bioavailability datasets are constrained to hundreds of molecules due to the cost of human pharmacokinetic trials.
- **Metabolic Complexity**: First-pass hepatic extraction involves complex polymorphic CYP450 enzyme kinetics and biliary excretion that purely 2D structural models cannot fully capture.
- **Carrier-Mediated Transport**: Active uptake transporters (PEPT1, OATP) and efflux pumps (P-gp, BCRP) can cause deviations from passive diffusion predictions.
        """)

    with col_scope:
        st.markdown("""
#### Future Scope
- **Graph Neural Networks (GNNs)**: Integrating directed message passing neural networks (D-MPNN / Chemprop) for end-to-end representation learning.
- **Conformal Prediction**: Implementing valid confidence intervals to guarantee calibration at fixed significance levels.
- **Multi-task ADMET Learning**: Co-training HIA, Caco-2 permeability, HOB, and clearance simultaneously to leverage shared ADME representations.
        """)

    st.markdown("---")

    # Non-Clinical Disclaimer
    disclaimer_html = (
        '<div style="background: rgba(234, 179, 8, 0.08); border: 1px solid rgba(234, 179, 8, 0.3); '
        'border-left: 4px solid #eab308; border-radius: 12px; padding: 16px 20px;">'
        '<strong style="color: #fde047;">Non-Clinical Research Disclaimer:</strong><br>'
        'OralAbsPredict is developed exclusively for computational screening, cheminformatics education, and academic research. '
        'Predictions generated by this tool do not guarantee pharmacokinetic behavior in vivo and must never be utilized as clinical or prescribing advice.'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(disclaimer_html)
    else:
        st.markdown(disclaimer_html, unsafe_allow_html=True)
