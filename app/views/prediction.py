"""
prediction.py
-------------
Molecular prediction dashboard page for OralAbsPredict.
Validates SMILES, runs dual HIA/HOB inference, renders 2D/3D structures,
displays physicochemical descriptors, and provides SHAP explanations.
"""

from typing import Dict, Any
import streamlit as st

from src.predictor import OralAbsPredictor

try:
    from app.components.metric_cards import render_prediction_badge
    from app.components.molecular_viewer import render_molecular_viewer
    from app.components.descriptor_table import render_descriptor_table
    from app.components.shap_plots import render_shap_explanation
except (ModuleNotFoundError, ImportError):
    from components.metric_cards import render_prediction_badge
    from components.molecular_viewer import render_molecular_viewer
    from components.descriptor_table import render_descriptor_table
    from components.shap_plots import render_shap_explanation

# Pre-defined demonstration chemical structures with known pharmacophores
DEMO_MOLECULES = {
    "Select an example molecule...": "",
    "Paracetamol (Acetaminophen) [Analgesic]": "CC(=O)Nc1ccc(O)cc1",
    "Aspirin (Acetylsalicylic acid) [NSAID]": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine [CNS Stimulant]": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Ibuprofen [NSAID]": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Ethanol [Simple Alcohol]": "CCO",
    "Metformin [Antidiabetic / Polar Biguanide]": "CN(C)C(=N)NC(=N)N",
    "Atorvastatin [Statin / Lipophilic Multi-ring]": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"
}


def render_prediction_page(predictor: OralAbsPredictor, hia_explainer=None, hob_explainer=None):
    st.markdown("## Molecular ADME Prediction")
    st.markdown(
        "Enter a chemical structure as a **SMILES** string or select a demonstration compound below "
        "to evaluate Human Intestinal Absorption (HIA) and Human Oral Bioavailability (HOB)."
    )

    # Example dropdown and input controls
    col_input, col_demo = st.columns([3, 2])

    with col_demo:
        demo_selection = st.selectbox(
            "Load Demonstration Molecule:",
            options=list(DEMO_MOLECULES.keys()),
            help="Pre-configured drug structures for rapid functional testing. Labeled as demonstration examples."
        )

    # Determine default SMILES from dropdown
    default_smiles = DEMO_MOLECULES[demo_selection] if demo_selection and DEMO_MOLECULES[demo_selection] else "CC(=O)Nc1ccc(O)cc1"

    with col_input:
        user_smiles = st.text_input(
            "Enter SMILES string:",
            value=default_smiles,
            placeholder="e.g. CC(=O)Nc1ccccc1",
            help="Simplified Molecular Input Line Entry System representation."
        )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        predict_btn = st.button("🚀 Predict ADME", type="primary", use_container_width=True)

    with col_info:
        if demo_selection and demo_selection != "Select an example molecule...":
            st.caption(f"Loaded demo example: **{demo_selection}** (Educational demonstration only).")

    st.markdown("---")

    # Execute Prediction
    if predict_btn or user_smiles:
        smiles_to_test = user_smiles.strip()

        if not smiles_to_test:
            st.warning("Please enter a valid SMILES string.")
            return

        with st.spinner("Processing molecular graph and calculating descriptors..."):
            result = predictor.predict_all(smiles_to_test)

        if not result.get("success", False):
            st.error(f"❌ Molecular Error: {result.get('error', 'Invalid chemical structure')}")
            st.info("Tip: Ensure the input string follows standard IUPAC/Daylight SMILES syntax (e.g., CCO for ethanol).")
            return

        # Top Section: Two Prediction Result Cards (HIA & HOB)
        st.markdown("### Dual Oral Absorption Predictions")
        pred_col1, pred_col2 = st.columns(2)

        hia_res = result.get("hia")
        hob_res = result.get("hob")

        with pred_col1:
            if hia_res:
                render_prediction_badge(hia_res, "Human Intestinal Absorption (HIA)")
            else:
                st.warning("HIA model not loaded or inference failed.")

        with pred_col2:
            if hob_res:
                render_prediction_badge(hob_res, "Human Oral Bioavailability (HOB)")
            else:
                st.warning("HOB model not loaded or inference failed.")

        # Middle Section: 2D/3D Molecule Viewer and Descriptors Table
        col_mol, col_desc = st.columns([1, 1])

        with col_mol:
            render_molecular_viewer(result)

        with col_desc:
            render_descriptor_table(result)

        st.markdown("---")

        # Bottom Section: Explainable AI (SHAP)
        if hia_res and predictor.hia_model is not None:
            shap_tabs = st.tabs(["HIA SHAP Explanation", "HOB SHAP Explanation"])

            with shap_tabs[0]:
                render_shap_explanation(
                    model=predictor.hia_model,
                    feature_names=predictor.hia_features,
                    feature_vector=hia_res["feature_vector"],
                    endpoint_name="Human Intestinal Absorption (HIA)",
                    explainer=hia_explainer
                )

            with shap_tabs[1]:
                if hob_res and predictor.hob_model is not None:
                    render_shap_explanation(
                        model=predictor.hob_model,
                        feature_names=predictor.hob_features,
                        feature_vector=hob_res["feature_vector"],
                        endpoint_name="Human Oral Bioavailability (HOB)",
                        explainer=hob_explainer
                    )
