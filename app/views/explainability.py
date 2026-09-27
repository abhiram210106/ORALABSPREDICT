"""
explainability.py
-----------------
Global Explainable AI (XAI) dashboard page for OralAbsPredict.
Visualizes global SHAP feature importance rankings and explains the
pharmacological mechanisms behind intestinal permeability and oral bioavailability.
"""

from typing import Dict, Any, Optional
import os
import json
import textwrap
import streamlit as st
import pandas as pd
import joblib

from src.explainability import get_global_feature_importance


def render_explainability_page():
    st.markdown("## Global Explainable AI (SHAP Analysis)")
    st.markdown(
        "To demystify black-box machine learning in drug discovery, **OralAbsPredict** integrates "
        "**SHAP (SHapley Additive exPlanations)** based on cooperative game theory. Below is the global "
        "feature importance distribution revealing the primary physicochemical drivers learned by the models."
    )

    endpoint_choice = st.radio(
        "Select Endpoint for Global Analysis:",
        options=["Human Intestinal Absorption (HIA)", "Human Oral Bioavailability (HOB)"],
        horizontal=True
    )

    model_file = "models/hia_model.pkl" if "Intestinal" in endpoint_choice else "models/hob_model.pkl"
    feat_file = "models/hia_feature_names.pkl" if "Intestinal" in endpoint_choice else "models/hob_feature_names.pkl"

    if not (os.path.exists(model_file) and os.path.exists(feat_file)):
        st.warning("Trained models not found. Please train models first.")
        return

    with st.spinner("Extracting global feature importance..."):
        model = joblib.load(model_file)
        feature_names = joblib.load(feat_file)
        df_imp, fig_global = get_global_feature_importance(model, feature_names, top_n=15)

    st.markdown("### Top Global Feature Importances")
    st.plotly_chart(fig_global, use_container_width=True)

    # Detailed Mechanistic Breakdown
    st.markdown("### Pharmacological Interpretation of Top Chemical Features")

    col1, col2 = st.columns(2)

    with col1:
        c1_html = (
            '<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 16px;">'
            '<div style="font-weight: 700; color: #38bdf8; font-size: 1rem; margin-bottom: 6px;">⚡ Formal Charge & Ionization</div>'
            '<div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.55;">'
            'Fixed ionic charges create severe desolvation and hydration penalties, dramatically retarding passive transcellular permeation across the nonpolar hydrocarbon core of enterocyte apical membranes. Neutral species penetrate significantly faster.'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c1_html)
        else:
            st.markdown(c1_html, unsafe_allow_html=True)

        c2_html = (
            '<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 16px;">'
            '<div style="font-weight: 700; color: #38bdf8; font-size: 1rem; margin-bottom: 6px;">💧 Topological Polar Surface Area (TPSA)</div>'
            '<div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.55;">'
            'TPSA represents the sum of surfaces from oxygen, nitrogen, and attached hydrogens. Veber\'s rule establishes that compounds with <strong>TPSA &le; 140 Å²</strong> possess markedly higher oral bioavailability due to lower hydrogen bonding barriers to lipid bilayer insertion.'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c2_html)
        else:
            st.markdown(c2_html, unsafe_allow_html=True)

    with col2:
        c3_html = (
            '<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 16px;">'
            '<div style="font-weight: 700; color: #38bdf8; font-size: 1rem; margin-bottom: 6px;">⚖️ Lipophilicity (Wildman-Crippen LogP)</div>'
            '<div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.55;">'
            'Optimal oral absorption requires balanced lipophilicity (<strong>0 &lt; LogP &lt; 5</strong>). Insufficient lipophilicity hinders cell membrane entry, whereas excessive lipophilicity causes poor aqueous dissolution in intestinal fluid and rapid hepatic cytochrome P450 clearance.'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c3_html)
        else:
            st.markdown(c3_html, unsafe_allow_html=True)

        c4_html = (
            '<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid #334155; border-radius: 12px; padding: 18px; margin-bottom: 16px;">'
            '<div style="font-weight: 700; color: #38bdf8; font-size: 1rem; margin-bottom: 6px;">🧬 Morgan Circular Fingerprints (ECFP4)</div>'
            '<div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.55;">'
            'Circular bit features capture specific functional group microenvironments (e.g. ester groups, sulfonamides, aromatic heterocycles). These substructures directly correlate with substrate susceptibility to intestinal efflux transporters (e.g., P-glycoprotein) and metabolic enzymes.'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c4_html)
        else:
            st.markdown(c4_html, unsafe_allow_html=True)

    if hasattr(st, "html"):
        st.html("<div style='height: 12px;'></div>")
    else:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    with st.expander("View Full Global Feature Importance Table"):
        st.dataframe(
            df_imp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "clean_name": st.column_config.TextColumn("Feature", width="large"),
                "importance": st.column_config.NumberColumn("Raw Score", format="%.4f"),
                "relative_pct": st.column_config.NumberColumn("Relative Share (%)", format="%.2f%%"),
            }
        )
