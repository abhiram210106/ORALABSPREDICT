"""
descriptor_table.py
-------------------
Streamlit component for rendering molecular physicochemical descriptors,
Lipinski's Rule of 5, and Veber oral bioavailability rules.
"""

from typing import Dict, Any
import textwrap
import streamlit as st
import pandas as pd


def render_descriptor_table(result: Dict[str, Any]):
    """
    Renders Lipinski & Veber drug-likeness rules alongside the full descriptor table.
    """
    descriptors = result.get("descriptors", {})
    desc_df = result.get("descriptor_table", pd.DataFrame())

    if not descriptors:
        st.info("No descriptors calculated.")
        return

    # Extract Key Oral Bioavailability Rules
    mw = descriptors.get("MolWt", 0.0)
    logp = descriptors.get("LogP", 0.0)
    tpsa = descriptors.get("TPSA", 0.0)
    hbd = int(descriptors.get("NumHDonors", 0))
    hba = int(descriptors.get("NumHAcceptors", 0))
    rot_bonds = int(descriptors.get("NumRotatableBonds", 0))
    ro5_violations = int(descriptors.get("Ro5_Violations", 0))
    veber_ok = bool(descriptors.get("Veber_Compliant", 1.0))

    st.markdown("### Molecular Descriptors & Oral Drug-Likeness")

    # Drug-likeness summary badges
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ro5_color = "#10b981" if ro5_violations <= 1 else "#f43f5e"
        ro5_status = "Compliant" if ro5_violations == 0 else f"{ro5_violations} Violation(s)"
        c1_html = (
            f'<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Lipinski Rule of 5</div>'
            f'<div style="font-size: 1.1rem; font-weight: 800; color: {ro5_color}; margin-top: 4px;">{ro5_status}</div>'
            f'</div>'
        )
        if hasattr(st, "html"):
            st.html(c1_html)
        else:
            st.markdown(c1_html, unsafe_allow_html=True)

    with col2:
        veber_color = "#10b981" if veber_ok else "#f43f5e"
        veber_status = "Compliant" if veber_ok else "Violated"
        c2_html = (
            f'<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Veber Rule</div>'
            f'<div style="font-size: 1.1rem; font-weight: 800; color: {veber_color}; margin-top: 4px;">{veber_status}</div>'
            f'</div>'
        )
        if hasattr(st, "html"):
            st.html(c2_html)
        else:
            st.markdown(c2_html, unsafe_allow_html=True)

    with col3:
        tpsa_color = "#10b981" if tpsa <= 140.0 else "#f43f5e"
        c3_html = (
            f'<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">TPSA (Permeability)</div>'
            f'<div style="font-size: 1.1rem; font-weight: 800; color: {tpsa_color}; margin-top: 4px;">{tpsa:.1f} Å²</div>'
            f'</div>'
        )
        if hasattr(st, "html"):
            st.html(c3_html)
        else:
            st.markdown(c3_html, unsafe_allow_html=True)

    with col4:
        logp_color = "#10b981" if (-0.4 <= logp <= 5.0) else "#f43f5e"
        c4_html = (
            f'<div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 10px; padding: 12px; text-align: center;">'
            f'<div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Wildman-Crippen LogP</div>'
            f'<div style="font-size: 1.1rem; font-weight: 800; color: {logp_color}; margin-top: 4px;">{logp:.2f}</div>'
            f'</div>'
        )
        if hasattr(st, "html"):
            st.html(c4_html)
        else:
            st.markdown(c4_html, unsafe_allow_html=True)

    if hasattr(st, "html"):
        st.html("<div style='height: 12px;'></div>")
    else:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Detailed Table
    if not desc_df.empty:
        st.dataframe(
            desc_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Property": st.column_config.TextColumn("Molecular Property", width="medium"),
                "Value": st.column_config.TextColumn("Calculated Value", width="small"),
                "Unit": st.column_config.TextColumn("Unit", width="small"),
                "Description": st.column_config.TextColumn("Pharmacological Relevance", width="large")
            }
        )
