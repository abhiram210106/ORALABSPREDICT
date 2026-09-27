"""
molecular_viewer.py
-------------------
Streamlit component for rendering 2D SVG molecular diagrams and
interactive 3Dmol.js conformer viewers for OralAbsPredict.
"""

from typing import Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components


def render_molecular_viewer(result: Dict[str, Any]):
    """
    Renders 2D structure and interactive 3D conformer viewer in Streamlit.
    """
    st.markdown("### Molecular Structure")

    tabs = st.tabs(["2D Structure", "3D Conformer Model"])

    with tabs[0]:
        svg_content = result.get("svg_2d")
        if svg_content:
            # Wrap SVG inside a styled card
            html_2d = f"""
            <div style="
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 16px;
                display: flex;
                justify-content: center;
                align-items: center;
                box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.4);
            ">
                <div style="max-width: 100%; height: auto;">
                    {svg_content}
                </div>
            </div>
            """
            components.html(html_2d, height=360)
        else:
            st.warning("2D depiction could not be generated.")

    with tabs[1]:
        has_3d = result.get("has_3d", False)
        html_3d = result.get("html_3d")

        if has_3d and html_3d:
            st.caption("Interactive 3D model: Click and drag to rotate, scroll to zoom.")
            components.html(html_3d, height=370)
        else:
            st.info(
                "3D conformer generation was unavailable for this specific chemical structure. "
                "The 2D structure above represents the valid molecular graph."
            )

    # Basic structural identifiers
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Molecular Formula:** `{result.get('molecular_formula', 'N/A')}`")
    with col_b:
        can_smi = result.get("canonical_smiles", "")
        truncated_smi = can_smi[:35] + "..." if len(can_smi) > 35 else can_smi
        st.markdown(f"**Canonical SMILES:** `{truncated_smi}`")
