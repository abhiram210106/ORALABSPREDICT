"""
shap_plots.py
--------------
Streamlit component for rendering SHAP local explanation waterfall/bar charts
and positive/negative feature contribution breakdowns.
"""

from typing import Dict, Any, List
import textwrap
import streamlit as st
from src.explainability import explain_prediction


def render_shap_explanation(
    model: Any,
    feature_names: List[str],
    feature_vector: Any,
    endpoint_name: str,
    explainer: Any = None
):
    """
    Renders SHAP local contribution bar chart and structured explanation tables.
    """
    st.markdown(f"### Explainable AI: Feature Attribution for {endpoint_name}")
    st.caption(
        "SHAP (SHapley Additive exPlanations) computes the exact marginal contribution of each molecular feature "
        "toward pushing the prediction higher (High absorption/bioavailability) or lower (Low absorption/bioavailability)."
    )

    with st.spinner("Computing SHAP attribution values..."):
        explanation = explain_prediction(
            model=model,
            feature_names=feature_names,
            feature_vector=feature_vector,
            top_n=8,
            explainer=explainer
        )

    fig = explanation.get("plotly_figure")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown by Positive and Negative Contributions
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("#### Top Positive Contributing Features (+)")
        st.caption("Features favoring high oral absorption / bioavailability:")
        pos_df = explanation.get("positive_features")
        if pos_df is not None and not pos_df.empty:
            for _, row in pos_df.head(5).iterrows():
                val_str = f"{row['feature_value']:.2f}" if isinstance(row['feature_value'], float) else str(row['feature_value'])
                pos_html = (
                    f'<div style="background: rgba(16, 185, 129, 0.12); border-left: 3px solid #10b981; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">'
                    f'<div style="font-weight: 700; color: #f8fafc; font-size: 0.88rem;">{row["clean_name"]}</div>'
                    f'<div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between; margin-top: 4px;">'
                    f'<span>Value: <strong>{val_str}</strong></span>'
                    f'<span style="color: #10b981; font-weight: 700;">SHAP: +{row["shap_value"]:.4f}</span>'
                    f'</div></div>'
                )
                if hasattr(st, "html"):
                    st.html(pos_html)
                else:
                    st.markdown(pos_html, unsafe_allow_html=True)
        else:
            st.info("No significant positive contributors.")

    with col_neg:
        st.markdown("#### Top Negative Contributing Features (-)")
        st.caption("Features favoring poor absorption or extensive clearance:")
        neg_df = explanation.get("negative_features")
        if neg_df is not None and not neg_df.empty:
            for _, row in neg_df.head(5).iterrows():
                val_str = f"{row['feature_value']:.2f}" if isinstance(row['feature_value'], float) else str(row['feature_value'])
                neg_html = (
                    f'<div style="background: rgba(244, 63, 94, 0.12); border-left: 3px solid #f43f5e; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">'
                    f'<div style="font-weight: 700; color: #f8fafc; font-size: 0.88rem;">{row["clean_name"]}</div>'
                    f'<div style="font-size: 0.78rem; color: #94a3b8; display: flex; justify-content: space-between; margin-top: 4px;">'
                    f'<span>Value: <strong>{val_str}</strong></span>'
                    f'<span style="color: #f43f5e; font-weight: 700;">SHAP: {row["shap_value"]:.4f}</span>'
                    f'</div></div>'
                )
                if hasattr(st, "html"):
                    st.html(neg_html)
                else:
                    st.markdown(neg_html, unsafe_allow_html=True)
        else:
            st.info("No significant negative contributors.")

    # Expandable detailed table
    with st.expander("View Full SHAP Feature Attribution Table"):
        exp_table = explanation.get("explanation_table")
        if exp_table is not None and not exp_table.empty:
            st.dataframe(exp_table, use_container_width=True, hide_index=True)
