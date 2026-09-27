"""
performance.py
--------------
Model Performance and Comparative Benchmarking page for OralAbsPredict.
Visualizes candidate model comparisons across 6 algorithms, test set metrics,
ROC/PR characteristics, and confusion matrices.
"""

from typing import Dict, Any, Optional
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff


def load_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def render_performance_page():
    st.markdown("## Machine Learning Model Benchmarking")
    st.markdown(
        "To ensure scientific rigor and avoid model selection bias, **OralAbsPredict** evaluated "
        "six diverse machine learning architectures specified in the project guidelines using stratified "
        "training, validation, and test splits with class balancing techniques."
    )

    hia_meta = load_metadata("models/metadata/hia_metadata.json")
    hob_meta = load_metadata("models/metadata/hob_metadata.json")

    endpoint_tab = st.radio(
        "Select ADME Target Endpoint:",
        options=["Human Intestinal Absorption (HIA)", "Human Oral Bioavailability (HOB)"],
        horizontal=True
    )

    meta = hia_meta if "Intestinal" in endpoint_tab else hob_meta
    tag = "HIA" if "Intestinal" in endpoint_tab else "HOB"

    if meta is None:
        st.warning(f"Metadata for {endpoint_tab} not found. Please run the training pipeline first.")
        return

    # Section 1: Final Test Set Evaluation KPI Cards
    test_metrics = meta.get("test_metrics", {})
    st.markdown(f"### Final Test Set Evaluation ({meta.get('model_name', 'CatBoost')})")
    st.caption(f"Evaluated on {meta.get('test_samples', 0)} previously unseen test compounds.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Test Accuracy", f"{test_metrics.get('accuracy', 0.0) * 100:.2f}%")
    with kpi2:
        st.metric("Balanced Accuracy", f"{test_metrics.get('balanced_accuracy', 0.0) * 100:.2f}%")
    with kpi3:
        st.metric("F1-Score", f"{test_metrics.get('f1', 0.0):.4f}")
    with kpi4:
        st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0.0):.4f}")

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    with kpi5:
        st.metric("PR-AUC", f"{test_metrics.get('pr_auc', 0.0):.4f}")
    with kpi6:
        st.metric("Precision", f"{test_metrics.get('precision', 0.0):.4f}")
    with kpi7:
        st.metric("Recall (Sensitivity)", f"{test_metrics.get('recall', 0.0):.4f}")
    with kpi8:
        st.metric("Input Features", f"{meta.get('feature_count', 0)}")

    if hasattr(st, "html"):
        st.html("<div style='height: 16px;'></div>")
    else:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Section 2: Validation Comparison Across All 6 Candidate Models
    st.markdown("### Comparative Evaluation Across 6 Machine Learning Architectures")
    val_records = meta.get("validation_comparison", [])
    if val_records:
        df_comp = pd.DataFrame(val_records)

        # Highlight best model
        st.dataframe(
            df_comp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Architecture", width="medium"),
                "Balanced Accuracy": st.column_config.NumberColumn("Balanced Acc", format="%.4f"),
                "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                "ROC-AUC": st.column_config.NumberColumn("ROC-AUC", format="%.4f"),
                "PR-AUC": st.column_config.NumberColumn("PR-AUC", format="%.4f"),
                "Recall (Sensitivity)": st.column_config.NumberColumn("Recall", format="%.4f"),
                "Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                "Accuracy": st.column_config.NumberColumn("Raw Accuracy", format="%.4f"),
            }
        )

        # Comparative Bar Chart
        st.markdown("#### Balanced Accuracy vs F1-Score Comparison")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Balanced Accuracy",
            x=df_comp["Model"],
            y=df_comp["Balanced Accuracy"],
            marker_color="#38bdf8"
        ))
        fig_comp.add_trace(go.Bar(
            name="F1-Score",
            x=df_comp["Model"],
            y=df_comp["F1-Score"],
            marker_color="#10b981"
        ))
        fig_comp.update_layout(
            barmode="group",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0.4, 1.0]),
            margin=dict(l=40, r=40, t=30, b=50),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Section 3: Confusion Matrix & Rationale
    col_cm, col_rationale = st.columns([1, 1])

    with col_cm:
        st.markdown("#### Test Set Confusion Matrix")
        cm = test_metrics.get("confusion_matrix")
        if cm:
            z = np.array(cm)
            x_labels = ["Pred: Low", "Pred: High"]
            y_labels = ["Actual: Low", "Actual: High"]

            fig_cm = ff.create_annotated_heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale="Blues",
                showscale=False
            )
            fig_cm.update_layout(
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120",
                font=dict(color="#f8fafc"),
                margin=dict(l=80, r=40, t=30, b=40),
                height=300
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_rationale:
        st.markdown("#### Scientific Model Selection Rationale")
        st.markdown(f"""
        - **Class Imbalance Resilience**: The {tag} dataset exhibits natural biological imbalance 
          ({meta.get('class_distribution_raw', {}).get('1', 0)} positive vs {meta.get('class_distribution_raw', {}).get('0', 0)} negative compounds).
        - **Raw Accuracy vs Balanced Metrics**: Naive models that over-predict the majority class can yield high raw accuracy but poor clinical utility. 
          Selection was strictly governed by **Balanced Accuracy** and **F1-Score**.
        - **Selected Algorithm ({meta.get('model_name')})**: Achieved superior minority class recall while maintaining robust generalizability across high-dimensional Morgan fingerprints and graph descriptors.
        """)
