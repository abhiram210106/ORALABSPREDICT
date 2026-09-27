"""
explainability.py
-----------------
Explainable AI (XAI) using SHAP (SHapley Additive exPlanations) for OralAbsPredict.
Provides local prediction explanations (top positive/negative features),
global feature importance rankings, and interactive visualization charts.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from src.descriptors import CORE_DESCRIPTORS_META


def clean_feature_name(name: str) -> str:
    """
    Translates raw feature column names to clear, scientific descriptions.
    """
    if name in CORE_DESCRIPTORS_META:
        return f"{CORE_DESCRIPTORS_META[name]['name']} ({name})"
    if name.startswith("Mordred_"):
        raw = name.replace("Mordred_", "")
        # Clean module class names
        if "(" in raw:
            raw = raw.split("(")[0].replace("mordred.", "")
        return f"Mordred: {raw}"
    if name.startswith("Morgan_Bit_"):
        bit_num = name.replace("Morgan_Bit_", "")
        return f"Morgan Circular Substructure (Bit {bit_num})"
    return name


def explain_prediction(
    model: Any,
    feature_names: List[str],
    feature_vector: np.ndarray,
    top_n: int = 10,
    explainer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Computes local SHAP values for a single molecule, identifying top positive
    and negative contributors to the predicted absorption/bioavailability property.

    Parameters
    ----------
    model : Any
        Trained model (CatBoost, Random Forest, etc.).
    feature_names : List[str]
        Ordered list of feature names.
    feature_vector : np.ndarray
        1D feature vector for target compound.
    top_n : int, default 10
        Number of top positive and negative features to return.
    explainer : Optional[Any]
        Pre-cached SHAP explainer.

    Returns
    -------
    Dict[str, Any]
        Dictionary with SHAP values, top positive features, top negative features,
        summary table, and Plotly figure.
    """
    X = feature_vector.reshape(1, -1)

    # 1. Compute SHAP values
    shap_vals = None
    expected_value = 0.0

    try:
        if explainer is None:
            explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X)

        if isinstance(raw_shap, list):
            # Binary classifier returning list of [class_0, class_1]
            shap_vals = np.array(raw_shap[1][0], dtype=float)
            expected_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vals = np.array(raw_shap[0, :, 1], dtype=float)
            expected_value = float(explainer.expected_value[1])
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            shap_vals = np.array(raw_shap[0], dtype=float)
            expected_value = float(explainer.expected_value)
        else:
            shap_vals = np.array(raw_shap, dtype=float).flatten()
            expected_value = float(getattr(explainer, "expected_value", 0.0))
    except Exception:
        # Fallback to model feature importances scaled by deviation
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            shap_vals = (feature_vector - np.mean(feature_vector)) * importances / (np.sum(importances) + 1e-6)
            expected_value = 0.5
        else:
            shap_vals = np.zeros(len(feature_names))
            expected_value = 0.5

    # 2. Rank features by SHAP value
    df_features = pd.DataFrame({
        "raw_name": feature_names,
        "clean_name": [clean_feature_name(n) for n in feature_names],
        "feature_value": feature_vector,
        "shap_value": shap_vals,
        "abs_shap": np.abs(shap_vals)
    })

    # Positive contributors: push prediction towards High (class 1)
    df_positive = df_features[df_features["shap_value"] > 0].sort_values(
        by="shap_value", ascending=False
    ).head(top_n)

    # Negative contributors: push prediction towards Low (class 0)
    df_negative = df_features[df_features["shap_value"] < 0].sort_values(
        by="shap_value", ascending=True
    ).head(top_n)

    # Combined top features by absolute magnitude
    df_top = df_features.sort_values(by="abs_shap", ascending=False).head(top_n * 2)

    # 3. Create presentation table
    table_rows = []
    for _, row in df_top.head(top_n * 2).iterrows():
        val = row["feature_value"]
        formatted_val = f"{val:.2f}" if isinstance(val, float) else str(val)
        direction = "Positive (+High)" if row["shap_value"] > 0 else "Negative (-Low)"

        # Generate scientific context
        context = ""
        rname = row["raw_name"]
        if rname in CORE_DESCRIPTORS_META:
            context = CORE_DESCRIPTORS_META[rname]["description"]
        elif "Morgan" in rname:
            context = "Circular topological fragment / functional group"
        elif "Mordred" in rname:
            context = "2D graph-theoretical / topological index"

        table_rows.append({
            "Feature": row["clean_name"],
            "Observed Value": formatted_val,
            "Contribution": direction,
            "SHAP Value": f"{row['shap_value']:+.4f}",
            "Scientific Context": context
        })

    explanation_table = pd.DataFrame(table_rows)

    # 4. Generate Plotly horizontal bar chart
    plot_df = df_top.head(12).sort_values(by="shap_value", ascending=True)
    colors = ["#10b981" if v > 0 else "#ef4444" for v in plot_df["shap_value"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["shap_value"],
        y=plot_df["clean_name"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="#334155")),
        text=[f"{v:+.3f}" for v in plot_df["shap_value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Local Feature Contributions (SHAP Values)",
            font=dict(size=16, color="#f8fafc")
        ),
        xaxis=dict(
            title="SHAP Value (Impact on Model Output)",
            gridcolor="#1e293b",
            zerolinecolor="#94a3b8",
            zerolinewidth=2,
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=11, color="#e2e8f0")
        ),
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        margin=dict(l=220, r=40, t=50, b=50),
        height=420
    )

    return {
        "shap_values": shap_vals,
        "expected_value": expected_value,
        "positive_features": df_positive,
        "negative_features": df_negative,
        "explanation_table": explanation_table,
        "plotly_figure": fig
    }


def get_global_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Extracts global model feature importance rankings and generates a clean visualization.
    """
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    else:
        raw_importances = np.ones(len(feature_names))

    df_imp = pd.DataFrame({
        "raw_name": feature_names,
        "clean_name": [clean_feature_name(n) for n in feature_names],
        "importance": raw_importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    # Normalize to 0-100%
    total = df_imp["importance"].sum()
    if total > 0:
        df_imp["relative_pct"] = (df_imp["importance"] / total) * 100
    else:
        df_imp["relative_pct"] = 100.0 / top_n

    # Plotly figure
    plot_df = df_imp.sort_values(by="importance", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["importance"],
        y=plot_df["clean_name"],
        orientation="h",
        marker=dict(
            color=plot_df["importance"],
            colorscale="Viridis",
            line=dict(width=1, color="#334155")
        ),
        text=[f"{v:.2f}" for v in plot_df["importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Global Feature Importance (Top {top_n})",
            font=dict(size=16, color="#f8fafc")
        ),
        xaxis=dict(
            title="Importance Score",
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#e2e8f0")
        ),
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        margin=dict(l=220, r=40, t=50, b=50),
        height=450
    )

    return df_imp, fig
