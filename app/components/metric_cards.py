"""
metric_cards.py
---------------
Reusable UI components for displaying ADME prediction KPI cards, confidence badges,
and probability meters in OralAbsPredict.
"""

from typing import Dict, Any
import streamlit as st


def render_prediction_badge(prediction_data: Dict[str, Any], endpoint_name: str):
    """
    Renders a rich glassmorphic KPI card for an ADME endpoint prediction (HIA or HOB).
    Uses continuous HTML string with zero newlines to prevent CommonMark code block parsing.
    """
    is_high = prediction_data.get("prediction_class", 0) == 1
    short_label = prediction_data.get("short_label", "Unknown")
    full_label = prediction_data.get("label", "")
    prob_high = prediction_data.get("probability_high", 0.5)
    prob_low = prediction_data.get("probability_low", 0.5)
    confidence = prediction_data.get("confidence_percent", "0%")
    model_name = prediction_data.get("model_name", "CatBoost")

    # Theming
    if is_high:
        accent_color = "#10b981"  # Emerald
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "#10b981"
        badge_text = "HIGH ABSORPTION / BIOAVAILABILITY"
        pred_tag = "HIGH"
    else:
        accent_color = "#f43f5e"  # Rose / Crimson
        badge_bg = "rgba(244, 63, 94, 0.15)"
        badge_border = "#f43f5e"
        badge_text = "LOW / POOR ABSORPTION"
        pred_tag = "LOW"

    pct_high = round(prob_high * 100, 1)
    pct_low = round(prob_low * 100, 1)

    card_html = (
        f'<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); '
        f'border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid {accent_color}; border-radius: 16px; '
        f'padding: 22px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); margin-bottom: 20px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">'
        f'<span style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.08em; color: #94a3b8; text-transform: uppercase;">{endpoint_name}</span>'
        f'<span style="background: {badge_bg}; color: {accent_color}; border: 1px solid {badge_border}; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em;">{pred_tag}</span>'
        f'</div>'
        f'<div style="margin-bottom: 16px;">'
        f'<div style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">{short_label}</div>'
        f'<div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 4px;">{full_label}</div>'
        f'</div>'
        f'<div style="margin-bottom: 16px;">'
        f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 6px;">'
        f'<span style="color: #94a3b8;">High Probability</span>'
        f'<span style="font-weight: 700; color: {accent_color};">{pct_high}%</span>'
        f'</div>'
        f'<div style="width: 100%; height: 8px; background: rgba(51, 65, 85, 0.5); border-radius: 9999px; overflow: hidden;">'
        f'<div style="width: {pct_high}%; height: 100%; background: linear-gradient(90deg, #06b6d4, {accent_color}); border-radius: 9999px;"></div>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748b; margin-top: 4px;">'
        f'<span>Low: {pct_low}%</span>'
        f'<span>Confidence: {confidence}</span>'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.75rem; color: #64748b;">'
        f'<span>Algorithm: <strong style="color: #94a3b8;">{model_name}</strong></span>'
        f'<span>Threshold: <strong style="color: #94a3b8;">0.50</strong></span>'
        f'</div>'
        f'</div>'
    )

    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)
