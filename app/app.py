"""
app.py
------
Main Streamlit application entrypoint for OralAbsPredict:
AI-Powered Prediction of Human Intestinal Absorption & Human Oral Bioavailability.
"""

import os
import sys
import textwrap
from pathlib import Path

# Add project root and app directory to Python module search path
project_root = Path(__file__).resolve().parent.parent
app_dir = Path(__file__).resolve().parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Ensure 'app' is recognized as a package in sys.modules if Streamlit loaded app.py as module 'app'
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    sys.modules["app"].__path__ = [str(app_dir)]

import streamlit as st
import shap

from src.predictor import OralAbsPredictor

try:
    from app.views.home import render_home_page
    from app.views.prediction import render_prediction_page
    from app.views.performance import render_performance_page
    from app.views.explainability import render_explainability_page
    from app.views.about import render_about_page
except (ModuleNotFoundError, ImportError):
    from views.home import render_home_page
    from views.prediction import render_prediction_page
    from views.performance import render_performance_page
    from views.explainability import render_explainability_page
    from views.about import render_about_page

# Configure Streamlit page
st.set_page_config(
    page_title="OralAbsPredict | AI ADME Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System & CSS Styling
CUSTOM_CSS = textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, [data-testid="stSidebarNav"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Dark App Background */
    .stApp {
        background-color: #030712;
        color: #f3f4f6;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: #ffffff;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6);
        transform: translateY(-1px);
    }

    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #0f172a;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.95rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    /* Hide any leftover auto MPA navigation in sidebar */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
</style>
""").strip()

if hasattr(st, "html"):
    st.html(CUSTOM_CSS)
else:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_predictor_cached() -> OralAbsPredictor:
    """
    Cached instance of predictor to avoid deserializing weights repeatedly.
    """
    predictor = OralAbsPredictor(models_dir=str(project_root / "models"))
    return predictor


@st.cache_resource(show_spinner=False)
def load_explainers_cached(_predictor: OralAbsPredictor):
    """
    Cached TreeExplainers for instantaneous SHAP computation.
    """
    hia_exp = None
    hob_exp = None
    if _predictor.is_loaded:
        try:
            if _predictor.hia_model is not None:
                hia_exp = shap.TreeExplainer(_predictor.hia_model)
            if _predictor.hob_model is not None:
                hob_exp = shap.TreeExplainer(_predictor.hob_model)
        except Exception:
            pass
    return hia_exp, hob_exp


def main():
    predictor = load_predictor_cached()
    hia_explainer, hob_explainer = load_explainers_cached(predictor)

    # Sidebar Header
    with st.sidebar:
        brand_html = (
            '<div style="padding: 10px 0 20px 0; border-bottom: 1px solid #1e293b; margin-bottom: 20px;">'
            '<div style="display: flex; align-items: center; gap: 10px;">'
            '<span style="font-size: 2rem;">💊</span>'
            '<div>'
            '<div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">OralAbsPredict</div>'
            '<div style="font-size: 0.72rem; color: #38bdf8; font-weight: 600;">HIA & HOB AI PREDICTION</div>'
            '</div></div></div>'
        )
        if hasattr(st, "html"):
            st.html(brand_html)
        else:
            st.markdown(brand_html, unsafe_allow_html=True)

        if hasattr(st, "html"):
            st.html("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>NAVIGATION</p>")
        else:
            st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>NAVIGATION</p>", unsafe_allow_html=True)

        navigation = st.radio(
            "Select Page:",
            options=[
                "🏠 Home",
                "💊 Molecular Prediction",
                "📊 Model Performance",
                "🔍 Explainable AI (SHAP)",
                "📖 About & Methodology"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System Status in Sidebar
        if hasattr(st, "html"):
            st.html("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>SYSTEM STATUS</p>")
        else:
            st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;'>SYSTEM STATUS</p>", unsafe_allow_html=True)

        if predictor.is_loaded:
            status_html = (
                '<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 12px;">'
                '<div style="color: #10b981; font-weight: 700; font-size: 0.8rem;">● Models Ready (Online)</div>'
                '<div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">HIA & HOB CatBoost Classifiers</div>'
                '<div style="color: #64748b; font-size: 0.7rem;">706 Features (RDKit + Mordred + ECFP4)</div>'
                '</div>'
            )
            if hasattr(st, "html"):
                st.html(status_html)
            else:
                st.markdown(status_html, unsafe_allow_html=True)
        else:
            status_html = (
                '<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 12px;">'
                '<div style="color: #ef4444; font-weight: 700; font-size: 0.8rem;">● Models Not Loaded</div>'
                '<div style="color: #94a3b8; font-size: 0.72rem; margin-top: 4px;">Run train_hia.py and train_hob.py</div>'
                '</div>'
            )
            if hasattr(st, "html"):
                st.html(status_html)
            else:
                st.markdown(status_html, unsafe_allow_html=True)

        st.caption("OralAbsPredict v1.0.0")
        st.caption("College Project School Specification")

    # Route Page
    if navigation == "🏠 Home":
        render_home_page()
    elif navigation == "💊 Molecular Prediction":
        render_prediction_page(predictor, hia_explainer=hia_explainer, hob_explainer=hob_explainer)
    elif navigation == "📊 Model Performance":
        render_performance_page()
    elif navigation == "🔍 Explainable AI (SHAP)":
        render_explainability_page()
    elif navigation == "📖 About & Methodology":
        render_about_page()

    # Footer
    footer_html = (
        '<div style="margin-top: 60px; padding-top: 20px; border-top: 1px solid #1e293b; text-align: center; font-size: 0.78rem; color: #64748b;">'
        'OralAbsPredict — AI-Driven Human Intestinal Absorption & Oral Bioavailability Prediction Platform<br>'
        'Academic Research Prototype • Computational Screening Only • Not for Clinical or Diagnostic Use'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(footer_html)
    else:
        st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
