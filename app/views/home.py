"""
home.py
-------
Home page overview for OralAbsPredict.
"""

import textwrap
import streamlit as st


def render_home_page():
    hero_html = (
        '<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%); '
        'border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 36px 32px; margin-bottom: 28px; '
        'box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);">'
        '<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">'
        '<span style="background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); '
        'padding: 4px 12px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;">AI CHEMINFORMATICS FRAMEWORK</span>'
        '<span style="font-size: 0.8rem; color: #94a3b8;">ADMET Screening Platform</span>'
        '</div>'
        '<h1 style="font-size: 2.6rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.03em; margin: 0 0 12px 0; line-height: 1.15;">OralAbsPredict</h1>'
        '<p style="font-size: 1.15rem; color: #38bdf8; font-weight: 600; margin-bottom: 16px;">'
        'A Data-Driven Framework to Predict Human Intestinal Absorption (HIA) and Human Oral Bioavailability (HOB) from Chemical Structures'
        '</p>'
        '<p style="font-size: 0.98rem; color: #cbd5e1; line-height: 1.65; max-width: 900px; margin: 0;">'
        'Poor oral absorption is one of the primary reasons for drug candidate attrition during preclinical drug discovery. '
        'OralAbsPredict is an AI-powered cheminformatics system that calculates 2D physicochemical descriptors, Mordred topological indices, '
        'and Morgan circular fingerprints directly from molecular SMILES to predict HIA and HOB, offering transparent decision attribution via SHAP explainable AI.'
        '</p>'
        '</div>'
    )
    if hasattr(st, "html"):
        st.html(hero_html)
    else:
        st.markdown(hero_html, unsafe_allow_html=True)

    # 3 Highlights Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        c1_html = (
            '<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 14px; padding: 20px; height: 100%;">'
            '<div style="font-size: 1.6rem; margin-bottom: 8px;">🧪</div>'
            '<div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Molecular Representation</div>'
            '<div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">'
            'Extracts 706 features per molecule combining Lipinski descriptors, Veber parameters, Mordred 2D topological graph indices, and 512-bit Morgan circular fingerprints (ECFP4).'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c1_html)
        else:
            st.markdown(c1_html, unsafe_allow_html=True)

    with col2:
        c2_html = (
            '<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 14px; padding: 20px; height: 100%;">'
            '<div style="font-size: 1.6rem; margin-bottom: 8px;">⚡</div>'
            '<div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Benchmark ML Models</div>'
            '<div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">'
            'Systematic comparison across 6 algorithms (CatBoost, LightGBM, Random Forest, Balanced Random Forest, SVM, Logistic Regression) addressing real-world ADME class imbalance.'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c2_html)
        else:
            st.markdown(c2_html, unsafe_allow_html=True)

    with col3:
        c3_html = (
            '<div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 14px; padding: 20px; height: 100%;">'
            '<div style="font-size: 1.6rem; margin-bottom: 8px;">🔍</div>'
            '<div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px;">Explainable AI (SHAP)</div>'
            '<div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">'
            'Local and global SHAP (SHapley Additive exPlanations) attribution identifies the physical chemistry and structural moieties driving intestinal permeation and oral bioavailability.'
            '</div></div>'
        )
        if hasattr(st, "html"):
            st.html(c3_html)
        else:
            st.markdown(c3_html, unsafe_allow_html=True)

    if hasattr(st, "html"):
        st.html("<div style='height: 28px;'></div>")
    else:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Problem Statement Section
    st.markdown("""
### Official Project Problem Statement
> *"Poor oral absorption is one of the primary reasons for drug failure during the early stages of drug discovery,
leading to increased development time and cost. Experimental evaluation of Human Intestinal Absorption (HIA) and
Human Oral Bioavailability (HOB) is expensive, time-consuming, and resource-intensive. Existing computational methods
often suffer from limited datasets, class imbalance, and poor interpretability. This project aims to develop an
AI-driven framework that predicts HIA and HOB directly from molecular structures represented as SMILES strings.
The system employs machine learning and molecular descriptors to accurately classify orally active compounds before
synthesis. It also provides feature interpretation using explainable AI techniques and identifies important molecular
substructures responsible for oral absorption. This approach enables rapid screening of drug candidates, reducing
experimental costs and accelerating drug discovery."*
    """)

    if hasattr(st, "html"):
        st.html("<div style='height: 20px;'></div>")
    else:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Scientific Disclaimer Box
    disc_html = (
        '<div style="background: rgba(234, 179, 8, 0.08); border: 1px solid rgba(234, 179, 8, 0.3); '
        'border-left: 4px solid #eab308; border-radius: 12px; padding: 18px 22px; margin-top: 10px;">'
        '<div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #fde047; font-size: 0.95rem;">'
        '<span>⚠️ SCIENTIFIC & ETHICAL DISCLAIMER</span>'
        '</div>'
        '<div style="font-size: 0.84rem; color: #cbd5e1; margin-top: 6px; line-height: 1.55;">'
        'OralAbsPredict is a computational research prototype developed for cheminformatics screening and educational evaluation. '
        'Predictions generated by this machine learning framework do <strong>NOT</strong> constitute clinical diagnoses, therapeutic prescriptions, '
        'or pharmacological guarantees. Computational predictions cannot substitute for empirical in vitro permeability assays (e.g., Caco-2, PAMPA) '
        'or in vivo pharmacokinetic studies.'
        '</div></div>'
    )
    if hasattr(st, "html"):
        st.html(disc_html)
    else:
        st.markdown(disc_html, unsafe_allow_html=True)
