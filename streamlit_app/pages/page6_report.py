# pages/page6_report.py
import streamlit as st
import pandas as pd
import sys

# === THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="Technical Report", page_icon="document", layout="wide")

# === Now import your common components (they should NOT call set_page_config) ===
sys.path.append('..')
from common_components import apply_custom_css, render_navigation, render_sidebar_filters, apply_filters, render_global_kpis

# Apply styling
apply_custom_css()

# Load original data for filters
@st.cache_data
def load_original_data():
    df = pd.read_csv("bank-additional//bank-additional-full.csv", sep=",")
    df.columns = df.columns.str.strip()
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df_original = load_original_data()

# Render shared components
render_navigation("report")
render_sidebar_filters(df_original)
df_filtered = apply_filters(df_original)
render_global_kpis(df_filtered, df_original)

# =================================================================
# ====================== PAGE CONTENT STARTS HERE =================
# =================================================================

st.title("Technical Report & Final Summary")

# Executive Summary
st.markdown("## Executive Summary")

if 'baseline_trained' in st.session_state and 'innovative_trained' in st.session_state:
    baseline = st.session_state['baseline_results']
    innovative = st.session_state['innovative_results']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        - **Goal**: Customer segmentation for targeted bank marketing
        - **Dataset**: Bank Marketing (UCI) — 41,188 records
        - **Features**: 20 (demographic + campaign behavior)
        - **Target**: Predict term deposit subscription (`yes`/`no`)
        """)
    
    with col2:
        sil_gain = ((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette']) * 100
        winner = "Innovative Hybrid GMM" if innovative['silhouette'] > baseline['silhouette'] else "Baseline GMM"
        
        st.markdown(f"""
        ### Key Results
        - **Optimal Clusters (K)**: `{baseline['n_clusters']}`
        - **Best Model**: **{winner}**
        - **Silhouette Gain**: **{sil_gain:+.2f}%**
        - **Training Time**: Innovative: `{innovative['training_time']:.2f}s` | Baseline: `{baseline['training_time']:.2f}s`
        """)
else:
    st.warning("Please complete both Baseline and Innovative GMM training first.")

st.markdown("---")

# Mathematical Foundation
st.markdown("## Mathematical Foundation")

with st.expander("Standard Gaussian Mixture Model", expanded=False):
    st.latex(r'''
    p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    ''')

with st.expander("Expectation-Maximization (EM) Algorithm", expanded=False):
    st.markdown("**E-Step**: Compute responsibilities")
    st.latex(r'''\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}''')
    st.markdown("**M-Step**: Update parameters")
    st.latex(r'''
    \pi_k = \frac{\sum_i \gamma_{ik}}{N}, \quad
    \boldsymbol{\mu}_k = \frac{\sum_i \gamma_{ik} \mathbf{x}_i}{\sum_i \gamma_{ik}}, \quad
    \boldsymbol{\Sigma}_k = \frac{\sum_i \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_i \gamma_{ik}}
    ''')

st.markdown("---")

# Innovation Highlights
st.markdown("## Innovation: Hybrid GMM with K-Means Initialization")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Why Standard GMM Fails Sometimes")
    st.markdown("""
    - Random initialization → poor local optima
    - Sensitive to starting points
    - Unstable results across runs
    """)

with col2:
    st.markdown("### Our Innovation")
    st.success("""
    **Hybrid Approach**:
    1. K-Means → find strong initial centroids
    2. Soft feature weighting (optional)
    3. GMM refinement with smart init
    
    → More stable, better, faster convergence
    """)

st.markdown("---")

# Performance Comparison Table
if 'baseline_trained' in st.session_state and 'innovative_trained' in st.session_state:
    st.markdown("## Performance Comparison")

    data = {
        "Metric": [
            "Initialization",
            "Feature Treatment",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz",
            "Training Time (s)"
        ],
        "Baseline GMM": [
            "Random",
            "Uniform",
            f"{baseline['silhouette']:.4f}",
            f"{baseline['davies_bouldin']:.4f}",
            f"{baseline['calinski']:.1f}",
            f"{baseline['training_time']:.2f}"
        ],
        "Innovative GMM": [
            "K-Means++",
            "Soft Weighted",
            f"{innovative['silhouette']:.4f}",
            f"{innovative['davies_bouldin']:.4f}",
            f"{innovative['calinski']:.1f}",
            f"{innovative['training_time']:.2f}"
        ]
    }
    comparison_df = pd.DataFrame(data)
    st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")

# Final Recommendations
st.markdown("## Recommendations")

if 'innovative_trained' in st.session_state:
    if innovative['silhouette'] > baseline['silhouette']:
        st.success("**Deploy the Innovative Hybrid GMM in production**")
    else:
        st.info("Baseline GMM is sufficient and slightly more stable")

st.markdown("""
### Actionable Insights:
- Use cluster profiles for **personalized campaign targeting**
- Focus marketing budget on **high-conversion clusters**
- Monitor cluster drift quarterly
- Retrain model every 6 months
""")

st.markdown("---")

# References
st.markdown("## References")
st.markdown("""
1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
2. Scikit-learn Documentation: GaussianMixture
3. Moro et al. (2014). Bank Marketing Dataset (UCI)
4. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding
""")

# Download Button
report_content = f"""
# Bank Marketing Customer Segmentation - Final Technical Report

**Date**: {pd.Timestamp('today').strftime('%Y-%m-%d')}
**Author**: Your Name
**Best Model**: {'Innovative Hybrid GMM' if st.session_state.get('innovative_trained') and st.session_state['innovative_results']['silhouette'] > st.session_state['baseline_results']['silhouette'] else 'Standard GMM'}

## Summary
- Optimal number of clusters: {st.session_state['baseline_results']['n_clusters']}
- Best silhouette score: {max(baseline['silhouette'], innovative['silhouette']):.4f}
- Recommended approach: Hybrid GMM with K-Means initialization

Project successfully completed.
"""

st.download_button(
    label="Download Full Report (Markdown)",
    data=report_content,
    file_name="bank_clustering_technical_report.md",
    mime="text/markdown"
)

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("Back to Comparison"):
        st.switch_page("pages/page5_comparison.py")
with col2:
    if st.button("Home"):
        st.switch_page("app.py")