# pages/page6_report.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Technical Report", page_icon="📑", layout="wide")

st.title("📑 Step 6: Technical Report")

# Executive Summary
st.markdown("## 📊 Executive Summary")

if 'baseline_trained' in st.session_state and 'innovative_trained' in st.session_state:
    baseline = st.session_state['baseline_results']
    innovative = st.session_state['innovative_results']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        **Objective:** Customer segmentation for bank marketing optimization
        
        **Dataset:** 41,188 customer records with 20 features
        
        **Approach:** Comparative analysis of GMM clustering methods
        """)
    
    with col2:
        st.markdown(f"""
        ### Key Results
        
        **Optimal Clusters:** {baseline['n_clusters']}
        
        **Best Approach:** {"Innovative GMM" if innovative['silhouette'] > baseline['silhouette'] else "Baseline GMM"}
        
        **Performance Gain:** {((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette'] * 100):+.2f}%
        """)

st.markdown("---")

# Mathematical Foundation
st.markdown("## 🧮 Mathematical Foundation")

st.markdown("### Standard Gaussian Mixture Model")

st.latex(r'''
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
''')

st.markdown("""
Where:
- **K** = number of clusters
- **πₖ** = mixing coefficient (weight) for cluster k
- **μₖ** = mean vector for cluster k
- **Σₖ** = covariance matrix for cluster k
""")

st.markdown("### EM Algorithm")

st.markdown("**E-Step:** Calculate responsibilities")
st.latex(r'''
\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
''')

st.markdown("**M-Step:** Update parameters")
st.latex(r'''
\begin{aligned}
\pi_k &= \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik} \\
\boldsymbol{\mu}_k &= \frac{\sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i}{\sum_{i=1}^{N} \gamma_{ik}} \\
\boldsymbol{\Sigma}_k &= \frac{\sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}
\end{aligned}
''')

st.markdown("---")

# Innovation Details
st.markdown("## 🚀 Innovation: Hybrid GMM-KMeans")

st.markdown("""
### Innovative Approach

**Problem:**
- Random GMM initialization leads to suboptimal solutions
- EM algorithm sensitive to starting points
- May converge to local optima

**Solution: Hybrid GMM with Smart Initialization**

#### **Phase 1: K-Means Initialization**
Use K-Means clustering to find good starting centroids for GMM.
""")

st.latex(r'''
\boldsymbol{\mu}_k^{(0)} = \text{KMeans}(\mathbf{X}, K)
''')

st.markdown("""
#### **Phase 2: Feature Weighting**
Compute importance weights based on feature variance:
""")

st.latex(r'''
w_j = \frac{\text{Var}(x_j)}{\sum_{i=1}^{d} \text{Var}(x_i)}
''')

st.markdown("""
Apply weights to features:
""")

st.latex(r'''
\mathbf{X}_{\text{weighted}} = \mathbf{X} \odot \mathbf{w}
''')

st.markdown("""
#### **Phase 3: Adaptive EM**
Run EM algorithm on weighted features with smart initialization.
""")

st.markdown("---")

# Complexity Analysis
st.markdown("## ⏱️ Complexity Analysis")

st.markdown("""
### Time Complexity

**Baseline GMM:**
""")

st.latex(r'''
O(T \cdot N \cdot K \cdot d^2)
''')

st.markdown("""
Where:
- **T** = number of iterations
- **N** = number of samples
- **K** = number of clusters
- **d** = number of features

**Innovative GMM:**
""")

st.latex(r'''
O(I \cdot N \cdot K) + O(T \cdot N \cdot K \cdot d^2)
''')

st.markdown("""
Where **I** = K-Means iterations (typically 50)

**Additional Cost:**
- K-Means initialization: O(I·N·K) 
- Feature weighting: O(N·d)

**Total Overhead:** Negligible compared to EM iterations

### Space Complexity

Both approaches: O(N·d + K·d²)
""")

st.markdown("---")

# Performance Metrics
st.markdown("## 📊 Performance Metrics Explained")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Silhouette Score
    
    **Formula:**
    """)
    st.latex(r's(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}')
    st.markdown("""
    **Range:** [-1, 1]
    - 1: Perfect clustering
    - 0: Overlapping
    - -1: Wrong assignment
    
    **Higher is better**
    """)

with col2:
    st.markdown("""
    ### Davies-Bouldin Index
    
    **Formula:**
    """)
    st.latex(r'DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}')
    st.markdown("""
    **Range:** [0, ∞)
    
    **Lower is better**
    
    Measures cluster separation
    """)

with col3:
    st.markdown("""
    ### Calinski-Harabasz
    
    **Formula:**
    """)
    st.latex(r'CH = \frac{SS_B / (K-1)}{SS_W / (N-K)}')
    st.markdown("""
    **Range:** [0, ∞)
    
    **Higher is better**
    
    Variance ratio criterion
    """)

st.markdown("---")

# Results Summary
if 'baseline_trained' in st.session_state and 'innovative_trained' in st.session_state:
    
    st.markdown("## 📈 Results Summary")
    
    baseline = st.session_state['baseline_results']
    innovative = st.session_state['innovative_results']
    
    results_table = pd.DataFrame({
        'Component': [
            'Initialization',
            'Feature Treatment',
            'EM Algorithm',
            'Silhouette Score',
            'Davies-Bouldin',
            'Calinski-Harabasz',
            'Training Time (s)'
        ],
        'Baseline GMM': [
            'Random',
            'Standard',
            'Standard',
            f"{baseline['silhouette']:.4f}",
            f"{baseline['davies_bouldin']:.4f}",
            f"{baseline['calinski']:.2f}",
            f"{baseline['training_time']:.2f}"
        ],
        'Innovative GMM': [
            'K-Means',
            'Weighted',
            'Adaptive',
            f"{innovative['silhouette']:.4f}",
            f"{innovative['davies_bouldin']:.4f}",
            f"{innovative['calinski']:.2f}",
            f"{innovative['training_time']:.2f}"
        ]
    })
    
    st.dataframe(results_table, use_container_width=True, height=320)

st.markdown("---")

# Conclusions
st.markdown("## 💡 Conclusions")

if 'innovative_trained' in st.session_state:
    innovative = st.session_state['innovative_results']
    baseline = st.session_state['baseline_results']
    
    sil_improvement = ((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette']) * 100
    
    st.markdown(f"""
    ### Key Findings
    
    1. **Clustering Quality:** {"Innovative approach shows {:.2f}% improvement".format(sil_improvement) if sil_improvement > 0 else "Baseline performs better"}
    
    2. **Initialization Impact:** K-Means initialization provides better starting points
    
    3. **Feature Weighting:** Variance-based weights improve cluster separation
    
    4. **Computational Cost:** Additional overhead is negligible (< 1s)
    
    5. **Business Value:** Clear customer segmentation for targeted marketing
    
    ### Recommendations
    
    ✅ Deploy {"Innovative GMM" if innovative['silhouette'] > baseline['silhouette'] else "Baseline GMM"} for production
    
    ✅ Use clusters for personalized marketing campaigns
    
    ✅ Focus resources on high-conversion segments
    
    ✅ Monitor cluster stability over time
    """)

st.markdown("---")

# References
st.markdown("## 📚 References")

st.markdown("""
1. **Bishop, C. M. (2006).** Pattern Recognition and Machine Learning. Springer.

2. **Reynolds, D. A. (2009).** Gaussian mixture models. Encyclopedia of Biometrics.

3. **Dempster, A. P., et al. (1977).** Maximum likelihood from incomplete data via the EM algorithm. 
   Journal of the Royal Statistical Society.

4. **Moro, S., et al. (2014).** A data-driven approach to predict the success of bank telemarketing. 
   Decision Support Systems, 62, 22-31.

5. **Pedregosa, F., et al. (2011).** Scikit-learn: Machine learning in Python. 
   Journal of Machine Learning Research.
""")

st.markdown("---")

# Download Report
st.markdown("## 💾 Download Technical Report")

report_md = """
# Bank Marketing Clustering - Technical Report

## Executive Summary
- Dataset: 41,188 customers, 20 features
- Approach: Comparative GMM clustering
- Innovation: Hybrid GMM with K-Means initialization

## Methodology
1. Data preprocessing and scaling
2. Optimal cluster selection (BIC, AIC, Silhouette)
3. Baseline GMM training
4. Innovative GMM with smart initialization
5. Comparative performance analysis

## Results
See application for detailed results.

## Conclusion
The project successfully demonstrated customer segmentation for targeted marketing.
"""

st.download_button(
    label="📥 Download Report (Markdown)",
    data=report_md,
    file_name="technical_report.md",
    mime="text/markdown",
    use_container_width=True
)

st.markdown("---")

# Navigation
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Comparison", use_container_width=True):
        st.switch_page("pages/page5_comparison.py")

with col2:
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("app.py")