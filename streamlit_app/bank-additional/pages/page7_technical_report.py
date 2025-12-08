# pages/7_📑_Technical_Report.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Technical Report", page_icon="📑", layout="wide")

st.title("📑 Technical Report: Mathematical Foundation & Complexity Analysis")
st.markdown("---")

# Table of Contents
st.markdown("""
## 📚 Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Innovation Details](#innovation-details)
3. [Complexity Analysis](#complexity-analysis)
4. [Theoretical Justification](#theoretical-justification)
5. [References](#references)
""")

st.markdown("---")

# Section 1: Mathematical Foundation
st.header("1. Mathematical Foundation")

st.markdown("""
### 1.1 Standard Gaussian Mixture Model (Baseline)

The traditional GMM models the probability distribution of data as a weighted sum of Gaussian components:

""")

st.latex(r'''
p(\mathbf{X}) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{X} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
''')

st.markdown("""
Where:
- **K** = number of mixture components (clusters)
- **πₖ** = mixing coefficient for component k (weight), with Σπₖ = 1
- **N(X | μₖ, Σₖ)** = Multivariate Gaussian distribution
- **μₖ** = mean vector of component k (d-dimensional)
- **Σₖ** = covariance matrix of component k (d × d)

#### **Multivariate Gaussian Distribution:**
""")

st.latex(r'''
\mathcal{N}(\mathbf{X} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
\frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} 
\exp\left(-\frac{1}{2}(\mathbf{X}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X}-\boldsymbol{\mu})\right)
''')

st.markdown("---")

st.subheader("1.2 Expectation-Maximization (EM) Algorithm")

st.markdown("""
The EM algorithm iteratively estimates GMM parameters through two steps:

#### **E-Step (Expectation):**

Calculate the responsibility (posterior probability) that component k generated observation i:
""")

st.latex(r'''
\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
''')

st.markdown("""
#### **M-Step (Maximization):**

Update parameters using the computed responsibilities:
""")

st.latex(r'''
\begin{aligned}
N_k &= \sum_{i=1}^{N} \gamma_{ik} \\
\pi_k &= \frac{N_k}{N} \\
\boldsymbol{\mu}_k &= \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i \\
\boldsymbol{\Sigma}_k &= \frac{1}{N_k} \sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
\end{aligned}
''')

st.markdown("""
#### **Convergence Criterion:**

Repeat E and M steps until the log-likelihood converges:
""")

st.latex(r'''
\mathcal{L}(\theta) = \sum_{i=1}^{N} \log \left[\sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right]
''')

st.markdown("---")

# Section 2: Innovation Details
st.header("2. Innovation: Deep-Based GMM")

st.markdown("""
Our innovation introduces three key enhancements to the standard GMM framework:
""")

st.subheader("2.1 Variable Contribution Analysis (GDVG)")

st.markdown("""
**Objective:** Classify variables into convergence-related and diversity-related groups.

#### **Methodology:**

For each variable j:

1. **Perturbation:** Generate perturbed samples
""")

st.latex(r'''
\mathbf{X}'_j = \mathbf{X}_j + \boldsymbol{\epsilon}, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
''')

st.markdown("""
2. **Direction Analysis:** Fit GMM on perturbed data and compute direction vector
""")

st.latex(r'''
\mathbf{d} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_0
''')

st.markdown("""
3. **Pearson Correlation Coefficient:** Calculate PCC with convergence direction
""")

st.latex(r'''
\rho_{j} = \frac{\text{Cov}(\mathbf{d}, \mathbf{n})}{\sigma_{\mathbf{d}} \cdot \sigma_{\mathbf{n}}}
''')

st.markdown("""
Where **n** is the normal vector to hyperplane (convergence direction):
""")

st.latex(r'''
\mathbf{n} = \left[\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}}, \ldots, \frac{1}{\sqrt{d}}\right]
''')

st.markdown("""
4. **Classification:** Use secondary GMM on PCC scores to classify variables
""")

st.latex(r'''
\text{Variable } j \text{ is} \begin{cases}
\text{Convergence-related} & \text{if } \rho_j < \text{threshold} \\
\text{Diversity-related} & \text{if } \rho_j \geq \text{threshold}
\end{cases}
''')

st.markdown("---")

st.subheader("2.2 Chaos-Based Linkage Identification (LIMC)")

st.markdown("""
**Objective:** Detect non-linear interactions between variables.

#### **Chaos Map (Logistic Map):**
""")

st.latex(r'''
C_{t+1} = \mu \cdot C_t \cdot (1 - C_t), \quad \mu = 3.9
''')

st.markdown("""
#### **Interaction Detection:**

For variables i and j:

1. Generate chaotic sequence **C**
2. Apply perturbations:
""")

st.latex(r'''
\begin{aligned}
\mathbf{X}'_i &= \mathbf{X}_i + \alpha \cdot \mathbf{C} \\
\mathbf{X}'_j &= \mathbf{X}_j + \alpha \cdot \mathbf{C}
\end{aligned}
''')

st.markdown("""
3. Calculate interaction score:
""")

st.latex(r'''
I(i,j) = \left|\text{Corr}(\mathbf{X}'_i, \mathbf{X}'_j) - \text{Corr}(\mathbf{X}_i, \mathbf{X}_j)\right|
''')

st.markdown("""
4. Link variables if interaction exceeds threshold:
""")

st.latex(r'''
\text{Link}(i, j) \text{ if } I(i,j) > \tau
''')

st.markdown("---")

st.subheader("2.3 Adaptive Multi-Phase Clustering")

st.markdown("""
#### **Phase 1: Convergence Optimization**

Focus on cluster separation using convergence-related variables:
""")

st.latex(r'''
\text{Minimize: } J_{\text{conv}} = \sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{x}_i[\text{conv\_vars}] - \boldsymbol{\mu}_k[\text{conv\_vars}]\|^2
''')

st.markdown("""
#### **Phase 2: Diversity Refinement**

For each cluster Cₖ, evaluate intra-cluster structure:
""")

st.latex(r'''
\begin{aligned}
\text{BIC}_{\text{split}} &= \log(n_k) \cdot p - 2 \cdot \log(\mathcal{L}_{\text{split}}) \\
\text{BIC}_{\text{single}} &= \log(n_k) \cdot \frac{p}{2} - 2 \cdot \log(\mathcal{L}_{\text{single}})
\end{aligned}
''')

st.markdown("""
Refine cluster if:
""")

st.latex(r'''
\text{BIC}_{\text{split}} < \alpha \cdot \text{BIC}_{\text{single}}
''')

st.markdown("""
#### **Phase 3: Integrated Clustering**

Final GMM with weighted feature importance:
""")

st.latex(r'''
\boldsymbol{\Sigma}_k = \mathbf{W}_{\text{conv}} \cdot \boldsymbol{\Sigma}_k[\text{conv\_vars}] + 
\mathbf{W}_{\text{div}} \cdot \boldsymbol{\Sigma}_k[\text{div\_vars}]
''')

st.markdown("---")

# Section 3: Complexity Analysis
st.header("3. Complexity Analysis")

st.subheader("3.1 Baseline GMM Complexity")

st.markdown("""
#### **Time Complexity per EM Iteration:**

**E-Step:** Computing responsibilities
""")

st.latex(r'''
O(N \cdot K \cdot d^2)
''')

st.markdown("""
Where:
- N = number of samples
- K = number of clusters
- d = number of features

**M-Step:** Updating parameters
""")

st.latex(r'''
O(N \cdot K \cdot d^2)
''')

st.markdown("""
**Total for convergence (T iterations):**
""")

st.latex(r'''
O(T \cdot N \cdot K \cdot d^2)
''')

st.markdown("""
#### **Space Complexity:**
""")

st.latex(r'''
O(N \cdot d + K \cdot d^2 + N \cdot K)
''')

st.markdown("---")

st.subheader("3.2 Innovative Deep GMM Complexity")

st.markdown("""
#### **Additional Operations:**

**1. Variable Contribution Analysis:**
""")

st.latex(r'''
O(d \cdot P \cdot N_s \cdot d^2) = O(d^3 \cdot P \cdot N_s)
''')

st.markdown("""
Where:
- P = number of perturbations (typically 10)
- Nₛ = sample size (Nₛ << N, typically 100)

**2. Chaos-Based Linkage:**
""")

st.latex(r'''
O(d_{\text{conv}}^2 \cdot N_s + d_{\text{div}}^2 \cdot N_s)
''')

st.markdown("""
**3. Adaptive Clustering:**

Same as baseline: 
""")

st.latex(r'''
O(T \cdot N \cdot K \cdot d^2)
''')

st.markdown("""
#### **Total Time Complexity:**
""")

st.latex(r'''
T_{\text{total}} = O(d^3 \cdot P \cdot N_s) + O(d^2 \cdot N_s) + O(T \cdot N \cdot K \cdot d^2)
''')

st.markdown("""
**Dominant Term:** O(T · N · K · d²)

**Overhead Ratio:**
""")

st.latex(r'''
\text{Overhead} = \frac{d^3 \cdot P \cdot N_s}{T \cdot N \cdot K \cdot d^2} = \frac{d \cdot P \cdot N_s}{T \cdot N \cdot K}
''')

st.markdown("""
For our dataset (d=20, P=10, Nₛ=100, T=200, N=40000, K=5):
""")

st.latex(r'''
\text{Overhead} = \frac{20 \times 10 \times 100}{200 \times 40000 \times 5} \approx 0.0005 = 0.05\%
''')

st.markdown("**Conclusion:** Negligible overhead compared to baseline GMM!")

st.markdown("---")

# Section 4: Theoretical Justification
st.header("4. Theoretical Justification")

st.subheader("4.1 Why Variable Grouping Improves Clustering")

st.markdown("""
**Theorem (Informal):**

Separating variables by their contribution to cluster structure reduces the effective 
dimensionality of the optimization problem in each phase, leading to:

1. **Better Convergence:** Convergence variables provide stronger signals for cluster separation
2. **Improved Stability:** Reducing noise from diversity variables in initial phase
3. **Enhanced Interpretability:** Variables grouped by function

#### **Mathematical Intuition:**

Standard GMM optimizes:
""")

st.latex(r'''
\mathcal{L}(\theta) = \sum_{i=1}^{N} \log\left[\sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right]
''')

st.markdown("""
Deep-based GMM decomposes this:
""")

st.latex(r'''
\mathcal{L}(\theta) \approx \alpha \cdot \mathcal{L}_{\text{conv}}(\theta_{\text{conv}}) + 
\beta \cdot \mathcal{L}_{\text{div}}(\theta_{\text{div}})
''')

st.markdown("""
Where:
- **α > β** (higher weight on convergence)
- This decomposition reduces search space
- Provides better initialization for final clustering
""")

st.markdown("---")

st.subheader("4.2 Chaos-Based Interaction Detection")

st.markdown("""
**Why Chaos?**

Linear correlation misses non-linear interactions. Chaotic perturbation:
- Explores non-linear relationships
- Provides diverse perturbation patterns
- Sensitive to interaction effects

**Property:** Chaotic systems amplify small differences, making subtle interactions detectable.

#### **Lyapunov Exponent:**

For logistic map with μ = 3.9:
""")

st.latex(r'''
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \log|f'(x_i)| > 0
''')

st.markdown("""
Positive Lyapunov exponent indicates chaotic behavior and sensitivity to initial conditions.
""")

st.markdown("---")

# Section 5: Performance Metrics
st.header("5. Performance Metrics Interpretation")

st.subheader("5.1 Silhouette Score")

st.markdown("""
**Formula:**
""")

st.latex(r'''
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
''')

st.markdown("""
Where:
- **a(i)** = average distance to points in same cluster
- **b(i)** = average distance to nearest neighboring cluster

**Range:** [-1, 1]
- +1: Perfect clustering
- 0: On cluster boundary
- -1: Misclassified
""")

st.subheader("5.2 Davies-Bouldin Index")

st.markdown("""
**Formula:**
""")

st.latex(r'''
DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left[\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}\right]
''')

st.markdown("""
Where:
- **σᵢ** = average distance within cluster i
- **d(cᵢ, cⱼ)** = distance between cluster centers

**Range:** [0, ∞)
- Lower is better
- 0 = perfect clustering
""")

st.subheader("5.3 Calinski-Harabasz Score")

st.markdown("""
**Formula:**
""")

st.latex(r'''
CH = \frac{\sum_{k=1}^{K} n_k \|\mathbf{c}_k - \mathbf{c}\|^2 / (K-1)}
{\sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{x}_i - \mathbf{c}_k\|^2 / (N-K)}
''')

st.markdown("""
**Interpretation:** Ratio of between-cluster to within-cluster dispersion

**Range:** [0, ∞)
- Higher is better
- Measures cluster definition quality
""")

st.markdown("---")

# Section 6: Comparison Summary
st.header("6. Experimental Results Summary")

if 'baseline_results' in st.session_state and 'innovative_results' in st.session_state:
    baseline = st.session_state.baseline_results
    innovative = st.session_state.innovative_results
    
    results_table = pd.DataFrame({
        'Metric': [
            'Silhouette Score',
            'Davies-Bouldin Index',
            'Calinski-Harabasz Score',
            'Training Time (s)',
            'Iterations'
        ],
        'Baseline': [
            f"{baseline['silhouette']:.4f}",
            f"{baseline['davies_bouldin']:.4f}",
            f"{baseline['calinski_harabasz']:.2f}",
            f"{baseline['training_time']:.2f}",
            baseline['iterations']
        ],
        'Innovative': [
            f"{innovative['silhouette']:.4f}",
            f"{innovative['davies_bouldin']:.4f}",
            f"{innovative['calinski_harabasz']:.2f}",
            f"{innovative['training_time']:.2f}",
            innovative['iterations']
        ],
        'Improvement': [
            f"{((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette'] * 100):+.2f}%",
            f"{((baseline['davies_bouldin'] - innovative['davies_bouldin']) / baseline['davies_bouldin'] * 100):+.2f}%",
            f"{((innovative['calinski_harabasz'] - baseline['calinski_harabasz']) / baseline['calinski_harabasz'] * 100):+.2f}%",
            f"{((baseline['training_time'] - innovative['training_time']) / baseline['training_time'] * 100):+.2f}%",
            f"{baseline['iterations'] - innovative['iterations']:+d}"
        ]
    })
    
    st.dataframe(results_table, use_container_width=True)
else:
    st.info("Run both Baseline and Innovative GMM to see comparison results.")

st.markdown("---")

# Section 7: References
st.header("7. References")

st.markdown("""
1. **Wang, M., Li, X., Chen, L., Chen, H., Chen, C., & Liu, M. (2025).** 
   A deep-based Gaussian mixture model algorithm for large-scale many objective optimization. 
   *Applied Soft Computing*, 172, 112874.

2. **Bishop, C. M. (2006).** 
   *Pattern Recognition and Machine Learning*. Springer.

3. **Reynolds, D. A. (2009).** 
   Gaussian mixture models. 
   *Encyclopedia of Biometrics*, 741, 659-663.

4. **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).** 
   Maximum likelihood from incomplete data via the EM algorithm. 
   *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.

5. **Moro, S., Cortez, P., & Rita, P. (2014).** 
   A data-driven approach to predict the success of bank telemarketing. 
   *Decision Support Systems*, 62, 22-31.

6. **Pedregosa, F., et al. (2011).** 
   Scikit-learn: Machine learning in Python. 
   *Journal of Machine Learning Research*, 12, 2825-2830.
""")

st.markdown("---")

# Download Report
st.subheader("💾 Download Technical Report")

report_content = """
# Deep-Based Gaussian Mixture Model for Bank Marketing Clustering
## Technical Report

### Executive Summary
This report presents the mathematical foundation, implementation details, and 
complexity analysis of an innovative clustering approach for bank marketing data.

### Key Innovation
The Deep-Based GMM extends standard Gaussian Mixture Models through:
1. Variable Contribution Analysis (GDVG)
2. Chaos-Based Linkage Identification (LIMC)
3. Adaptive Multi-Phase Clustering

### Complexity Analysis
- Time Complexity: O(T·N·K·d²) + O(d³·P·Nₛ)
- Space Complexity: O(N·d + K·d² + N·K)
- Overhead: ~0.05% compared to baseline

### Performance Improvement
See comparison section for detailed metrics.

### Conclusion
The innovative approach maintains baseline complexity while providing
enhanced clustering quality through intelligent variable grouping.
"""

st.download_button(
    label="📥 Download Report (Markdown)",
    data=report_content,
    file_name="technical_report.md",
    mime="text/markdown",
)

st.success("✅ Technical report complete!")

st.info("""
**📚 For Full Project Code:**
- Check the individual page implementations
- Review the mathematical derivations above
- Refer to the original research paper for deeper theoretical insights
""")
