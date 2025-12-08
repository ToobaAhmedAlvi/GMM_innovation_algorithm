# pages/4_📈_Baseline_GMM.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Baseline GMM", page_icon="📈", layout="wide")

st.title("📈 Step 4: Baseline GMM Clustering")
st.markdown("---")

# Load and preprocess data
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv",sep=",")
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
    df['y_bin'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Define features
    numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                      'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 
                      'euribor3m', 'nr_employed']
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    # Encode categorical
    df_encoded = df.copy()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Build feature matrix
    encoded_features = [col + '_encoded' for col in categorical_cols]
    all_features = numerical_cols + encoded_features
    X = df_encoded[all_features].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, all_features, df_encoded, df

X_scaled, feature_names, df_encoded, df = load_and_preprocess()

# Theory Section
st.header("4.1 Gaussian Mixture Model Theory")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📐 Mathematical Foundation
    
    A Gaussian Mixture Model represents the probability distribution of data as 
    a weighted sum of multiple Gaussian (normal) distributions.
    
    #### **Model Definition:**
    
    ```
    p(X) = Σ(k=1 to K) πₖ · N(X | μₖ, Σₖ)
    ```
    
    Where:
    - **K** = number of mixture components (clusters)
    - **πₖ** = mixing coefficient for component k (weight)
    - **N(X | μₖ, Σₖ)** = Multivariate Gaussian distribution
    - **μₖ** = mean vector of component k
    - **Σₖ** = covariance matrix of component k
    
    #### **Multivariate Gaussian:**
    
    ```
    N(X | μ, Σ) = (1 / ((2π)^(d/2) |Σ|^(1/2))) · exp(-1/2 (X-μ)ᵀ Σ⁻¹ (X-μ))
    ```
    
    #### **Constraints:**
    
    ```
    Σ(k=1 to K) πₖ = 1
    πₖ > 0 for all k
    Σₖ is positive definite
    ```
    """)

with col2:
    st.info("""
    ### 🎯 Key Concepts
    
    **Soft Clustering:**
    - Each point has probability 
      of belonging to each cluster
    - More flexible than K-means
    
    **Covariance Matrix:**
    - Captures cluster shape
    - Allows ellipsoidal clusters
    - More expressive than K-means
    
    **Mixture Weights:**
    - Relative size of clusters
    - Sum to 1.0
    - Learned from data
    """)

# EM Algorithm
st.markdown("---")
st.subheader("4.1.1 Expectation-Maximization (EM) Algorithm")

tab1, tab2 = st.tabs(["📖 Algorithm Steps", "🔄 Visualization"])

with tab1:
    st.markdown("""
    The EM algorithm iteratively estimates GMM parameters:
    
    ### **E-Step (Expectation):**
    
    Calculate responsibility (probability) that component k generated point i:
    
    ```
    γ(i,k) = πₖ · N(xᵢ | μₖ, Σₖ) / Σ(j=1 to K) πⱼ · N(xᵢ | μⱼ, Σⱼ)
    ```
    
    ### **M-Step (Maximization):**
    
    Update parameters using responsibilities:
    
    ```
    Nₖ = Σ(i=1 to N) γ(i,k)
    
    πₖ = Nₖ / N
    
    μₖ = (1/Nₖ) · Σ(i=1 to N) γ(i,k) · xᵢ
    
    Σₖ = (1/Nₖ) · Σ(i=1 to N) γ(i,k) · (xᵢ - μₖ)(xᵢ - μₖ)ᵀ
    ```
    
    ### **Convergence:**
    
    Repeat E and M steps until log-likelihood change < tolerance:
    
    ```
    L(θ) = Σ(i=1 to N) log[Σ(k=1 to K) πₖ · N(xᵢ | μₖ, Σₖ)]
    ```
    """)

with tab2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/EM_Clustering_of_Old_Faithful_data.gif/400px-EM_Clustering_of_Old_Faithful_data.gif", 
             caption="EM Algorithm Visualization (source: Wikipedia)")
    st.markdown("""
    The animation shows how EM algorithm:
    1. **Initializes** cluster parameters randomly
    2. **E-Step:** Assigns points to clusters (soft assignment)
    3. **M-Step:** Updates cluster parameters
    4. **Repeats** until convergence
    """)

# Optimal Cluster Selection
st.markdown("---")
st.header("4.2 Determining Optimal Number of Clusters")

st.markdown("""
### 📊 Model Selection Criteria

We use **BIC (Bayesian Information Criterion)** and **AIC (Akaike Information Criterion)** to select K:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### **BIC Formula:**
    ```
    BIC = -2·log(L) + p·log(N)
    ```
    
    - **L** = likelihood
    - **p** = number of parameters
    - **N** = number of samples
    - **Lower is better**
    - Penalizes complexity more
    """)

with col2:
    st.markdown("""
    #### **AIC Formula:**
    ```
    AIC = -2·log(L) + 2·p
    ```
    
    - **L** = likelihood
    - **p** = number of parameters
    - **Lower is better**
    - Less penalty for complexity
    """)

# Compute BIC/AIC for different K
st.subheader("4.2.1 Model Selection Analysis")

with st.spinner("Computing BIC and AIC scores for different cluster counts..."):
    
    @st.cache_data
    def compute_model_selection(X_scaled):
        n_components_range = range(2, 11)
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, 
                                 random_state=42,
                                 covariance_type='full',
                                 max_iter=200,
                                 n_init=10)
            gmm.fit(X_scaled)
            bic_scores.append(gmm.bic(X_scaled))
            aic_scores.append(gmm.aic(X_scaled))
            
            labels = gmm.predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        return list(n_components_range), bic_scores, aic_scores, silhouette_scores
    
    n_range, bic_scores, aic_scores, sil_scores = compute_model_selection(X_scaled)

# Plot selection criteria
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_range, y=bic_scores, mode='lines+markers',
                            name='BIC', line=dict(color='blue', width=2),
                            marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=n_range, y=aic_scores, mode='lines+markers',
                            name='AIC', line=dict(color='red', width=2),
                            marker=dict(size=8)))
    fig.update_layout(title='BIC and AIC Scores vs Number of Clusters',
                     xaxis_title='Number of Clusters (K)',
                     yaxis_title='Score (lower is better)',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_range, y=sil_scores, mode='lines+markers',
                            name='Silhouette', line=dict(color='green', width=2),
                            marker=dict(size=8)))
    fig.update_layout(title='Silhouette Score vs Number of Clusters',
                     xaxis_title='Number of Clusters (K)',
                     yaxis_title='Silhouette Score (higher is better)',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

# Determine optimal K
optimal_k_bic = n_range[np.argmin(bic_scores)]
optimal_k_aic = n_range[np.argmin(aic_scores)]
optimal_k_sil = n_range[np.argmax(sil_scores)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Optimal K (BIC)", optimal_k_bic, 
             f"Score: {min(bic_scores):.2f}")
with col2:
    st.metric("Optimal K (AIC)", optimal_k_aic,
             f"Score: {min(aic_scores):.2f}")
with col3:
    st.metric("Optimal K (Silhouette)", optimal_k_sil,
             f"Score: {max(sil_scores):.4f}")

# Use BIC as primary criterion
optimal_k = optimal_k_bic
st.info(f"**Selected K = {optimal_k}** (based on BIC criterion)")

# Fit Final Model
st.markdown("---")
st.header("4.3 Baseline GMM Model Training")

with st.spinner(f"Training GMM with {optimal_k} components..."):
    
    @st.cache_data
    def fit_baseline_gmm(X_scaled, n_components):
        start_time = time.time()
        
        gmm = GaussianMixture(n_components=n_components,
                             covariance_type='full',
                             random_state=42,
                             max_iter=200,
                             n_init=10,
                             verbose=0)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        probabilities = gmm.predict_proba(X_scaled)
        
        training_time = time.time() - start_time
        
        return gmm, labels, probabilities, training_time
    
    baseline_gmm, baseline_labels, baseline_probs, training_time = fit_baseline_gmm(X_scaled, optimal_k)

# Training Results
st.subheader("4.3.1 Training Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Converged", "✅ Yes" if baseline_gmm.converged_ else "❌ No")
with col2:
    st.metric("Iterations", baseline_gmm.n_iter_)
with col3:
    st.metric("Log-Likelihood", f"{baseline_gmm.lower_bound_:.2f}")
with col4:
    st.metric("Training Time", f"{training_time:.2f}s")

# Cluster sizes
st.subheader("4.3.2 Cluster Distribution")

cluster_counts = pd.Series(baseline_labels).value_counts().sort_index()

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                 labels={'x': 'Cluster', 'y': 'Number of Samples'},
                 title='Samples per Cluster')
    fig.update_traces(marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:optimal_k])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Cluster Statistics:**")
    for cluster_id in range(optimal_k):
        count = cluster_counts[cluster_id]
        percentage = (count / len(baseline_labels)) * 100
        st.metric(f"Cluster {cluster_id}", f"{count:,} ({percentage:.1f}%)")

# Performance Metrics
st.markdown("---")
st.header("4.4 Baseline Performance Evaluation")

# Calculate metrics
baseline_silhouette = silhouette_score(X_scaled, baseline_labels)
baseline_davies_bouldin = davies_bouldin_score(X_scaled, baseline_labels)
baseline_calinski_harabasz = calinski_harabasz_score(X_scaled, baseline_labels)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Silhouette Score
    **Range:** [-1, 1]
    - **1:** Perfect clustering
    - **0:** Overlapping clusters
    - **-1:** Wrong clustering
    """)
    st.metric("Silhouette Score", f"{baseline_silhouette:.4f}")
    
    if baseline_silhouette > 0.5:
        st.success("Excellent clustering!")
    elif baseline_silhouette > 0.3:
        st.info("Good clustering structure")
    else:
        st.warning("Weak clustering structure")

with col2:
    st.markdown("""
    ### 📉 Davies-Bouldin Index
    **Range:** [0, ∞)
    - **Lower is better**
    - Measures cluster separation
    - < 1.0 is good
    """)
    st.metric("Davies-Bouldin", f"{baseline_davies_bouldin:.4f}")
    
    if baseline_davies_bouldin < 1.0:
        st.success("Well-separated clusters!")
    elif baseline_davies_bouldin < 1.5:
        st.info("Moderate separation")
    else:
        st.warning("Poor separation")

with col3:
    st.markdown("""
    ### 📈 Calinski-Harabasz
    **Range:** [0, ∞)
    - **Higher is better**
    - Variance ratio criterion
    - > 1000 is good
    """)
    st.metric("Calinski-Harabasz", f"{baseline_calinski_harabasz:.2f}")
    
    if baseline_calinski_harabasz > 1000:
        st.success("Strong cluster definition!")
    elif baseline_calinski_harabasz > 500:
        st.info("Moderate definition")
    else:
        st.warning("Weak definition")

# Visualization
st.markdown("---")
st.header("4.5 Cluster Visualization")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create interactive visualization
fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                 color=baseline_labels.astype(str),
                 labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% var)',
                        'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% var)',
                        'color': 'Cluster'},
                 title='Baseline GMM Clusters (PCA Projection)',
                 opacity=0.6)

# Add cluster centers
pca_means = pca.transform(baseline_gmm.means_)
fig.add_trace(go.Scatter(x=pca_means[:, 0], y=pca_means[:, 1],
                        mode='markers',
                        marker=dict(size=20, color='red', symbol='x', line=dict(width=2)),
                        name='Cluster Centers',
                        showlegend=True))

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Cluster Profiling
st.markdown("---")
st.header("4.6 Cluster Profiling & Business Interpretation")

# Add cluster labels to dataframe
df_results = df.copy()
df_results['cluster'] = baseline_labels

st.subheader("4.6.1 Demographic & Campaign Characteristics")

profile_data = []
for cluster_id in range(optimal_k):
    cluster_data = df_results[df_results['cluster'] == cluster_id]
    
    profile_data.append({
        'Cluster': cluster_id,
        'Size': len(cluster_data),
        'Size %': f"{len(cluster_data)/len(df_results)*100:.1f}%",
        'Avg Age': f"{cluster_data['age'].mean():.1f}",
        'Avg Duration': f"{cluster_data['duration'].mean():.0f}s",
        'Avg Campaign': f"{cluster_data['campaign'].mean():.2f}",
        'Conversion Rate': f"{cluster_data['y_bin'].mean()*100:.2f}%",
        'Most Common Job': cluster_data['job'].mode().values[0],
        'Most Common Education': cluster_data['education'].mode().values[0]
    })

profile_df = pd.DataFrame(profile_data)
st.dataframe(profile_df, use_container_width=True)

# Detailed cluster analysis
st.subheader("4.6.2 Detailed Cluster Analysis")

selected_cluster = st.selectbox("Select cluster for detailed analysis:", 
                                range(optimal_k))

cluster_data = df_results[df_results['cluster'] == selected_cluster]

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### Cluster {selected_cluster} Demographics")
    
    st.metric("Total Customers", f"{len(cluster_data):,}")
    st.metric("Average Age", f"{cluster_data['age'].mean():.1f} years")
    st.metric("Conversion Rate", f"{cluster_data['y_bin'].mean()*100:.2f}%")
    
    st.markdown("**Top Job Categories:**")
    st.write(cluster_data['job'].value_counts().head(5))

with col2:
    st.markdown(f"### Cluster {selected_cluster} Campaign Metrics")
    
    st.metric("Avg Contact Duration", f"{cluster_data['duration'].mean():.0f} seconds")
    st.metric("Avg Campaign Contacts", f"{cluster_data['campaign'].mean():.2f}")
    st.metric("Avg Previous Contacts", f"{cluster_data['previous'].mean():.2f}")
    
    st.markdown("**Education Distribution:**")
    st.write(cluster_data['education'].value_counts())

# Save results
st.markdown("---")
st.subheader("💾 Save Baseline Results")

# Store in session state for comparison
if 'baseline_results' not in st.session_state:
    st.session_state.baseline_results = {
        'optimal_k': optimal_k,
        'silhouette': baseline_silhouette,
        'davies_bouldin': baseline_davies_bouldin,
        'calinski_harabasz': baseline_calinski_harabasz,
        'labels': baseline_labels,
        'training_time': training_time,
        'converged': baseline_gmm.converged_,
        'iterations': baseline_gmm.n_iter_
    }
    st.success("✅ Baseline results saved to session state!")

# Download results
col1, col2 = st.columns(2)

with col1:
    results_df = df_results[['age', 'job', 'education', 'duration', 'campaign', 'y', 'cluster']]
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Clustering Results (CSV)",
        data=csv,
        file_name="baseline_gmm_results.csv",
        mime="text/csv",
    )

with col2:
    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score'],
        'Value': [baseline_silhouette, baseline_davies_bouldin, baseline_calinski_harabasz]
    })
    csv = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Performance Metrics (CSV)",
        data=csv,
        file_name="baseline_metrics.csv",
        mime="text/csv",
    )

st.success("✅ Baseline GMM clustering complete! Proceed to **Innovative Deep GMM** page.")
