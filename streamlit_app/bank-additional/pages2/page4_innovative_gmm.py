# pages/page4_innovative_gmm.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import time

st.set_page_config(page_title="Innovative Deep GMM", page_icon="🚀", layout="wide")

# Check prerequisites
if 'X_scaled' not in st.session_state:
    st.error("⚠️ Please complete Step 2 (Preprocessing) first!")
    if st.button("Go to Preprocessing"):
        st.switch_page("pages/page2_preprocessing.py")
    st.stop()

if 'baseline_trained' not in st.session_state:
    st.error("⚠️ Please complete Step 3 (Baseline GMM) first!")
    if st.button("Go to Baseline GMM"):
        st.switch_page("pages/page3_baseline_gmm.py")
    st.stop()

# Load data
X_scaled = st.session_state['X_scaled']
feature_names = st.session_state['feature_names']
df = st.session_state['df_processed']
baseline = st.session_state['baseline_results']
n_clusters = baseline['n_clusters']

st.title("🚀 Step 4: Innovative Deep-Based GMM")

# Top KPIs
st.markdown("## 📊 Innovation Performance Metrics")

if 'innovative_trained' in st.session_state:
    innovative = st.session_state['innovative_results']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Clusters (K)", innovative['n_clusters'])
    with col2:
        improvement = ((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette']) * 100
        st.metric("Silhouette", f"{innovative['silhouette']:.4f}",
                 delta=f"{improvement:+.2f}%")
    with col3:
        improvement = ((baseline['davies_bouldin'] - innovative['davies_bouldin']) / baseline['davies_bouldin']) * 100
        st.metric("Davies-Bouldin", f"{innovative['davies_bouldin']:.4f}",
                 delta=f"{improvement:+.2f}%")
    with col4:
        improvement = ((innovative['calinski'] - baseline['calinski']) / baseline['calinski']) * 100
        st.metric("Calinski-Harabasz", f"{innovative['calinski']:.2f}",
                 delta=f"{improvement:+.2f}%")
    with col5:
        st.metric("Training Time", f"{innovative['training_time']:.2f}s")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Clusters (K)", "Not trained")
    with col2:
        st.metric("Silhouette", "--")
    with col3:
        st.metric("Davies-Bouldin", "--")
    with col4:
        st.metric("Calinski-Harabasz", "--")
    with col5:
        st.metric("Training Time", "--")

st.markdown("---")

# Innovation Description
st.markdown("## 💡 Innovation: Hybrid GMM with K-Means Initialization")

with st.expander("ℹ️ About This Innovation", expanded=True):
    st.markdown("""
    ### 🎯 Innovative Approach: Hybrid GMM-KMeans
    
    **Problem with Standard GMM:**
    - Random initialization can lead to suboptimal solutions
    - EM algorithm sensitive to starting points
    - May converge to local optima
    
    **Our Innovation:**
    1. **Intelligent Initialization** - Use K-Means to find good starting cluster centers
    2. **Weighted Features** - Give more importance to high-variance features
    3. **Adaptive Refinement** - Iteratively refine clusters with smart updates
    
    **Expected Benefits:**
    - Better convergence
    - More stable results
    - Improved cluster quality
    - Faster training time
    
    **Mathematical Foundation:**
    ```
    Phase 1: K-Means Pre-clustering
    - Initialize with K-Means centroids
    - Faster than random initialization
    
    Phase 2: Weighted GMM
    - Feature weights based on variance
    - W_i = var(feature_i) / Σ var(features)
    
    Phase 3: Adaptive EM
    - Standard EM with smart initialization
    - Better starting point → Better solution
    ```
    """)

st.markdown("---")

# User Parameters
st.markdown("## ⚙️ Configure Innovative GMM Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    use_kmeans_init = st.checkbox(
        "Use K-Means Initialization",
        value=True,
        help="Initialize GMM with K-Means centroids for better starting point"
    )

with col2:
    use_weighted_features = st.checkbox(
        "Use Weighted Features",
        value=True,
        help="Give more weight to high-variance features"
    )

with col3:
    adaptive_em = st.checkbox(
        "Adaptive EM Algorithm",
        value=True,
        help="Use adaptive convergence criteria"
    )

# Advanced options
with st.expander("🔧 Advanced Innovation Parameters"):
    col1, col2 = st.columns(2)
    
    with col1:
        kmeans_iterations = st.number_input(
            "K-Means Iterations",
            min_value=10, max_value=100,
            value=50,
            help="Iterations for K-Means initialization"
        )
    
    with col2:
        feature_weight_threshold = st.slider(
            "Feature Weight Threshold",
            min_value=0.0, max_value=1.0,
            value=0.1,
            help="Minimum weight for features"
        )

st.markdown("---")

# Hybrid GMM Implementation
class HybridGMM:
    """Innovative Hybrid GMM with K-Means Initialization"""
    
    def __init__(self, n_components, use_kmeans=True, use_weights=True, 
                 adaptive=True, kmeans_iter=50, random_state=42):
        self.n_components = n_components
        self.use_kmeans = use_kmeans
        self.use_weights = use_weights
        self.adaptive = adaptive
        self.kmeans_iter = kmeans_iter
        self.random_state = random_state
        self.feature_weights = None
        self.gmm = None
        
    def _safe_normalize(self, X):
        """Safely normalize data to avoid NaN"""
        X_clean = np.copy(X)
        
        # Replace any NaN or inf with 0
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_clean
    
    def _kmeans_initialization(self, X):
        """Initialize with K-Means for better starting point"""
        from sklearn.cluster import KMeans
        
        X_clean = self._safe_normalize(X)
        
        kmeans = KMeans(
            n_clusters=self.n_components,
            max_iter=self.kmeans_iter,
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(X_clean)
        
        return kmeans.cluster_centers_
    
    def _compute_feature_weights(self, X):
        """Compute feature importance weights based on variance"""
        X_clean = self._safe_normalize(X)
        
        variances = np.var(X_clean, axis=0)
        
        # Normalize weights
        weights = variances / (np.sum(variances) + 1e-10)
        
        # Apply threshold
        weights = np.maximum(weights, 0.01)  # Minimum weight
        weights = weights / np.sum(weights)  # Renormalize
        
        return weights
    
    def fit(self, X):
        """Fit the Hybrid GMM"""
        X_clean = self._safe_normalize(X)
        
        # Phase 1: K-Means Initialization
        if self.use_kmeans:
            initial_means = self._kmeans_initialization(X_clean)
            means_init = initial_means
        else:
            means_init = 'random'
        
        # Phase 2: Feature Weighting
        if self.use_weights:
            self.feature_weights = self._compute_feature_weights(X_clean)
            # Apply weights
            X_weighted = X_clean * self.feature_weights
        else:
            self.feature_weights = np.ones(X_clean.shape[1]) / X_clean.shape[1]
            X_weighted = X_clean
        
        # Ensure no NaN in weighted data
        X_weighted = self._safe_normalize(X_weighted)
        
        # Phase 3: Train GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=300 if self.adaptive else 200,
            n_init=1,  # We already have good initialization
            tol=1e-4 if self.adaptive else 1e-3,
            random_state=self.random_state,
            means_init=means_init if self.use_kmeans else None
        )
        
        self.gmm.fit(X_weighted)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels"""
        X_clean = self._safe_normalize(X)
        
        if self.use_weights:
            X_weighted = X_clean * self.feature_weights
        else:
            X_weighted = X_clean
        
        X_weighted = self._safe_normalize(X_weighted)
        
        return self.gmm.predict(X_weighted)
    
    def predict_proba(self, X):
        """Predict cluster probabilities"""
        X_clean = self._safe_normalize(X)
        
        if self.use_weights:
            X_weighted = X_clean * self.feature_weights
        else:
            X_weighted = X_clean
        
        X_weighted = self._safe_normalize(X_weighted)
        
        return self.gmm.predict_proba(X_weighted)

# Train Button
if st.button("🚀 Train Innovative Hybrid GMM", type="primary", use_container_width=True):
    
    with st.spinner("Training Innovative GMM..."):
        
        start_time = time.time()
        
        # Train Hybrid GMM
        hybrid_gmm = HybridGMM(
            n_components=n_clusters,
            use_kmeans=use_kmeans_init,
            use_weights=use_weighted_features,
            adaptive=adaptive_em,
            kmeans_iter=kmeans_iterations,
            random_state=42
        )
        
        hybrid_gmm.fit(X_scaled)
        labels = hybrid_gmm.predict(X_scaled)
        probabilities = hybrid_gmm.predict_proba(X_scaled)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        
        # Store results
        st.session_state['innovative_results'] = {
            'n_clusters': n_clusters,
            'labels': labels,
            'probabilities': probabilities,
            'gmm_model': hybrid_gmm,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski': calinski,
            'training_time': training_time,
            'feature_weights': hybrid_gmm.feature_weights,
            'use_kmeans': use_kmeans_init,
            'use_weights': use_weighted_features,
            'adaptive': adaptive_em
        }
        
        st.session_state['innovative_trained'] = True
        
    st.success(f"✅ Training completed in {training_time:.2f} seconds!")
    st.rerun()

# Display results if trained
if 'innovative_trained' in st.session_state:
    
    innovative = st.session_state['innovative_results']
    labels = innovative['labels']
    
    # Show what was used
    st.markdown("### ✅ Innovation Features Used:")
    innovations_used = []
    if innovative['use_kmeans']:
        innovations_used.append("✓ K-Means Initialization")
    if innovative['use_weights']:
        innovations_used.append("✓ Weighted Features")
    if innovative['adaptive']:
        innovations_used.append("✓ Adaptive EM")
    
    st.info(" | ".join(innovations_used))
    
    st.markdown("---")
    
    # Cluster Distribution
    st.markdown("## 📊 Cluster Distribution")
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Cluster {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                text=cluster_counts.values,
                textposition='auto',
                marker_color=px.colors.qualitative.Pastel[:len(cluster_counts)]
            )
        ])
        fig.update_layout(
            title="Samples per Cluster (Innovative GMM)",
            xaxis_title="Cluster",
            yaxis_title="Number of Samples",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Cluster Sizes")
        for cluster_id in cluster_counts.index:
            count = cluster_counts[cluster_id]
            pct = (count / len(labels)) * 100
            st.metric(f"Cluster {cluster_id}", 
                     f"{count:,} ({pct:.1f}%)")
    
    st.markdown("---")
    
    # PCA Visualization
    st.markdown("## 🔍 Cluster Visualization (PCA Projection)")
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=labels.astype(str),
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                'color': 'Cluster'},
        title='Innovative GMM Clusters in 2D Space',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance (if weighted)
    if innovative['use_weights']:
        st.markdown("## 🎯 Feature Importance Weights")
        
        weights = innovative['feature_weights']
        weight_df = pd.DataFrame({
            'Feature': feature_names,
            'Weight': weights
        }).sort_values('Weight', ascending=False).head(15)
        
        fig = px.bar(
            weight_df,
            x='Weight',
            y='Feature',
            orientation='h',
            title='Top 15 Most Important Features',
            color='Weight',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Navigation
st.markdown("## 🚀 Next Steps")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if 'innovative_trained' in st.session_state:
        if st.button("➡️ Proceed to Performance Comparison", 
                    use_container_width=True, type="primary"):
            st.switch_page("pages/page5_comparison.py")
    else:
        st.warning("⚠️ Please train the innovative model first")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Baseline GMM", use_container_width=True):
        st.switch_page("pages/page3_baseline_gmm.py")

with col2:
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("app.py")