# pages/page4_innovative_gmm.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
import sys
sys.path.append('..')
from common_components import apply_custom_css, render_navigation, render_sidebar_filters, apply_filters, render_global_kpis

st.set_page_config(page_title="Innovative GMM", page_icon="rocket", layout="wide")
apply_custom_css()

# === Prerequisites ===
if 'X_scaled' not in st.session_state:
    st.error("Please complete Step 2 (Preprocessing) first!")
    if st.button("Go to Preprocessing"):
        st.switch_page("pages/page2_preprocessing.py")
    st.stop()

if 'baseline_trained' not in st.session_state:
    st.error("Please complete Step 3 (Baseline GMM) first!")
    if st.button("Go to Baseline GMM"):
        st.switch_page("pages/page3_baseline_gmm.py")
    st.stop()

# Load session data
X_scaled = st.session_state['X_scaled']
feature_names = st.session_state['feature_names']
df = st.session_state['df_processed']
baseline = st.session_state['baseline_results']
n_clusters = baseline['n_clusters']

# Load original data for filters
@st.cache_data
def load_original_data():
    df = pd.read_csv("bank-additional//bank-additional-full.csv", sep=",")
    df.columns = df.columns.str.strip()
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df_original = load_original_data()

render_navigation("innovative_gmm")
render_sidebar_filters(df_original)
df_filtered = apply_filters(df_original)
render_global_kpis(df_filtered, df_original)

st.markdown("## rocket: Step 4: Innovative Hybrid GMM")

# === Top KPIs ===
if 'innovative_trained' in st.session_state and st.session_state['innovative_trained']:
    innovative = st.session_state['innovative_results']
    col1, col2, col3, col4, col5 = st.columns(5)
    improvement_sil = ((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette']) * 100
    improvement_db = ((baseline['davies_bouldin'] - innovative['davies_bouldin']) / baseline['davies_bouldin']) * 100
    improvement_ch = ((innovative['calinski'] - baseline['calinski']) / baseline['calinski']) * 100

    with col1:
        st.metric("Clusters (K)", innovative['n_clusters'])
    with col2:
        st.metric("Silhouette Score", f"{innovative['silhouette']:.4f}", delta=f"{improvement_sil:+.2f}%")
    with col3:
        st.metric("Davies-Bouldin", f"{innovative['davies_bouldin']:.4f}", delta=f"{improvement_db:+.2f}%")
    with col4:
        st.metric("Calinski-Harabasz", f"{innovative['calinski']:.1f}", delta=f"{improvement_ch:+.2f}%")
    with col5:
        st.metric("Training Time", f"{innovative['training_time']:.2f}s")
else:
    st.markdown("### chart_with_upwards_trend: Innovation Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    for col in [col1, col2, col3, col4, col5]:
        col.metric("•", "Not trained yet")

st.markdown("---")

# === Innovation Description ===
with st.expander("About This Innovation", expanded=True):
    st.markdown("""
    ### target: Hybrid GMM with Smart Initialization
    
    **Why this is better than standard GMM:**
    - K-Means++ initialization → avoids bad local minima
    - Optional soft feature weighting → emphasizes informative dimensions
    - More stable and reproducible clusters
    
    **You control the innovations:**
    - K-Means Initialization (highly recommended)
    - Soft Feature Weighting (safe & moderate)
    """)

st.markdown("---")

# === User Controls ===
st.markdown("### gear: Configure Innovative GMM")

col1, col2 = st.columns(2)
with col1:
    use_kmeans_init = st.checkbox("Use K-Means Initialization (Recommended)", value=True)
with col2:
    use_weighted_features = st.checkbox("Use Soft Feature Weighting", value=False, help="Gently emphasizes high-variance features")

with st.expander("Advanced Options"):
    kmeans_iters = st.slider("K-Means Pre-training Iterations", 10, 200, 50)

# === Safe Hybrid GMM Class ===
class HybridGMM:
    def __init__(self, n_components, use_kmeans=True, use_weights=False, kmeans_iters=50, random_state=42):
        self.n_components = n_components
        self.use_kmeans = use_kmeans
        self.use_weights = use_weights
        self.kmeans_iters = kmeans_iters
        self.random_state = random_state
        self.feature_weights = None
        self.gmm = None

    def _safe_data(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _kmeans_init(self, X):
        kmeans = KMeans(
            n_clusters=self.n_components,
            n_init=10,
            max_iter=self.kmeans_iters,
            random_state=self.random_state
        )
        kmeans.fit(X)
        return kmeans.cluster_centers_

    def _soft_feature_weights(self, X):
        # Soft inverse-std weighting (very safe)
        stds = np.std(X, axis=0)
        stds = np.clip(stds, 1e-8, None)
        weights = 1.0 / stds
        weights = np.sqrt(weights)  # soften
        weights = np.clip(weights, 0.3, 3.0)  # prevent extremes
        weights = weights / weights.sum()
        return weights

    def fit(self, X):
        X = self._safe_data(X)

        # Feature weighting
        if self.use_weights:
            self.feature_weights = self._soft_feature_weights(X)
            X = X * self.feature_weights
        else:
            self.feature_weights = np.ones(X.shape[1]) / X.shape[1]

        # K-Means initialization
        means_init = self._kmeans_init(X) if self.use_kmeans else None

        # Final GMM
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=300,
            n_init=1,
            tol=1e-4,
            random_state=self.random_state,
            means_init=means_init
        )
        self.gmm.fit(X)
        return self

    def predict(self, X):
        X = self._safe_data(X)
        if self.use_weights and self.feature_weights is not None:
            X = X * self.feature_weights
        return self.gmm.predict(X)

    def predict_proba(self, X):
        X = self._safe_data(X)
        if self.use_weights and self.feature_weights is not None:
            X = X * self.feature_weights
        return self.gmm.predict_proba(X)

# === Train Button ===
if st.button("rocket: Train Innovative Hybrid GMM", type="primary", use_container_width=True):
    with st.spinner("Training innovative model... This takes 5–15 seconds"):
        start = time.time()

        model = HybridGMM(
            n_components=n_clusters,
            use_kmeans=use_kmeans_init,
            use_weights=use_weighted_features,
            kmeans_iters=kmeans_iters,
            random_state=42
        )
        model.fit(X_scaled)
        labels = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        # === Safe Metric Computation ===
        unique_labels = len(np.unique(labels))
        if unique_labels < 2:
            st.error(f"Model collapsed to {unique_labels} cluster(s). Try disabling 'Soft Feature Weighting'.")
            silhouette = -1
            davies_bouldin = 999
            calinski = 0
        else:
            silhouette = silhouette_score(X_scaled, labels)
            davies_bouldin = davies_bouldin_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)

        training_time = time.time() - start

        # Save results
        st.session_state['innovative_results'] = {
            'n_clusters': n_clusters,
            'labels': labels,
            'probabilities': probs,
            'gmm_model': model,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski': calinski,
            'training_time': training_time,
            'feature_weights': model.feature_weights,
            'use_kmeans': use_kmeans_init,
            'use_weights': use_weighted_features
        }
        st.session_state['innovative_trained'] = True

    st.success(f"Training completed in {training_time:.2f}s!")
    st.rerun()

# === Display Results ===
if st.session_state.get('innovative_trained'):
    res = st.session_state['innovative_results']
    labels = res['labels']

    st.info(f"Innovations used: {'K-Means Init' if res['use_kmeans'] else ''} | {'Soft Weighting' if res['use_weights'] else ''}")

    # Cluster distribution
    st.markdown("### bar_chart: Cluster Distribution")
    counts = pd.Series(labels).value_counts().sort_index()
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure(go.Bar(
            x=[f"Cluster {i}" for i in counts.index],
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker_color=px.colors.qualitative.Pastel
        ))
        fig.update_layout(title="Samples per Cluster", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        for k, v in counts.items():
            st.metric(f"Cluster {k}", f"{v:,} ({v/len(labels):.1%})")

    # PCA Visualization
    st.markdown("### scatter_plot: 2D Cluster Visualization (PCA)")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=labels.astype(str),
        title="Innovative GMM Clusters (PCA Projection)",
        labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        opacity=0.7
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Feature weights (only if used)
    if res['use_weights']:
        st.markdown("### balance_scale: Feature Importance (Soft Weights)")
        weights_df = pd.DataFrame({
            "Feature": feature_names,
            "Weight": res['feature_weights']
        }).sort_values("Weight", ascending=False).head(15)

        fig = px.bar(weights_df, x="Weight", y="Feature", orientation="h",
                     title="Top 15 Weighted Features", color="Weight")
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Business Insights: Cluster Profiling (copied/adapted from Baseline GMM)
    # ----------------------------
    st.markdown("---")
    st.markdown("### 💼 Business Insights: Cluster Profiling")

    # Add labels to dataframe
    df_results = df.copy()
    df_results['cluster'] = labels

    st.markdown("#### Customer Segment Characteristics")

    # ensure cluster index range covers 0..n_clusters-1
    # use the n_clusters from results if available, otherwise fallback
    try:
        display_n_clusters = res.get('n_clusters', n_clusters)
    except Exception:
        display_n_clusters = n_clusters

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    for cluster_id in range(display_n_clusters):
        size = int(cluster_counts.get(cluster_id, 0))
        with st.expander(f"🔍 Cluster {cluster_id} Profile ({size:,} customers)"):
            cluster_data = df_results[df_results['cluster'] == cluster_id]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Demographics**")
                # Use try/except in case columns are missing in df_processed
                try:
                    st.write(f"Avg Age: {cluster_data['age'].mean():.1f} years")
                except Exception:
                    st.write("Avg Age: N/A")
                try:
                    st.write(f"Avg Duration: {cluster_data['duration'].mean():.0f} sec")
                except Exception:
                    st.write("Avg Duration: N/A")
                try:
                    st.write(f"Avg Campaign: {cluster_data['campaign'].mean():.2f}")
                except Exception:
                    st.write("Avg Campaign: N/A")

            with col2:
                st.markdown("**Most Common**")
                try:
                    job_mode = cluster_data['job'].mode().values[0]
                except Exception:
                    job_mode = "N/A"
                try:
                    edu_mode = cluster_data['education'].mode().values[0]
                except Exception:
                    edu_mode = "N/A"
                try:
                    contact_mode = cluster_data['contact'].mode().values[0]
                except Exception:
                    contact_mode = "N/A"

                st.write(f"Job: {job_mode}")
                st.write(f"Education: {edu_mode}")
                st.write(f"Contact: {contact_mode}")

            with col3:
                st.markdown("**Conversion**")
                try:
                    conv_rate = cluster_data['y_binary'].mean() * 100
                    st.metric("Conversion Rate", f"{conv_rate:.2f}%")
                    overall_conv = df['y_binary'].mean() * 100
                    if conv_rate > overall_conv:
                        st.success("🎯 High-value segment")
                    elif conv_rate < overall_conv * 0.5:
                        st.error("⚠️ Low-engagement segment")
                    else:
                        st.info("📊 Average segment")
                except Exception:
                    st.write("Conversion Rate: N/A")

    # Conversion comparison
    st.markdown("#### 📈 Conversion Rate by Cluster")

    try:
        conversion_by_cluster = df_results.groupby('cluster')['y_binary'].mean() * 100

        fig = px.bar(
            x=[f"Cluster {i}" for i in conversion_by_cluster.index],
            y=conversion_by_cluster.values,
            text=[f"{v:.2f}%" for v in conversion_by_cluster.values],
            title="Conversion Rate by Cluster",
            labels={'x': 'Cluster', 'y': 'Conversion Rate (%)'},
            color=conversion_by_cluster.values,
            color_continuous_scale='RdYlGn'
        )
        overall_avg = df['y_binary'].mean()*100
        fig.add_hline(
            y=overall_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overall Average ({overall_avg:.2f}%)"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.write("Conversion comparison not available (missing y_binary).")

    st.markdown("---")

    # Download Results
    st.markdown("### 💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        try:
            results_df = df_results[['age', 'job', 'education', 'duration',
                                     'campaign', 'y', 'cluster']]
        except Exception:
            # Fallback: include whatever columns exist
            results_df = df_results.copy()
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cluster Assignments",
            data=csv,
            file_name="innovative_gmm_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        metrics_df = pd.DataFrame({
            'Metric': ['Silhouette', 'Davies-Bouldin',
                       'Calinski-Harabasz', 'Training Time (s)'],
            'Value': [res['silhouette'], res['davies_bouldin'],
                      res['calinski'], res['training_time']]
        })
        csv = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Metrics",
            data=csv,
            file_name="innovative_gmm_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")
st.markdown("**Next → Go to Page 5: Comparison & Final Report**")
if st.button("Go to Comparison Page"):
    st.switch_page("pages/page5_comparison.py")
