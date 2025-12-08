# pages/page3_baseline_gmm.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import time
import sys
sys.path.append('..')
from common_components import apply_custom_css, render_navigation, render_sidebar_filters, apply_filters, render_global_kpis

st.set_page_config(page_title="Baseline GMM", page_icon="📈", layout="wide")

# Apply custom styling
apply_custom_css()

# Check prerequisites
if 'X_scaled' not in st.session_state:
    st.error("⚠️ Please complete Step 2 (Preprocessing) first!")
    if st.button("Go to Preprocessing"):
        st.switch_page("pages/page2_preprocessing.py")
    st.stop()

# Load data from session
X_scaled = st.session_state['X_scaled']
feature_names = st.session_state['feature_names']
df = st.session_state['df_processed']
recommended_k = st.session_state.get('recommended_k', 3)

# Load original for filters
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional//bank-additional-full.csv", sep=",")
    df.columns = df.columns.str.strip()
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df_original = load_data()

# Render navigation
render_navigation("baseline_gmm")

# Render sidebar filters
render_sidebar_filters(df_original)

# Apply filters to original data
df_filtered = apply_filters(df_original)

# Render global KPIs
render_global_kpis(df_filtered, df_original)

# Page Content
st.markdown("## 📈 Step 3: Baseline Gaussian Mixture Model")

# Top KPIs
st.markdown("### 📊 Model Performance Metrics")

if 'baseline_trained' in st.session_state:
    baseline = st.session_state['baseline_results']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Clusters (K)", baseline['n_clusters'])
    with col2:
        st.metric("Silhouette", f"{baseline['silhouette']:.4f}",
                 help="Higher is better (max 1.0)")
    with col3:
        st.metric("Davies-Bouldin", f"{baseline['davies_bouldin']:.4f}",
                 help="Lower is better")
    with col4:
        st.metric("Calinski-Harabasz", f"{baseline['calinski']:.2f}",
                 help="Higher is better")
    with col5:
        st.metric("Training Time", f"{baseline['training_time']:.2f}s")
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

# Model Configuration
st.markdown("### ⚙️ Configure GMM Parameters")

with st.expander("ℹ️ Parameter Guide", expanded=False):
    st.markdown("""
    **GMM Parameters:**
    
    1. **Number of Clusters (K)**: Customer segments to identify
    2. **Covariance Type**: Shape of cluster distributions
       - `full`: Most flexible (ellipsoids, any orientation)
       - `tied`: All clusters share same shape
       - `diag`: Axis-aligned ellipsoids
       - `spherical`: Circular/spherical clusters
    3. **Max Iterations**: EM algorithm iterations
    4. **Random Seed**: For reproducibility
    """)

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_clusters = st.number_input(
        "Number of Clusters (K)",
        min_value=2, max_value=10,
        value=recommended_k,
        help="Recommended from Step 2"
    )

with col2:
    covariance_type = st.selectbox(
        "Covariance Type",
        options=['full', 'tied', 'diag', 'spherical'],
        index=0
    )

with col3:
    max_iter = st.number_input(
        "Max Iterations",
        min_value=50, max_value=500,
        value=200,
        step=50
    )

with col4:
    random_state = st.number_input(
        "Random Seed",
        min_value=0, max_value=100,
        value=42
    )

# Advanced options
with st.expander("🔧 Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        n_init = st.number_input(
            "Number of Initializations",
            min_value=1, max_value=20,
            value=10
        )
    
    with col2:
        tol = st.number_input(
            "Convergence Tolerance",
            min_value=1e-6, max_value=1e-2,
            value=1e-3,
            format="%.6f"
        )

st.markdown("---")

# Train Model
if st.button("🚀 Train Baseline GMM", type="primary", use_container_width=True):
    
    with st.spinner("Training Gaussian Mixture Model..."):
        
        start_time = time.time()
        
        # Train GMM
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            tol=tol,
            random_state=random_state,
            verbose=0
        )
        
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        probabilities = gmm.predict_proba(X_scaled)
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        
        # Store results
        st.session_state['baseline_results'] = {
            'n_clusters': n_clusters,
            'covariance_type': covariance_type,
            'max_iter': max_iter,
            'n_init': n_init,
            'tol': tol,
            'random_state': random_state,
            'labels': labels,
            'probabilities': probabilities,
            'gmm_model': gmm,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski': calinski,
            'training_time': training_time,
            'converged': gmm.converged_,
            'n_iter': gmm.n_iter_
        }
        
        st.session_state['baseline_trained'] = True
        
    st.success(f"✅ Training completed in {training_time:.2f} seconds!")
    st.rerun()

# Display results
if 'baseline_trained' in st.session_state:
    
    baseline = st.session_state['baseline_results']
    labels = baseline['labels']
    gmm = baseline['gmm_model']
    
    # Training Info
    st.markdown("### ✅ Training Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "✅ Converged" if baseline['converged'] else "❌ Not Converged"
        st.metric("Convergence Status", status)
    
    with col2:
        st.metric("Iterations Used", f"{baseline['n_iter']}/{baseline['max_iter']}")
    
    with col3:
        log_likelihood = gmm.lower_bound_
        st.metric("Log-Likelihood", f"{log_likelihood:.2f}")
    
    st.markdown("---")
    
    # Cluster Distribution
    st.markdown("### 📊 Cluster Distribution")
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Cluster {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                text=cluster_counts.values,
                textposition='auto',
                marker_color=px.colors.qualitative.Set2[:len(cluster_counts)]
            )
        ])
        fig.update_layout(
            title="Samples per Cluster",
            xaxis_title="Cluster",
            yaxis_title="Number of Samples",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Cluster Sizes")
        for cluster_id in cluster_counts.index:
            count = cluster_counts[cluster_id]
            pct = (count / len(labels)) * 100
            st.metric(f"Cluster {cluster_id}", 
                     f"{count:,} ({pct:.1f}%)")
    
    st.markdown("---")
    
    # PCA Visualization
    st.markdown("### 🔍 Cluster Visualization (PCA)")
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=labels.astype(str),
        labels={
            'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
            'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
            'color': 'Cluster'
        },
        title='Clusters in 2D Space (PCA)',
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Add cluster centers
    centers_pca = pca.transform(gmm.means_)
    fig.add_trace(
        go.Scatter(
            x=centers_pca[:, 0],
            y=centers_pca[:, 1],
            mode='markers',
            marker=dict(size=20, color='red', symbol='x', 
                       line=dict(width=2, color='white')),
            name='Cluster Centers',
            showlegend=True
        )
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Business Insights
    st.markdown("### 💼 Business Insights: Cluster Profiling")
    
    # Add labels to dataframe
    df_results = df.copy()
    df_results['cluster'] = labels
    
    st.markdown("#### Customer Segment Characteristics")
    
    for cluster_id in range(n_clusters):
        with st.expander(f"🔍 Cluster {cluster_id} Profile ({cluster_counts[cluster_id]:,} customers)"):
            
            cluster_data = df_results[df_results['cluster'] == cluster_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                st.write(f"Avg Age: {cluster_data['age'].mean():.1f} years")
                st.write(f"Avg Duration: {cluster_data['duration'].mean():.0f} sec")
                st.write(f"Avg Campaign: {cluster_data['campaign'].mean():.2f}")
            
            with col2:
                st.markdown("**Most Common**")
                st.write(f"Job: {cluster_data['job'].mode().values[0]}")
                st.write(f"Education: {cluster_data['education'].mode().values[0]}")
                st.write(f"Contact: {cluster_data['contact'].mode().values[0]}")
            
            with col3:
                st.markdown("**Conversion**")
                conv_rate = cluster_data['y_binary'].mean() * 100
                st.metric("Conversion Rate", f"{conv_rate:.2f}%")
                
                if conv_rate > df['y_binary'].mean() * 100:
                    st.success("🎯 High-value segment")
                elif conv_rate < df['y_binary'].mean() * 100 * 0.5:
                    st.error("⚠️ Low-engagement segment")
                else:
                    st.info("📊 Average segment")
    
    # Conversion comparison
    st.markdown("#### 📈 Conversion Rate by Cluster")
    
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
    fig.add_hline(
        y=df['y_binary'].mean()*100, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Overall Average ({df['y_binary'].mean()*100:.2f}%)"
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Download Results
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        results_df = df_results[['age', 'job', 'education', 'duration', 
                                 'campaign', 'y', 'cluster']]
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cluster Assignments",
            data=csv,
            file_name="baseline_gmm_clusters.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        metrics_df = pd.DataFrame({
            'Metric': ['Silhouette', 'Davies-Bouldin', 
                      'Calinski-Harabasz', 'Training Time (s)'],
            'Value': [baseline['silhouette'], baseline['davies_bouldin'],
                     baseline['calinski'], baseline['training_time']]
        })
        csv = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Metrics",
            data=csv,
            file_name="baseline_gmm_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )
