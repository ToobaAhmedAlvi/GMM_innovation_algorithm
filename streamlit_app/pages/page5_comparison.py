# pages/page5_comparison.py
# At top of every page
import sys
sys.path.append('..')
from common_components import apply_custom_css, render_navigation, render_sidebar_filters, apply_filters, render_global_kpis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA



# Then your page-specific content
st.set_page_config(page_title="Comparison", page_icon="📊", layout="wide")
# After st.set_page_config
apply_custom_css()
# Load original data for filters
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv", sep=",")
    df.columns = df.columns.str.strip()
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df_original = load_data()

# Render components
render_navigation("current_page_name")  # e.g., "comparison" or "report"
render_sidebar_filters(df_original)
df_filtered = apply_filters(df_original)
render_global_kpis(df_filtered, df_original)

# Check prerequisites
if 'baseline_trained' not in st.session_state:
    st.error("⚠️ Please complete Baseline GMM first!")
    if st.button("Go to Baseline GMM"):
        st.switch_page("pages/page3_baseline_gmm.py")
    st.stop()

if 'innovative_trained' not in st.session_state:
    st.error("⚠️ Please complete Innovative GMM first!")
    if st.button("Go to Innovative GMM"):
        st.switch_page("pages/page4_innovative_gmm.py")
    st.stop()

# Load data
X_scaled = st.session_state['X_scaled']
df = st.session_state['df_processed']
baseline = st.session_state['baseline_results']
innovative = st.session_state['innovative_results']

st.title("📊 Step 5: Performance Comparison")

# Top Comparison KPIs
st.markdown("## 📈 Key Metrics Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### Silhouette Score")
    st.markdown(f"**Baseline:** {baseline['silhouette']:.4f}")
    st.markdown(f"**Innovative:** {innovative['silhouette']:.4f}")
    improvement = ((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette']) * 100
    if improvement > 0:
        st.success(f"↑ +{improvement:.2f}%")
    else:
        st.error(f"↓ {improvement:.2f}%")

with col2:
    st.markdown("### Davies-Bouldin")
    st.markdown(f"**Baseline:** {baseline['davies_bouldin']:.4f}")
    st.markdown(f"**Innovative:** {innovative['davies_bouldin']:.4f}")
    improvement = ((baseline['davies_bouldin'] - innovative['davies_bouldin']) / baseline['davies_bouldin']) * 100
    if improvement > 0:
        st.success(f"↓ Better by {improvement:.2f}%")
    else:
        st.error(f"↑ Worse by {abs(improvement):.2f}%")

with col3:
    st.markdown("### Calinski-Harabasz")
    st.markdown(f"**Baseline:** {baseline['calinski']:.2f}")
    st.markdown(f"**Innovative:** {innovative['calinski']:.2f}")
    improvement = ((innovative['calinski'] - baseline['calinski']) / baseline['calinski']) * 100
    if improvement > 0:
        st.success(f"↑ +{improvement:.2f}%")
    else:
        st.error(f"↓ {improvement:.2f}%")

with col4:
    st.markdown("### Training Time")
    st.markdown(f"**Baseline:** {baseline['training_time']:.2f}s")
    st.markdown(f"**Innovative:** {innovative['training_time']:.2f}s")
    diff = innovative['training_time'] - baseline['training_time']
    if diff < 0:
        st.success(f"⚡ {abs(diff):.2f}s faster")
    else:
        st.info(f"⏱️ {diff:.2f}s slower")

st.markdown("---")

# Detailed Comparison Table
st.markdown("## 📋 Detailed Metrics Comparison")

comparison_df = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Score',
        'Training Time (seconds)',
        'Converged',
        'Iterations'
    ],
    'Baseline GMM': [
        f"{baseline['silhouette']:.4f}",
        f"{baseline['davies_bouldin']:.4f}",
        f"{baseline['calinski']:.2f}",
        f"{baseline['training_time']:.2f}",
        "Yes" if baseline['converged'] else "No",
        baseline['n_iter']
    ],
    'Innovative GMM': [
        f"{innovative['silhouette']:.4f}",
        f"{innovative['davies_bouldin']:.4f}",
        f"{innovative['calinski']:.2f}",
        f"{innovative['training_time']:.2f}",
        "Yes",
        "N/A"
    ],
    'Winner': [
        '🚀 Innovative' if innovative['silhouette'] > baseline['silhouette'] else '📈 Baseline',
        '🚀 Innovative' if innovative['davies_bouldin'] < baseline['davies_bouldin'] else '📈 Baseline',
        '🚀 Innovative' if innovative['calinski'] > baseline['calinski'] else '📈 Baseline',
        '🚀 Innovative' if innovative['training_time'] < baseline['training_time'] else '📈 Baseline',
        '-',
        '-'
    ]
})

st.dataframe(comparison_df, use_container_width=True, height=280)

st.markdown("---")

# Visual Comparison
st.markdown("## 📊 Visual Performance Comparison")

# Metrics bar chart
fig = go.Figure()

metrics = ['Silhouette', 'Davies-Bouldin\n(inverted)', 'Calinski/1000']
baseline_values = [
    baseline['silhouette'],
    1 - baseline['davies_bouldin'],  # Invert so higher is better
    baseline['calinski'] / 1000
]
innovative_values = [
    innovative['silhouette'],
    1 - innovative['davies_bouldin'],
    innovative['calinski'] / 1000
]

fig.add_trace(go.Bar(
    name='Baseline GMM',
    x=metrics,
    y=baseline_values,
    marker_color='#1f77b4',
    text=[f"{v:.3f}" for v in baseline_values],
    textposition='auto'
))

fig.add_trace(go.Bar(
    name='Innovative GMM',
    x=metrics,
    y=innovative_values,
    marker_color='#2ca02c',
    text=[f"{v:.3f}" for v in innovative_values],
    textposition='auto'
))

fig.update_layout(
    title='Performance Metrics Comparison (Higher is Better)',
    barmode='group',
    yaxis_title='Score',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Side-by-side PCA
st.markdown("## 🔍 Side-by-Side Cluster Visualization")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Baseline GMM")
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=baseline['labels'].astype(str),
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                'color': 'Cluster'},
        title=f"Silhouette: {baseline['silhouette']:.4f}",
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🚀 Innovative GMM")
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=innovative['labels'].astype(str),
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                'color': 'Cluster'},
        title=f"Silhouette: {innovative['silhouette']:.4f}",
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Business Impact
st.markdown("## 💼 Business Impact Analysis")

df_baseline = df.copy()
df_baseline['cluster'] = baseline['labels']

df_innovative = df.copy()
df_innovative['cluster'] = innovative['labels']

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Baseline Conversion Rates")
    baseline_conv = df_baseline.groupby('cluster')['y_binary'].mean() * 100
    
    fig = px.bar(
        x=[f"C{i}" for i in baseline_conv.index],
        y=baseline_conv.values,
        text=[f"{v:.2f}%" for v in baseline_conv.values],
        title="Conversion Rate by Cluster",
        labels={'x': 'Cluster', 'y': 'Conversion %'},
        color=baseline_conv.values,
        color_continuous_scale='Blues'
    )
    fig.add_hline(y=df['y_binary'].mean()*100, line_dash="dash",
                  annotation_text="Average")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    variance = baseline_conv.var()
    st.metric("Conversion Variance", f"{variance:.2f}",
             help="Higher variance = better segment separation")

with col2:
    st.markdown("### 🚀 Innovative Conversion Rates")
    innovative_conv = df_innovative.groupby('cluster')['y_binary'].mean() * 100
    
    fig = px.bar(
        x=[f"C{i}" for i in innovative_conv.index],
        y=innovative_conv.values,
        text=[f"{v:.2f}%" for v in innovative_conv.values],
        title="Conversion Rate by Cluster",
        labels={'x': 'Cluster', 'y': 'Conversion %'},
        color=innovative_conv.values,
        color_continuous_scale='Greens'
    )
    fig.add_hline(y=df['y_binary'].mean()*100, line_dash="dash",
                  annotation_text="Average")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    variance = innovative_conv.var()
    st.metric("Conversion Variance", f"{variance:.2f}",
             help="Higher variance = better segment separation")

st.markdown("---")

# Final Recommendation
st.markdown("## 🎯 Final Recommendation")

# Count wins
metrics_won = {
    'Baseline': 0,
    'Innovative': 0
}

if innovative['silhouette'] > baseline['silhouette']:
    metrics_won['Innovative'] += 1
else:
    metrics_won['Baseline'] += 1

if innovative['davies_bouldin'] < baseline['davies_bouldin']:
    metrics_won['Innovative'] += 1
else:
    metrics_won['Baseline'] += 1

if innovative['calinski'] > baseline['calinski']:
    metrics_won['Innovative'] += 1
else:
    metrics_won['Baseline'] += 1

# Display recommendation
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if metrics_won['Innovative'] > metrics_won['Baseline']:
        st.success(f"""
        ### ✅ Recommendation: Use Innovative GMM
        
        **Performance:** Won {metrics_won['Innovative']}/3 key metrics
        
        **Benefits:**
        - Better clustering quality
        - Enhanced interpretability
        - Smart initialization
        
        **Deploy the Innovative approach for production!**
        """)
    elif metrics_won['Innovative'] == metrics_won['Baseline']:
        st.info("""
        ### ⚖️ Recommendation: Comparable Performance
        
        Both approaches show similar performance.
        
        **Consider:**
        - Baseline for simplicity
        - Innovative for better initialization
        """)
    else:
        st.warning(f"""
        ### 📈 Recommendation: Use Baseline GMM
        
        **Performance:** Baseline won {metrics_won['Baseline']}/3 metrics
        
        The standard approach performs better on this dataset.
        """)

st.markdown("---")

# Download
st.markdown("## 💾 Download Results")

comparison_csv = comparison_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Comparison Results",
    data=comparison_csv,
    file_name="gmm_comparison.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("---")

# Navigation
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("➡️ View Technical Report", 
                use_container_width=True, type="primary"):
        st.switch_page("pages/page6_report.py")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Innovative GMM", use_container_width=True):
        st.switch_page("pages/page4_innovative_gmm.py")

with col2:
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("app.py")