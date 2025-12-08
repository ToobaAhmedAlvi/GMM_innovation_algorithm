# pages/6_📊_Performance_Comparison.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Performance Comparison", page_icon="📊", layout="wide")

st.title("📊 Step 6: Performance Comparison")
st.markdown("---")

# Check if both results exist
if 'baseline_results' not in st.session_state:
    st.error("❌ Baseline GMM results not found. Please run Baseline GMM first!")
    st.stop()

if 'innovative_results' not in st.session_state:
    st.error("❌ Innovative Deep GMM results not found. Please run Innovative Deep GMM first!")
    st.stop()

# Load data for visualization
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("bank-additional-full.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
    df['y_bin'] = df['y'].map({'yes': 1, 'no': 0})
    
    numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                      'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 
                      'euribor3m', 'nr_employed']
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                       'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
    
    encoded_features = [col + '_encoded' for col in categorical_cols]
    all_features = numerical_cols + encoded_features
    X = df_encoded[all_features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, all_features, df_encoded, df

X_scaled, feature_names, df_encoded, df = load_and_preprocess()

# Extract results
baseline = st.session_state.baseline_results
innovative = st.session_state.innovative_results

# Overview
st.header("6.1 Overall Comparison Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Baseline GMM")
    st.metric("Silhouette Score", f"{baseline['silhouette']:.4f}")
    st.metric("Davies-Bouldin Index", f"{baseline['davies_bouldin']:.4f}")
    st.metric("Calinski-Harabasz Score", f"{baseline['calinski_harabasz']:.2f}")
    st.metric("Training Time", f"{baseline['training_time']:.2f}s")
    st.metric("Iterations", baseline['iterations'])

with col2:
    st.markdown("### 🚀 Innovative Deep GMM")
    sil_imp = ((innovative['silhouette'] - baseline['silhouette']) / baseline['silhouette']) * 100
    db_imp = ((baseline['davies_bouldin'] - innovative['davies_bouldin']) / baseline['davies_bouldin']) * 100
    ch_imp = ((innovative['calinski_harabasz'] - baseline['calinski_harabasz']) / baseline['calinski_harabasz']) * 100
    
    st.metric("Silhouette Score", f"{innovative['silhouette']:.4f}", 
             delta=f"{sil_imp:+.2f}%")
    st.metric("Davies-Bouldin Index", f"{innovative['davies_bouldin']:.4f}",
             delta=f"{db_imp:+.2f}%")
    st.metric("Calinski-Harabasz Score", f"{innovative['calinski_harabasz']:.2f}",
             delta=f"{ch_imp:+.2f}%")
    st.metric("Training Time", f"{innovative['training_time']:.2f}s")
    st.metric("Iterations", innovative['iterations'])

# Detailed Metrics Comparison
st.markdown("---")
st.header("6.2 Detailed Metrics Comparison")

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Metric': [
        'Silhouette Score',
        'Davies-Bouldin Index',
        'Calinski-Harabasz Score',
        'Training Time (seconds)',
        'Iterations to Converge'
    ],
    'Baseline GMM': [
        f"{baseline['silhouette']:.4f}",
        f"{baseline['davies_bouldin']:.4f}",
        f"{baseline['calinski_harabasz']:.2f}",
        f"{baseline['training_time']:.2f}",
        baseline['iterations']
    ],
    'Innovative Deep GMM': [
        f"{innovative['silhouette']:.4f}",
        f"{innovative['davies_bouldin']:.4f}",
        f"{innovative['calinski_harabasz']:.2f}",
        f"{innovative['training_time']:.2f}",
        innovative['iterations']
    ],
    'Improvement': [
        f"{sil_imp:+.2f}%",
        f"{db_imp:+.2f}%",
        f"{ch_imp:+.2f}%",
        f"{((baseline['training_time'] - innovative['training_time']) / baseline['training_time']) * 100:+.2f}%",
        f"{baseline['iterations'] - innovative['iterations']:+d}"
    ],
    'Winner': [
        '🚀 Innovative' if innovative['silhouette'] > baseline['silhouette'] else '🎯 Baseline',
        '🚀 Innovative' if innovative['davies_bouldin'] < baseline['davies_bouldin'] else '🎯 Baseline',
        '🚀 Innovative' if innovative['calinski_harabasz'] > baseline['calinski_harabasz'] else '🎯 Baseline',
        '🚀 Innovative' if innovative['training_time'] < baseline['training_time'] else '🎯 Baseline',
        '🚀 Innovative' if innovative['iterations'] < baseline['iterations'] else '🎯 Baseline'
    ]
})

st.dataframe(comparison_df, use_container_width=True, height=250)

# Visual Comparison
st.subheader("6.2.1 Visual Metrics Comparison")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Silhouette Score (Higher is Better)',
                   'Davies-Bouldin Index (Lower is Better)',
                   'Calinski-Harabasz Score (Higher is Better)',
                   'Training Time (seconds)'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# Silhouette
fig.add_trace(
    go.Bar(x=['Baseline', 'Innovative'], 
           y=[baseline['silhouette'], innovative['silhouette']],
           marker_color=['#1f77b4', '#2ca02c'],
           text=[f"{baseline['silhouette']:.4f}", f"{innovative['silhouette']:.4f}"],
           textposition='auto'),
    row=1, col=1
)

# Davies-Bouldin
fig.add_trace(
    go.Bar(x=['Baseline', 'Innovative'],
           y=[baseline['davies_bouldin'], innovative['davies_bouldin']],
           marker_color=['#1f77b4', '#2ca02c'],
           text=[f"{baseline['davies_bouldin']:.4f}", f"{innovative['davies_bouldin']:.4f}"],
           textposition='auto'),
    row=1, col=2
)

# Calinski-Harabasz
fig.add_trace(
    go.Bar(x=['Baseline', 'Innovative'],
           y=[baseline['calinski_harabasz'], innovative['calinski_harabasz']],
           marker_color=['#1f77b4', '#2ca02c'],
           text=[f"{baseline['calinski_harabasz']:.2f}", f"{innovative['calinski_harabasz']:.2f}"],
           textposition='auto'),
    row=2, col=1
)

# Training Time
fig.add_trace(
    go.Bar(x=['Baseline', 'Innovative'],
           y=[baseline['training_time'], innovative['training_time']],
           marker_color=['#1f77b4', '#2ca02c'],
           text=[f"{baseline['training_time']:.2f}s", f"{innovative['training_time']:.2f}s"],
           textposition='auto'),
    row=2, col=2
)

fig.update_layout(height=700, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# Side-by-Side Visualization
st.markdown("---")
st.header("6.3 Side-by-Side Cluster Visualization")

# PCA projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Baseline GMM")
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=baseline['labels'].astype(str),
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)',
                'color': 'Cluster'},
        title=f"Silhouette: {baseline['silhouette']:.4f}",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🚀 Innovative Deep GMM")
    fig = px.scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        color=innovative['labels'].astype(str),
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)',
                'color': 'Cluster'},
        title=f"Silhouette: {innovative['silhouette']:.4f}",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Cluster Distribution Comparison
st.markdown("---")
st.header("6.4 Cluster Size Distribution")

baseline_counts = pd.Series(baseline['labels']).value_counts().sort_index()
innovative_counts = pd.Series(innovative['labels']).value_counts().sort_index()

fig = go.Figure(data=[
    go.Bar(name='Baseline GMM', x=baseline_counts.index, y=baseline_counts.values,
           marker_color='#1f77b4', text=baseline_counts.values, textposition='auto'),
    go.Bar(name='Innovative Deep GMM', x=innovative_counts.index, y=innovative_counts.values,
           marker_color='#2ca02c', text=innovative_counts.values, textposition='auto')
])

fig.update_layout(
    title='Cluster Size Comparison',
    xaxis_title='Cluster ID',
    yaxis_title='Number of Samples',
    barmode='group',
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Business Impact Comparison
st.markdown("---")
st.header("6.5 Business Impact Analysis")

# Add cluster labels to dataframe
df_baseline = df.copy()
df_baseline['cluster'] = baseline['labels']

df_innovative = df.copy()
df_innovative['cluster'] = innovative['labels']

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Baseline GMM - Conversion Rates")
    baseline_conversion = df_baseline.groupby('cluster')['y_bin'].agg(['mean', 'count'])
    baseline_conversion['conversion_rate'] = baseline_conversion['mean'] * 100
    baseline_conversion.columns = ['Mean', 'Count', 'Conversion Rate (%)']
    st.dataframe(baseline_conversion[['Count', 'Conversion Rate (%)']], use_container_width=True)
    
    # Visualize
    fig = px.bar(x=baseline_conversion.index, 
                 y=baseline_conversion['Conversion Rate (%)'],
                 title='Baseline Conversion Rates by Cluster',
                 labels={'x': 'Cluster', 'y': 'Conversion Rate (%)'})
    fig.update_traces(marker_color='#1f77b4')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🚀 Innovative Deep GMM - Conversion Rates")
    innovative_conversion = df_innovative.groupby('cluster')['y_bin'].agg(['mean', 'count'])
    innovative_conversion['conversion_rate'] = innovative_conversion['mean'] * 100
    innovative_conversion.columns = ['Mean', 'Count', 'Conversion Rate (%)']
    st.dataframe(innovative_conversion[['Count', 'Conversion Rate (%)']], use_container_width=True)
    
    # Visualize
    fig = px.bar(x=innovative_conversion.index,
                 y=innovative_conversion['Conversion Rate (%)'],
                 title='Innovative Conversion Rates by Cluster',
                 labels={'x': 'Cluster', 'y': 'Conversion Rate (%)'})
    fig.update_traces(marker_color='#2ca02c')
    st.plotly_chart(fig, use_container_width=True)

# Conversion rate variance
baseline_var = baseline_conversion['Conversion Rate (%)'].var()
innovative_var = innovative_conversion['Conversion Rate (%)'].var()

st.subheader("6.5.1 Cluster Quality Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Baseline Conversion Variance", 
             f"{baseline_var:.2f}",
             help="Higher variance = better separation of high/low value segments")

with col2:
    st.metric("Innovative Conversion Variance",
             f"{innovative_var:.2f}",
             help="Higher variance = better separation of high/low value segments")

with col3:
    improvement = ((innovative_var - baseline_var) / baseline_var) * 100
    st.metric("Variance Improvement",
             f"{improvement:+.2f}%",
             help="Positive = better business segmentation")

# Statistical Significance
st.markdown("---")
st.header("6.6 Key Findings & Interpretation")

findings = []

if innovative['silhouette'] > baseline['silhouette']:
    findings.append("✅ **Silhouette Score:** Innovative approach shows better cluster cohesion and separation")
else:
    findings.append("⚠️ **Silhouette Score:** Baseline performs better in overall cluster quality")

if innovative['davies_bouldin'] < baseline['davies_bouldin']:
    findings.append("✅ **Davies-Bouldin Index:** Innovative approach achieves better cluster separation")
else:
    findings.append("⚠️ **Davies-Bouldin Index:** Baseline has better separation")

if innovative['calinski_harabasz'] > baseline['calinski_harabasz']:
    findings.append("✅ **Calinski-Harabasz Score:** Innovative approach has stronger cluster definition")
else:
    findings.append("⚠️ **Calinski-Harabasz Score:** Baseline has stronger definition")

if innovative_var > baseline_var:
    findings.append("✅ **Business Impact:** Innovative approach better separates high/low value customer segments")
else:
    findings.append("⚠️ **Business Impact:** Baseline provides similar business segmentation")

for finding in findings:
    st.markdown(finding)

# Overall Assessment
st.markdown("---")
st.header("6.7 Overall Assessment")

# Count wins
metrics_comparison = [
    innovative['silhouette'] > baseline['silhouette'],
    innovative['davies_bouldin'] < baseline['davies_bouldin'],
    innovative['calinski_harabasz'] > baseline['calinski_harabasz']
]

innovative_wins = sum(metrics_comparison)
baseline_wins = len(metrics_comparison) - innovative_wins

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    fig = go.Figure(data=[
        go.Bar(name='Metrics Won', 
               x=['Baseline GMM', 'Innovative Deep GMM'],
               y=[baseline_wins, innovative_wins],
               marker_color=['#1f77b4', '#2ca02c'],
               text=[baseline_wins, innovative_wins],
               textposition='auto')
    ])
    fig.update_layout(title='Number of Metrics Where Each Approach Performed Better',
                     yaxis_title='Count',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

# Recommendation
if innovative_wins > baseline_wins:
    st.success(f"""
    ### 🎯 **Recommendation: Innovative Deep GMM**
    
    The innovative approach wins on **{innovative_wins}/3** key metrics, demonstrating:
    - Superior clustering quality
    - Better variable utilization
    - More actionable business segmentation
    
    **Deployment Recommendation:** ✅ Use Innovative Deep GMM for production
    """)
elif innovative_wins == baseline_wins:
    st.info("""
    ### ⚖️ **Recommendation: Situation Dependent**
    
    Both approaches show comparable performance. Consider:
    - Baseline GMM for simplicity and faster execution
    - Innovative Deep GMM for better interpretability and variable insights
    """)
else:
    st.warning("""
    ### 🎯 **Recommendation: Baseline GMM**
    
    The baseline approach performs better on majority metrics. However, the innovative
    approach provides valuable variable insights that can inform feature engineering.
    """)

# Download Results
st.markdown("---")
st.subheader("💾 Download Comparison Results")

col1, col2, col3 = st.columns(3)

with col1:
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Metrics Comparison",
        data=csv,
        file_name="metrics_comparison.csv",
        mime="text/csv",
    )

with col2:
    results_df = pd.DataFrame({
        'sample_id': range(len(baseline['labels'])),
        'baseline_cluster': baseline['labels'],
        'innovative_cluster': innovative['labels']
    })
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cluster Labels",
        data=csv,
        file_name="cluster_labels_comparison.csv",
        mime="text/csv",
    )

with col3:
    conversion_df = pd.DataFrame({
        'cluster': baseline_conversion.index,
        'baseline_conversion': baseline_conversion['Conversion Rate (%)'],
        'innovative_conversion': innovative_conversion['Conversion Rate (%)']
    })
    csv = conversion_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Conversion Analysis",
        data=csv,
        file_name="conversion_comparison.csv",
        mime="text/csv",
    )

st.success("✅ Performance comparison complete! Proceed to **Technical Report** for mathematical details.")
