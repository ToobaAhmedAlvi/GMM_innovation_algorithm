# pages/3_🔧_Preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧", layout="wide")

st.title("🔧 Step 3: Data Preprocessing Pipeline")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv",sep=",")
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
    df['y_bin'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df = load_data()

# Preprocessing Methodology
st.header("3.1 Preprocessing Methodology")

st.markdown("""
### 🔄 Preprocessing Flowchart

```
📥 Raw Data (41,188 × 21)
    ↓
🏷️ Identify Variable Types
    ├→ Categorical Variables (10)
    └→ Numerical Variables (10)
    ↓
🔢 Encode Categorical Variables
    └→ Label Encoding
    ↓
📊 Feature Scaling
    └→ StandardScaler (Z-score normalization)
    ↓
🔍 Correlation Analysis
    └→ Identify highly correlated features
    ↓
📉 Dimensionality Assessment
    └→ PCA for visualization
    ↓
✅ Clustering-Ready Dataset
```
""")

st.info("""
**🎯 Objective:** Transform raw data into a clean, normalized format suitable 
for Gaussian Mixture Model clustering while preserving meaningful variance.
""")

# Variable Identification
st.markdown("---")
st.header("3.2 Variable Type Identification")

col1, col2 = st.columns(2)

# Identify variables
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                  'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 
                  'euribor3m', 'nr_employed']

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'day_of_week', 'poutcome']

with col1:
    st.markdown("### 📊 Numerical Variables")
    num_df = pd.DataFrame({
        'Variable': numerical_cols,
        'Description': [
            'Age of the customer',
            'Last contact duration (seconds)',
            'Number of contacts this campaign',
            'Days since previous campaign contact',
            'Previous campaign contacts',
            'Employment variation rate',
            'Consumer price index',
            'Consumer confidence index',
            'Euribor 3 month rate',
            'Number of employees'
        ],
        'Type': ['Demographic'] + ['Campaign'] * 4 + ['Economic'] * 5
    })
    st.dataframe(num_df, use_container_width=True, height=400)
    
    st.metric("Total Numerical Features", len(numerical_cols))

with col2:
    st.markdown("### 🏷️ Categorical Variables")
    cat_df = pd.DataFrame({
        'Variable': categorical_cols,
        'Unique Values': [df[col].nunique() for col in categorical_cols],
        'Most Common': [df[col].mode()[0] for col in categorical_cols]
    })
    st.dataframe(cat_df, use_container_width=True, height=400)
    
    st.metric("Total Categorical Features", len(categorical_cols))

# Encoding Process
st.markdown("---")
st.header("3.3 Categorical Variable Encoding")

st.markdown("""
### 🔢 Label Encoding Strategy

**Why Label Encoding?**
- Preserves ordinal relationships where applicable
- Memory efficient for GMM (vs one-hot encoding)
- Maintains dimensionality for better clustering
- Compatible with covariance matrix calculations

**Implementation:**
Each categorical value is mapped to an integer [0, n-1] where n is the number of unique categories.
""")

# Perform encoding
@st.cache_data
def encode_data(df, categorical_cols):
    df_encoded = df.copy()
    label_encoders = {}
    encoding_maps = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        
        # Store mapping
        encoding_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    return df_encoded, label_encoders, encoding_maps

df_encoded, label_encoders, encoding_maps = encode_data(df, categorical_cols)

# Show encoding example
st.subheader("Encoding Examples")

selected_cat = st.selectbox("Select categorical variable to view encoding:", categorical_cols)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**Original Values: `{selected_cat}`**")
    st.write(df[selected_cat].value_counts().head(10))

with col2:
    st.markdown(f"**Encoded Values: `{selected_cat}_encoded`**")
    encoding_df = pd.DataFrame({
        'Original': encoding_maps[selected_cat].keys(),
        'Encoded': encoding_maps[selected_cat].values()
    })
    st.dataframe(encoding_df, use_container_width=True)

# Feature Matrix Construction
st.markdown("---")
st.header("3.4 Feature Matrix Construction")

# Build feature matrix
encoded_features = [col + '_encoded' for col in categorical_cols]
all_features = numerical_cols + encoded_features

X = df_encoded[all_features].values
feature_names = all_features

st.markdown(f"""
### 📐 Feature Matrix Created

**Shape:** {X.shape[0]:,} samples × {X.shape[1]} features

**Composition:**
- Numerical features: {len(numerical_cols)}
- Encoded categorical features: {len(encoded_features)}
- **Total features:** {len(all_features)}
""")

# Show sample
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**Sample Feature Matrix (first 10 rows)**")
    sample_df = pd.DataFrame(X[:10], columns=feature_names)
    st.dataframe(sample_df, use_container_width=True)

with col2:
    st.markdown("**Matrix Statistics**")
    st.metric("Mean", f"{X.mean():.4f}")
    st.metric("Std Dev", f"{X.std():.4f}")
    st.metric("Min", f"{X.min():.4f}")
    st.metric("Max", f"{X.max():.4f}")

# Feature Scaling
st.markdown("---")
st.header("3.5 Feature Scaling (Standardization)")

st.markdown("""
### 📏 StandardScaler Application

**Formula:** Z = (X - μ) / σ

Where:
- X = original value
- μ = mean of feature
- σ = standard deviation of feature
- Z = standardized value

**Why Standardization?**
- GMM is sensitive to feature scales
- Ensures all features contribute equally
- Improves convergence of EM algorithm
- Necessary for distance-based calculations
""")

# Apply scaling
@st.cache_data
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

X_scaled, scaler = scale_features(X)

# Visualize scaling effect
selected_feature_idx = st.selectbox(
    "Select feature to visualize scaling effect:",
    range(len(feature_names)),
    format_func=lambda x: feature_names[x]
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Before Scaling**")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X[:, selected_feature_idx], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(feature_names[selected_feature_idx])
    ax.set_ylabel('Frequency')
    ax.set_title(f'Original Distribution')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.metric("Original Mean", f"{X[:, selected_feature_idx].mean():.4f}")
    st.metric("Original Std", f"{X[:, selected_feature_idx].std():.4f}")

with col2:
    st.markdown("**After Scaling**")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X_scaled[:, selected_feature_idx], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel(feature_names[selected_feature_idx] + ' (scaled)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Standardized Distribution')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.metric("Scaled Mean", f"{X_scaled[:, selected_feature_idx].mean():.6f}")
    st.metric("Scaled Std", f"{X_scaled[:, selected_feature_idx].std():.6f}")

# Scaling verification
st.markdown("---")
st.subheader("3.5.1 Scaling Verification")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Overall Mean (scaled)", f"{X_scaled.mean():.8f}")
with col2:
    st.metric("Overall Std (scaled)", f"{X_scaled.std():.6f}")
with col3:
    st.metric("Range (scaled)", f"[{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

st.success("✅ All features successfully standardized to zero mean and unit variance!")

# Correlation Analysis
st.markdown("---")
st.header("3.6 Feature Correlation Analysis")

st.markdown("""
### 🔍 Identifying Multicollinearity

High correlation between features can:
- Create redundancy in clustering
- Inflate covariance matrices
- Reduce model interpretability
- Cause numerical instability

**Threshold:** We identify pairs with |correlation| > 0.85
""")

# Calculate correlation
correlation_matrix = pd.DataFrame(X_scaled, columns=feature_names).corr()

# Find high correlations
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.85:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': corr_val
            })

col1, col2 = st.columns([2, 1])

with col1:
    # Visualize correlation matrix
    fig = px.imshow(correlation_matrix,
                    labels=dict(color="Correlation"),
                    x=feature_names,
                    y=feature_names,
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1,
                    aspect='auto')
    fig.update_layout(height=600, title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Highly Correlated Pairs** (|r| > 0.85)")
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df['Correlation'] = high_corr_df['Correlation'].apply(lambda x: f"{x:.3f}")
        st.dataframe(high_corr_df, use_container_width=True)
        st.warning(f"⚠️ Found {len(high_corr_pairs)} highly correlated pairs")
    else:
        st.success("✅ No highly correlated features found!")
        st.info("All features maintain sufficient independence for clustering.")

# Dimensionality Assessment
st.markdown("---")
st.header("3.7 Dimensionality Assessment (PCA)")

st.markdown("""
### 📉 Principal Component Analysis

PCA helps us understand:
- Intrinsic dimensionality of data
- Variance captured by top components
- Visualization in reduced space
""")

# Apply PCA
@st.cache_data
def apply_pca(X_scaled):
    pca = PCA(random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

X_pca, pca = apply_pca(X_scaled)

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

col1, col2 = st.columns(2)

with col1:
    # Scree plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(explained_var) + 1)),
        y=explained_var * 100,
        name='Individual',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(explained_var) + 1)),
        y=cumulative_var * 100,
        name='Cumulative',
        mode='lines+markers',
        marker_color='red',
        yaxis='y2'
    ))
    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Principal Component',
        yaxis=dict(title='Individual Variance (%)', side='left'),
        yaxis2=dict(title='Cumulative Variance (%)', side='right', overlaying='y'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Variance Statistics**")
    st.metric("PC1 Variance", f"{explained_var[0]*100:.2f}%")
    st.metric("PC2 Variance", f"{explained_var[1]*100:.2f}%")
    st.metric("PC1+PC2 Total", f"{(explained_var[0]+explained_var[1])*100:.2f}%")
    
    # Find components needed for 90% variance
    n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
    st.metric("Components for 90% Variance", n_components_90)

# 2D Visualization
st.subheader("3.7.1 2D PCA Visualization")

fig = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    labels={'x': f'PC1 ({explained_var[0]*100:.2f}% variance)',
            'y': f'PC2 ({explained_var[1]*100:.2f}% variance)'},
    title='Data Distribution in Principal Component Space',
    opacity=0.6
)
fig.update_traces(marker=dict(size=3))
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Summary
st.markdown("---")
st.header("3.8 Preprocessing Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("""
    ### ✅ Completed Steps
    
    - ✓ Variable identification
    - ✓ Categorical encoding
    - ✓ Feature matrix construction
    - ✓ Standardization
    - ✓ Correlation analysis
    - ✓ Dimensionality assessment
    """)

with summary_col2:
    st.markdown("""
    ### 📊 Final Dataset
    
    - Samples: 41,188
    - Features: 20
    - Mean: ~0.0
    - Std: ~1.0
    - No missing values
    - Ready for clustering
    """)

with summary_col3:
    st.markdown("""
    ### 🎯 Next Steps
    
    1. Baseline GMM clustering
    2. Optimal cluster selection
    3. Performance evaluation
    4. Innovative approach
    5. Comparative analysis
    """)

# Cache processed data
st.cache_data
def get_processed_data():
    return X_scaled, feature_names, df_encoded

st.success("✅ Preprocessing complete! Data is ready for clustering. Proceed to **Baseline GMM** page.")

# Download option
st.markdown("---")
st.markdown("### 💾 Download Processed Data")

processed_df = pd.DataFrame(X_scaled, columns=feature_names)
csv = processed_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Scaled Feature Matrix (CSV)",
    data=csv,
    file_name="preprocessed_features.csv",
    mime="text/csv",
)
