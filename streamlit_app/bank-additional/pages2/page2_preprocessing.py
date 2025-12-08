# pages/page2_preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Preprocessing", page_icon="🔧", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv",sep=",")
    df.columns = df.columns.str.strip()
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df = load_data()

st.title("🔧 Step 2: Data Preprocessing & Optimal Cluster Selection")

# KPIs
st.markdown("## 📊 Preprocessing Status")

col1, col2, col3, col4 = st.columns(4)

numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous',
                  'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                  'euribor3m', 'nr.employed']

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'day_of_week', 'poutcome']

with col1:
    st.metric("Numerical Features", len(numerical_cols))
with col2:
    st.metric("Categorical Features", len(categorical_cols))
with col3:
    st.metric("Total Features", len(numerical_cols) + len(categorical_cols))
with col4:
    st.metric("Missing Values", df.isnull().sum().sum())

st.markdown("---")

# Preprocessing Steps
st.markdown("## 🔄 Preprocessing Pipeline")

with st.expander("ℹ️ View Preprocessing Steps", expanded=False):
    st.markdown("""
    **Step 1:** Identify variable types (numerical vs categorical)  
    **Step 2:** Encode categorical variables using Label Encoding  
    **Step 3:** Standardize all features (mean=0, std=1)  
    **Step 4:** Validate preprocessed data  
    """)

# Preprocessing execution
@st.cache_data
def preprocess_data(df, numerical_cols, categorical_cols):
    df_processed = df.copy()
    
    # Encode categorical
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Build feature matrix
    encoded_features = [col + '_encoded' for col in categorical_cols]
    all_features = numerical_cols + encoded_features
    
    X = df_processed[all_features].values
    
    # Check for NaN after encoding
    if np.isnan(X).any():
        st.error("⚠️ NaN values detected after encoding!")
        # Impute NaN values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        st.success("✅ NaN values imputed with mean")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Final NaN check
    if np.isnan(X_scaled).any():
        st.error("⚠️ NaN values still present after scaling!")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        st.warning("⚠️ Replaced remaining NaN with 0")
    
    return X_scaled, all_features, scaler, label_encoders

with st.spinner("Processing data..."):
    X_scaled, feature_names, scaler, label_encoders = preprocess_data(
        df, numerical_cols, categorical_cols
    )

# Preprocessing Results
st.markdown("## ✅ Preprocessing Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Processed Samples", f"{X_scaled.shape[0]:,}")
with col2:
    st.metric("Total Features", X_scaled.shape[1])
with col3:
    st.metric("Mean (scaled)", f"{X_scaled.mean():.6f}")
with col4:
    st.metric("Std (scaled)", f"{X_scaled.std():.6f}")

st.success("✅ Data preprocessing completed successfully!")

# Save to session state
st.session_state['X_scaled'] = X_scaled
st.session_state['feature_names'] = feature_names
st.session_state['df_processed'] = df
st.session_state['scaler'] = scaler
st.session_state['label_encoders'] = label_encoders

st.markdown("---")

# Optimal Cluster Selection
st.markdown("## 🎯 Optimal Number of Clusters (K) Selection")

st.info("""
📌 **Why is this important?**  
Selecting the right number of clusters (K) is crucial for meaningful segmentation.  
We use multiple criteria (BIC, AIC, Silhouette) to find the optimal K.
""")

# User input for K range
col1, col2 = st.columns(2)

with col1:
    k_min = st.number_input("Minimum K", min_value=2, max_value=10, value=2)
with col2:
    k_max = st.number_input("Maximum K", min_value=k_min+1, max_value=15, value=8)

if st.button("🔍 Analyze Optimal K", type="primary", use_container_width=True):
    
    with st.spinner("Computing model selection criteria..."):
        
        @st.cache_data
        def compute_optimal_k(_X_scaled, k_range):
            bic_scores = []
            aic_scores = []
            silhouette_scores = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, k in enumerate(k_range):
                status_text.text(f"Evaluating K = {k}...")
                
                gmm = GaussianMixture(n_components=k, 
                                     random_state=42,
                                     covariance_type='full',
                                     max_iter=200,
                                     n_init=5)
                gmm.fit(_X_scaled)
                
                bic_scores.append(gmm.bic(_X_scaled))
                aic_scores.append(gmm.aic(_X_scaled))
                
                labels = gmm.predict(_X_scaled)
                silhouette_scores.append(silhouette_score(_X_scaled, labels))
                
                progress_bar.progress((idx + 1) / len(k_range))
            
            status_text.empty()
            progress_bar.empty()
            
            return bic_scores, aic_scores, silhouette_scores
        
        k_range = list(range(k_min, k_max + 1))
        bic_scores, aic_scores, sil_scores = compute_optimal_k(X_scaled, k_range)
        
        # Store in session
        st.session_state['k_range'] = k_range
        st.session_state['bic_scores'] = bic_scores
        st.session_state['aic_scores'] = aic_scores
        st.session_state['sil_scores'] = sil_scores
        
        # Results
        st.markdown("### 📊 Model Selection Criteria")
        
        # Plot
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('BIC Score (Lower is Better)', 
                          'AIC Score (Lower is Better)',
                          'Silhouette Score (Higher is Better)')
        )
        
        fig.add_trace(
            go.Scatter(x=k_range, y=bic_scores, mode='lines+markers',
                      name='BIC', line=dict(color='blue', width=2),
                      marker=dict(size=10)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=k_range, y=aic_scores, mode='lines+markers',
                      name='AIC', line=dict(color='red', width=2),
                      marker=dict(size=10)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=k_range, y=sil_scores, mode='lines+markers',
                      name='Silhouette', line=dict(color='green', width=2),
                      marker=dict(size=10)),
            row=1, col=3
        )
        
        fig.update_xaxes(title_text="Number of Clusters (K)")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal K recommendations
        optimal_k_bic = k_range[np.argmin(bic_scores)]
        optimal_k_aic = k_range[np.argmin(aic_scores)]
        optimal_k_sil = k_range[np.argmax(sil_scores)]
        
        st.markdown("### 🎯 Recommended K Values")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BIC Recommends", optimal_k_bic,
                     help=f"BIC Score: {min(bic_scores):.2f}")
        with col2:
            st.metric("AIC Recommends", optimal_k_aic,
                     help=f"AIC Score: {min(aic_scores):.2f}")
        with col3:
            st.metric("Silhouette Recommends", optimal_k_sil,
                     help=f"Silhouette Score: {max(sil_scores):.4f}")
        
        # Final recommendation
        from collections import Counter
        votes = [optimal_k_bic, optimal_k_aic, optimal_k_sil]
        vote_counts = Counter(votes)
        recommended_k = vote_counts.most_common(1)[0][0]
        
        st.session_state['recommended_k'] = recommended_k
        
        st.success(f"""
        ✅ **Final Recommendation: K = {recommended_k}**  
        This value is selected based on majority voting from all three criteria.
        """)
        
        # Detailed table
        results_df = pd.DataFrame({
            'K': k_range,
            'BIC': [f"{score:.2f}" for score in bic_scores],
            'AIC': [f"{score:.2f}" for score in aic_scores],
            'Silhouette': [f"{score:.4f}" for score in sil_scores]
        })
        
        st.markdown("### 📋 Detailed Results Table")
        st.dataframe(results_df, use_container_width=True, height=300)

# Show stored recommendation
if 'recommended_k' in st.session_state:
    st.info(f"💡 **Recommended K for next steps: {st.session_state['recommended_k']}**")

st.markdown("---")

# PCA Visualization
st.markdown("## 🔍 Data Visualization (PCA Projection)")

with st.expander("ℹ️ About PCA Visualization"):
    st.markdown("""
    Principal Component Analysis (PCA) reduces high-dimensional data to 2D for visualization.  
    This helps us understand the data structure before clustering.
    """)

# Apply PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig = px.scatter(
    x=X_pca[:, 0], y=X_pca[:, 1],
    labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
            'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'},
    title='Data Distribution in 2D Space',
    opacity=0.5,
    color_discrete_sequence=['#1f77b4']
)
fig.update_traces(marker=dict(size=3))
fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
**Explained Variance:**
- PC1: {pca.explained_variance_ratio_[0]*100:.2f}%
- PC2: {pca.explained_variance_ratio_[1]*100:.2f}%
- **Total: {sum(pca.explained_variance_ratio_)*100:.2f}%**
""")

st.markdown("---")

# Navigation
st.markdown("## 🚀 Next Steps")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if 'recommended_k' in st.session_state:
        if st.button("➡️ Proceed to Baseline GMM", 
                    use_container_width=True, type="primary"):
            st.switch_page("pages/page3_baseline_gmm.py")
    else:
        st.warning("⚠️ Please analyze optimal K first")

col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Back to Data Exploration", use_container_width=True):
        st.switch_page("pages/page1_data_exploration.py")

with col2:
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("app.py")