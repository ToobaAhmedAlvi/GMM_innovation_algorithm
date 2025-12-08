# pages1/5_🚀_Innovative_Deep_GMM.py
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
from scipy.stats import pearsonr
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Innovative Deep GMM", page_icon="🚀", layout="wide")

st.title("🚀 Step 5: Innovative Deep-Based GMM")
st.markdown("---")

st.info("""
**🎯 Innovation Objective:** Enhance standard GMM clustering through hierarchical 
variable grouping and adaptive clustering strategies inspired by research in 
many-objective optimization.
""")

# Load data
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv",sep=",")
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

# Get baseline results
if 'baseline_results' not in st.session_state:
    st.warning("⚠️ Please run Baseline GMM first!")
    st.stop()

optimal_k = st.session_state.baseline_results['optimal_k']

# Innovation Overview
st.header("5.1 Innovation Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🧬 Deep-Based GMM Architecture
    
    Our innovation extends standard GMM through three key enhancements:
    
    #### **1. Variable Contribution Analysis (GDVG)**
    - Analyze each variable's role in clustering
    - Separate convergence-related from diversity-related variables
    - Use GMM-based framework for classification
    
    #### **2. Chaos-Based Linkage Identification (LIMC)**
    - Detect non-linear variable interactions
    - Apply chaotic perturbations for sensitivity analysis
    - Group interacting variables together
    
    #### **3. Adaptive Multi-Phase Clustering**
    - Phase 1: Focus on convergence variables
    - Phase 2: Refine with diversity variables
    - Phase 3: Integrated final clustering
    
    **Inspired by:** Wang et al. (2025) - "A deep-based Gaussian mixture model 
    algorithm for large-scale many objective optimization"
    """)

with col2:
    st.markdown("""
    ### 📊 Key Differences from Baseline
    
    **Baseline GMM:**
    - Treats all variables equally
    - Single-phase clustering
    - Standard EM algorithm
    
    **Deep-Based GMM:**
    - Variable-aware clustering
    - Multi-phase adaptive approach
    - Enhanced initialization
    
    **Expected Benefits:**
    - Better cluster separation
    - More interpretable results
    - Improved performance metrics
    """)

# Deep GMM Implementation
st.markdown("---")
st.header("5.2 Deep-Based GMM Implementation")

# IMPLEMENTATION CLASS
class DeepGMM:
    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.convergence_vars = []
        self.diversity_vars = []
        self.variable_groups = {}
        self.gmm_model = None
        self.variable_importance = {}
        
    def _analyze_variable_contribution(self, X, feature_names, n_samples=100, n_perturbations=10):
        """Variable contribution analysis using GMM framework"""
        n_features = X.shape[1]
        pcc_scores = np.zeros(n_features)
        
        sample_indices = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
        X_sample = X[sample_indices]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for var_idx in range(n_features):
            status_text.text(f"Analyzing variable {var_idx+1}/{n_features}: {feature_names[var_idx]}")
            pcc_values = []
            
            for _ in range(n_perturbations):
                X_perturbed = X_sample.copy()
                perturbation = np.random.normal(0, 0.1, X_sample.shape[0])
                X_perturbed[:, var_idx] += perturbation
                
                gmm_temp = GaussianMixture(n_components=2, random_state=self.random_state, max_iter=50)
                gmm_temp.fit(X_perturbed)
                
                means = gmm_temp.means_
                direction = means[1] - means[0]
                normal = np.ones(n_features) / np.sqrt(n_features)
                
                if np.std(direction) > 0:
                    pcc, _ = pearsonr(direction, normal)
                    pcc_values.append(abs(pcc))
            
            pcc_scores[var_idx] = np.mean(pcc_values)
            progress_bar.progress((var_idx + 1) / n_features)
        
        status_text.empty()
        progress_bar.empty()
        
        # Normalize
        pcc_scores = (pcc_scores - pcc_scores.min()) / (pcc_scores.max() - pcc_scores.min() + 1e-10)
        
        # Use GMM to cluster variables
        var_gmm = GaussianMixture(n_components=2, random_state=self.random_state)
        var_labels = var_gmm.fit_predict(pcc_scores.reshape(-1, 1))
        
        mean_pcc_0 = pcc_scores[var_labels == 0].mean()
        mean_pcc_1 = pcc_scores[var_labels == 1].mean()
        
        if mean_pcc_0 < mean_pcc_1:
            self.convergence_vars = [i for i in range(n_features) if var_labels[i] == 0]
            self.diversity_vars = [i for i in range(n_features) if var_labels[i] == 1]
        else:
            self.convergence_vars = [i for i in range(n_features) if var_labels[i] == 1]
            self.diversity_vars = [i for i in range(n_features) if var_labels[i] == 0]
        
        self.variable_importance = {
            feature_names[i]: {
                'pcc_score': pcc_scores[i],
                'type': 'convergence' if i in self.convergence_vars else 'diversity'
            }
            for i in range(n_features)
        }
        
        return pcc_scores
    
    def _chaos_based_linkage(self, X, var_indices, mu=3.9):
        """Chaos-based linkage identification"""
        linkage_groups = []
        processed_vars = set()
        
        for var_i in var_indices:
            if var_i in processed_vars:
                continue
            
            current_group = [var_i]
            processed_vars.add(var_i)
            
            for var_j in var_indices:
                if var_j in processed_vars:
                    continue
                
                n_samples = min(1000, X.shape[0])
                sample_idx = np.random.choice(X.shape[0], n_samples, replace=False)
                X_sample = X[sample_idx]
                
                chaos_values = np.random.random(n_samples)
                for _ in range(10):
                    chaos_values = mu * chaos_values * (1 - chaos_values)
                
                X_i_perturbed = X_sample[:, var_i] + 0.1 * chaos_values
                X_j_perturbed = X_sample[:, var_j] + 0.1 * chaos_values
                
                corr_original = np.corrcoef(X_sample[:, var_i], X_sample[:, var_j])[0, 1]
                corr_perturbed = np.corrcoef(X_i_perturbed, X_j_perturbed)[0, 1]
                
                interaction_score = abs(corr_perturbed - corr_original)
                
                if interaction_score > 0.1:
                    current_group.append(var_j)
                    processed_vars.add(var_j)
            
            linkage_groups.append(current_group)
        
        return linkage_groups
    
    def _adaptive_gmm_fitting(self, X, convergence_vars, diversity_vars):
        """Adaptive multi-phase clustering"""
        # Phase 1: Convergence variables
        X_convergence = X[:, convergence_vars]
        gmm_conv = GaussianMixture(n_components=self.n_components,
                                   covariance_type='full',
                                   random_state=self.random_state,
                                   max_iter=300,
                                   n_init=10)
        gmm_conv.fit(X_convergence)
        initial_labels = gmm_conv.predict(X_convergence)
        
        # Phase 2: Refine with diversity
        X_diversity = X[:, diversity_vars]
        refined_labels = initial_labels.copy()
        
        for cluster_id in range(self.n_components):
            cluster_mask = initial_labels == cluster_id
            cluster_diversity_data = X_diversity[cluster_mask]
            
            if len(cluster_diversity_data) > 10:
                within_gmm = GaussianMixture(n_components=2, random_state=self.random_state)
                within_gmm.fit(cluster_diversity_data)
                bic_split = within_gmm.bic(cluster_diversity_data)
                
                single_gmm = GaussianMixture(n_components=1, random_state=self.random_state)
                single_gmm.fit(cluster_diversity_data)
                bic_single = single_gmm.bic(cluster_diversity_data)
                
                if bic_split < bic_single * 0.95:
                    sub_labels = within_gmm.predict(cluster_diversity_data)
                    refined_labels[cluster_mask] = cluster_id + sub_labels * 0.1
        
        # Phase 3: Final clustering
        self.gmm_model = GaussianMixture(n_components=self.n_components,
                                        covariance_type='full',
                                        random_state=self.random_state,
                                        max_iter=300,
                                        n_init=10)
        self.gmm_model.fit(X)
        final_labels = self.gmm_model.predict(X)
        
        return final_labels
    
    def fit_predict(self, X, feature_names):
        """Complete pipeline"""
        st.subheader("🔄 Phase 1: Variable Contribution Analysis")
        with st.spinner("Analyzing variable contributions..."):
            pcc_scores = self._analyze_variable_contribution(X, feature_names)
        
        st.success(f"✅ Identified {len(self.convergence_vars)} convergence and {len(self.diversity_vars)} diversity variables")
        
        st.subheader("🔄 Phase 2: Linkage Identification")
        with st.spinner("Detecting variable interactions..."):
            conv_groups = self._chaos_based_linkage(X, self.convergence_vars)
            div_groups = self._chaos_based_linkage(X, self.diversity_vars)
        
        st.success(f"✅ Found {len(conv_groups)} convergence groups and {len(div_groups)} diversity groups")
        
        self.variable_groups = {
            'convergence_groups': conv_groups,
            'diversity_groups': div_groups,
            'pcc_scores': pcc_scores
        }
        
        st.subheader("🔄 Phase 3: Adaptive Multi-Phase Clustering")
        with st.spinner("Training adaptive GMM..."):
            labels = self._adaptive_gmm_fitting(X, self.convergence_vars, self.diversity_vars)
        
        st.success("✅ Deep-based GMM training complete!")
        
        return labels

# Execute Deep GMM
st.subheader("5.2.1 Training Deep-Based GMM")

if st.button("🚀 Train Innovative Deep GMM", type="primary"):
    start_time = time.time()
    
    deep_gmm = DeepGMM(n_components=optimal_k, random_state=42)
    innovative_labels = deep_gmm.fit_predict(X_scaled, feature_names)
    
    training_time = time.time() - start_time
    
    # Calculate metrics
    innovative_silhouette = silhouette_score(X_scaled, innovative_labels)
    innovative_davies_bouldin = davies_bouldin_score(X_scaled, innovative_labels)
    innovative_calinski_harabasz = calinski_harabasz_score(X_scaled, innovative_labels)
    
    # Store results
    st.session_state.innovative_results = {
        'labels': innovative_labels,
        'silhouette': innovative_silhouette,
        'davies_bouldin': innovative_davies_bouldin,
        'calinski_harabasz': innovative_calinski_harabasz,
        'training_time': training_time,
        'deep_gmm': deep_gmm,
        'converged': deep_gmm.gmm_model.converged_,
        'iterations': deep_gmm.gmm_model.n_iter_
    }
    
    st.success(f"✅ Training completed in {training_time:.2f} seconds!")
    
    # Display results
    st.markdown("---")
    st.header("5.3 Innovative Approach Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Converged", "✅ Yes" if deep_gmm.gmm_model.converged_ else "❌ No")
    with col2:
        st.metric("Iterations", deep_gmm.gmm_model.n_iter_)
    with col3:
        st.metric("Training Time", f"{training_time:.2f}s")
    with col4:
        st.metric("Convergence Vars", len(deep_gmm.convergence_vars))
    
    # Performance Metrics
    st.subheader("5.3.1 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    baseline_sil = st.session_state.baseline_results['silhouette']
    baseline_db = st.session_state.baseline_results['davies_bouldin']
    baseline_ch = st.session_state.baseline_results['calinski_harabasz']
    
    with col1:
        improvement = ((innovative_silhouette - baseline_sil) / baseline_sil) * 100
        st.metric("Silhouette Score", f"{innovative_silhouette:.4f}",
                 delta=f"{improvement:+.2f}%")
    
    with col2:
        improvement = ((baseline_db - innovative_davies_bouldin) / baseline_db) * 100
        st.metric("Davies-Bouldin", f"{innovative_davies_bouldin:.4f}",
                 delta=f"{improvement:+.2f}%")
    
    with col3:
        improvement = ((innovative_calinski_harabasz - baseline_ch) / baseline_ch) * 100
        st.metric("Calinski-Harabasz", f"{innovative_calinski_harabasz:.2f}",
                 delta=f"{improvement:+.2f}%")
    
    # Variable Importance
    st.subheader("5.3.2 Variable Classification")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Convergence Variables", "🌈 Diversity Variables"])
    
    with tab1:
        var_types = [info['type'] for info in deep_gmm.variable_importance.values()]
        type_counts = pd.Series(var_types).value_counts()
        
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                     title='Variable Type Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        conv_vars = {k: v for k, v in deep_gmm.variable_importance.items() 
                     if v['type'] == 'convergence'}
        conv_df = pd.DataFrame.from_dict(conv_vars, orient='index')
        conv_df = conv_df.sort_values('pcc_score')
        st.dataframe(conv_df, use_container_width=True)
    
    with tab3:
        div_vars = {k: v for k, v in deep_gmm.variable_importance.items() 
                    if v['type'] == 'diversity'}
        div_df = pd.DataFrame.from_dict(div_vars, orient='index')
        div_df = div_df.sort_values('pcc_score', ascending=False)
        st.dataframe(div_df, use_container_width=True)
    
    # Visualization
    st.subheader("5.3.3 Cluster Visualization")
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                     color=innovative_labels.astype(str),
                     title='Innovative Deep GMM Clusters (PCA)',
                     labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                            'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)',
                            'color': 'Cluster'},
                     opacity=0.6)
    
    pca_means = pca.transform(deep_gmm.gmm_model.means_)
    fig.add_trace(go.Scatter(x=pca_means[:, 0], y=pca_means[:, 1],
                            mode='markers',
                            marker=dict(size=20, color='red', symbol='x'),
                            name='Centers'))
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Click the button above to train the Innovative Deep-Based GMM")

# Check if results exist
if 'innovative_results' in st.session_state:
    st.success("✅ Innovative Deep GMM results available! Proceed to **Performance Comparison** page.")
else:
    st.warning("⚠️ Train the model first to see results and proceed to comparison.")
