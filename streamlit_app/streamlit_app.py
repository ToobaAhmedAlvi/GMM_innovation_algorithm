# streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Bank Marketing Clustering",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .nav-button {
        width: 100%;
        height: 120px;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .business-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .objective-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .current-page {
        background-color: #e6f3ff !important;
        border: 2px solid #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)
# Header
st.markdown('<h1 class="main-title">🏦 Bank Marketing Customer Clustering</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deep-Based Gaussian Mixture Model with Adaptive Variable Grouping</p>', unsafe_allow_html=True)

# === TOP NAVIGATION (Consistent across all pages) ===
st.markdown("## 🚀 Navigate to Project Steps")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='current-page'>", unsafe_allow_html=True)
    st.button("📊 Step 1: Data Exploration",  use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Analyze dataset statistics, distributions, and correlations")

with col2:
    if st.button("🔧 Step 2: Preprocessing & Cluster Selection", use_container_width=True, type="primary"):
        st.switch_page("pages/page2_preprocessing.py")
    st.caption("Prepare data and determine optimal number of clusters")

with col3:
    if st.button("📈 Step 3: Baseline GMM", use_container_width=True, type="primary"):
        st.switch_page("pages/page3_baseline_gmm.py")
    st.caption("Train standard Gaussian Mixture Model")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 Step 4: Innovative Deep GMM", use_container_width=True, type="primary"):
        st.switch_page("pages/page4_innovative_gmm.py")
    st.caption("Apply advanced clustering with variable grouping")

with col2:
    if st.button("📊 Step 5: Performance Comparison", use_container_width=True, type="primary"):
        st.switch_page("pages/page5_comparison.py")
    st.caption("Compare baseline vs innovative approach")

with col3:
    if st.button("📑 Step 6: Technical Report", use_container_width=True, type="primary"):
        st.switch_page("pages/page6_report.py")
    st.caption("View mathematical foundations and complexity")

st.markdown("---")


# Business Context
st.markdown('<div class="business-card">', unsafe_allow_html=True)
st.markdown("""
## 💼 Business Context
**Industry:** Banking & Financial Services  
**Use Case:** Direct Marketing Campaign Optimization  
**Data:** Portuguese Bank Telemarketing Campaigns (2008–2010)

### 🎯 Business Problem
Maximize term deposit subscriptions while minimizing marketing costs through intelligent customer segmentation.

### 💡 Solution Approach
Use advanced clustering techniques to identify distinct customer groups for targeted marketing strategies.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Objectives
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="objective-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 📊 Technical Objectives
    1. **Baseline Clustering** – Standard GMM  
    2. **Innovation** – Deep-Based GMM with variable grouping  
    3. **Comparative Analysis** – Performance benchmarking  
    4. **Business Insights** – Actionable customer segments
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="objective-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 💰 Expected Business Impact
    - **15–25%** ↑ Campaign conversion rate  
    - **20–30%** ↓ Marketing costs  
    - **30–40%** ↑ Contact efficiency  
    - **Clear segmentation** for strategy
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Dataset Overview
st.markdown("### 📈 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", "41,188")
with col2:
    st.metric("Features", "20")
with col3:
    st.metric("Campaign Period", "2008–2010")
with col4:
    st.metric("Success Rate", "11.3%")

st.markdown("---")

# Key Features
st.markdown("### ✨ Key Features")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**📊 Interactive Analysis**\n- Real-time visualizations\n- Customizable parameters")
with col2:
    st.markdown("**🎯 User Control**\n- Select GMM parameters\n- Adjust preprocessing")
with col3:
    st.markdown("**💡 Business Insights**\n- Conversion analysis\n- Segment profiling")

st.info("Start by exploring the data in **Step 1: Data Exploration** above")