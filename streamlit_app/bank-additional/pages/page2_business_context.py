# pages/2_💼_Business_Context.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Business Context", page_icon="💼", layout="wide")

st.title("💼 Step 2: Business Context & Clustering Objectives")
st.markdown("---")

# Business Domain
st.header("2.1 Business Domain: Banking & Direct Marketing")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🏦 Dataset Background
    
    **Source:** Bank Direct Marketing Campaigns (Portuguese Banking Institution)
    
    **Campaign Type:** Telemarketing campaigns for term deposit subscriptions
    
    **Data Period:** May 2008 - November 2010
    
    **Total Contacts:** 41,188 customer interactions
    
    ### 📞 Campaign Overview
    
    The marketing campaigns were conducted via phone calls to promote term deposit products. 
    Multiple contacts with the same client were often necessary to determine if they would 
    subscribe to the bank's term deposit product.
    
    **Key Characteristics:**
    - Direct telemarketing approach
    - Multiple contact attempts per customer
    - Economic crisis period (2008-2010)
    - Focus on term deposit products
    """)

with col2:
    st.info("""
    ### 🎯 Term Deposit
    
    A **term deposit** is a fixed-term 
    investment account where money 
    is deposited for a predetermined 
    period at a fixed interest rate.
    
    **Benefits:**
    - Guaranteed returns
    - Higher interest than savings
    - Low risk investment
    - Fixed maturity period
    """)

st.markdown("---")

# Why Clustering?
st.header("2.2 Why Clustering This Data?")

st.markdown("""
### 🎯 Primary Objectives

Customer segmentation through clustering enables banks to:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 1️⃣ Targeted Marketing
    - Identify high-value segments
    - Tailor messaging per group
    - Optimize contact strategies
    - Reduce marketing costs
    """)

with col2:
    st.markdown("""
    #### 2️⃣ Resource Optimization
    - Focus on high-potential customers
    - Reduce wasted contact attempts
    - Improve conversion rates
    - Maximize ROI on campaigns
    """)

with col3:
    st.markdown("""
    #### 3️⃣ Strategic Insights
    - Understand customer behavior
    - Identify risk patterns
    - Guide product development
    - Enable data-driven decisions
    """)

st.markdown("---")

# Expected Clusters
st.header("2.3 Expected Clustering Outputs")

st.markdown("""
Based on domain knowledge and exploratory analysis, we expect to identify 
the following customer segments:
""")

# Create visual representation of expected clusters
clusters_data = {
    'Segment': [
        '💎 High-Value Customers',
        '🎓 Young Professionals', 
        '🛡️ Risk-Averse Segment',
        '📊 Economically Sensitive',
        '💤 Low-Engagement Group'
    ],
    'Size': [15, 25, 20, 25, 15],
    'Conversion': [35, 18, 28, 12, 5],
    'Priority': ['Very High', 'High', 'High', 'Medium', 'Low']
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💎 High-Value", 
    "🎓 Young Professional", 
    "🛡️ Risk-Averse",
    "📊 Economic-Sensitive",
    "💤 Low-Engagement"
])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 💎 High-Value Customers (Prime Target)
        
        **Characteristics:**
        - Age: 35-55 years
        - Education: University degree
        - Job: Management, professional roles
        - Financial Status: Stable, high balance
        - Economic Awareness: High
        
        **Behavior:**
        - Positive response to previous campaigns
        - Longer conversation duration
        - Higher deposit amounts
        - Lower contact resistance
        
        **Marketing Strategy:**
        - Premium product offerings
        - Personalized service
        - Priority contact list
        - Relationship-based approach
        """)
    
    with col2:
        st.metric("Expected Size", "12-18%", help="Percentage of customer base")
        st.metric("Expected Conversion", "30-40%", help="Likelihood to subscribe")
        st.metric("Campaign Priority", "🔴 Very High", help="Resource allocation")
        st.metric("Lifetime Value", "High", help="Long-term profitability")

with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Young Professionals (Growth Potential)
        
        **Characteristics:**
        - Age: 25-35 years
        - Education: University or ongoing
        - Job: Professional, technical roles
        - Financial Status: Growing, moderate income
        - Tech-Savvy: High digital adoption
        
        **Behavior:**
        - Career growth phase
        - Building financial portfolio
        - Moderate risk appetite
        - Responsive to education-based marketing
        
        **Marketing Strategy:**
        - Long-term relationship building
        - Financial education programs
        - Digital-first communication
        - Flexible product options
        """)
    
    with col2:
        st.metric("Expected Size", "20-28%", help="Percentage of customer base")
        st.metric("Expected Conversion", "15-25%", help="Likelihood to subscribe")
        st.metric("Campaign Priority", "🟡 High", help="Resource allocation")
        st.metric("Lifetime Value", "Very High", help="Long-term growth potential")

with tab3:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🛡️ Risk-Averse Segment (Conservative Investors)
        
        **Characteristics:**
        - Age: 45-65 years
        - Education: Varied
        - Job: Stable employment
        - Financial Status: Conservative, security-focused
        - Previous Success: Positive campaign history
        
        **Behavior:**
        - Preference for guaranteed returns
        - Lower risk tolerance
        - Longer decision-making process
        - Influenced by economic stability
        
        **Marketing Strategy:**
        - Emphasize security and guarantees
        - Detailed product information
        - Trust-building communication
        - Fixed-return term deposits
        """)
    
    with col2:
        st.metric("Expected Size", "18-25%", help="Percentage of customer base")
        st.metric("Expected Conversion", "25-35%", help="Likelihood to subscribe")
        st.metric("Campaign Priority", "🟡 High", help="Resource allocation")
        st.metric("Lifetime Value", "Moderate-High", help="Stable revenue")

with tab4:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Economically Sensitive (Market-Driven)
        
        **Characteristics:**
        - Age: Varied
        - Economic Awareness: Very High
        - Decision Factors: Interest rates, employment
        - Financial Status: Variable
        - Market Timing: Critical factor
        
        **Behavior:**
        - Heavily influenced by economic indicators
        - Wait for favorable conditions
        - Research-oriented decision making
        - Sensitive to Euribor rates
        
        **Marketing Strategy:**
        - Timing-dependent campaigns
        - Economic incentive-based offers
        - Data-driven communication
        - Market condition highlights
        """)
    
    with col2:
        st.metric("Expected Size", "22-30%", help="Percentage of customer base")
        st.metric("Expected Conversion", "10-18%", help="Highly variable by timing")
        st.metric("Campaign Priority", "🟢 Medium", help="Timing-dependent")
        st.metric("Lifetime Value", "Moderate", help="Opportunity-based")

with tab5:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 💤 Low-Engagement Group (Minimal Interest)
        
        **Characteristics:**
        - Minimal previous contact
        - Low campaign responsiveness
        - Varied demographics
        - Financial disinterest or constraints
        
        **Behavior:**
        - Short conversation duration
        - Multiple failed contact attempts
        - Low conversion history
        - High opt-out rate
        
        **Marketing Strategy:**
        - Minimal resource allocation
        - Reactivation campaigns (periodic)
        - Alternative product offerings
        - Cost-efficient communication channels
        - Consider customer lifecycle stage
        """)
    
    with col2:
        st.metric("Expected Size", "12-20%", help="Percentage of customer base")
        st.metric("Expected Conversion", "2-8%", help="Very low likelihood")
        st.metric("Campaign Priority", "⚪ Low", help="Minimal resources")
        st.metric("Lifetime Value", "Low", help="Limited potential")

st.markdown("---")

# Business Impact
st.header("2.4 Expected Business Impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📈 Quantitative Benefits
    
    **Campaign Efficiency:**
    - **15-25%** increase in conversion rates
    - **20-30%** reduction in marketing costs
    - **30-40%** improvement in contact efficiency
    
    **Financial Impact:**
    - Better resource allocation across segments
    - Higher ROI on marketing investments
    - Increased term deposit subscriptions
    - Reduced operational costs
    
    **Customer Management:**
    - Targeted communication strategies
    - Improved customer satisfaction
    - Better timing of campaign efforts
    - Optimized contact frequency
    """)

with col2:
    st.markdown("""
    ### 🎯 Qualitative Benefits
    
    **Strategic Decision Making:**
    - Data-driven marketing strategies
    - Clear segment identification
    - Predictive campaign planning
    - Risk mitigation strategies
    
    **Customer Understanding:**
    - Deeper behavioral insights
    - Personalization opportunities
    - Lifecycle stage identification
    - Preference mapping
    
    **Competitive Advantage:**
    - Market differentiation
    - Customer-centric approach
    - Innovation in campaign design
    - Adaptive marketing strategies
    """)

# Clustering Success Criteria
st.markdown("---")
st.header("2.5 Clustering Success Criteria")

st.markdown("""
### 📊 How We Measure Success

Our clustering solution will be evaluated based on:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🎯 Technical Metrics
    
    - **Silhouette Score** > 0.3
      - Measures cluster cohesion
      - Higher is better
    
    - **Davies-Bouldin Index** < 1.5
      - Cluster separation quality
      - Lower is better
    
    - **Calinski-Harabasz** > 1000
      - Between/within cluster variance
      - Higher indicates better definition
    """)

with col2:
    st.markdown("""
    #### 💼 Business Metrics
    
    - **Conversion Variation**
      - Significant differences across clusters
      - Clear high/low performing segments
    
    - **Segment Interpretability**
      - Clusters make business sense
      - Actionable characteristics
    
    - **Segment Stability**
      - Reproducible results
      - Robust to data variations
    """)

with col3:
    st.markdown("""
    #### 🚀 Innovation Metrics
    
    - **Performance Improvement**
      - Better than baseline GMM
      - At least 5-10% metric improvement
    
    - **Computational Efficiency**
      - Reasonable execution time
      - Scalable to larger datasets
    
    - **Methodological Rigor**
      - Theoretically sound
      - Well-documented approach
    """)

# Strategic Recommendations Framework
st.markdown("---")
st.header("2.6 From Clusters to Action")

st.markdown("""
### 🔄 Segmentation-to-Strategy Framework

Once clusters are identified, the following action framework will be applied:
""")

action_framework = pd.DataFrame({
    'Stage': ['1. Identify', '2. Profile', '3. Strategize', '4. Execute', '5. Monitor'],
    'Action': [
        'Determine cluster membership for all customers',
        'Analyze cluster characteristics and behavior patterns',
        'Design targeted campaigns for each segment',
        'Implement differentiated marketing strategies',
        'Track performance and refine approach'
    ],
    'Output': [
        'Customer-to-segment mapping',
        'Segment profiles and personas',
        'Campaign blueprints per segment',
        'Active personalized campaigns',
        'Performance dashboards & insights'
    ]
})

st.table(action_framework)

st.info("""
**🎯 Ultimate Goal:** Transform clustering insights into actionable marketing strategies 
that increase term deposit subscriptions while optimizing resource utilization and 
improving customer satisfaction.
""")

st.success("✅ Business context established! Proceed to **Preprocessing** page.")
