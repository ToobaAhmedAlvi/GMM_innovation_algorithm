import streamlit as st
import pandas as pd

# Custom CSS for professional look
def apply_custom_css():
    st.markdown("""
    <style>
        /* Main container */
        .main {
            padding-top: 1rem;
        }
        
        /* Navigation bar */
        .nav-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        
        .nav-title {
            color: white;
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .nav-subtitle {
            color: #e0e0e0;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Step buttons */
        .step-button {
            background: white;
            color: #667eea;
            border: 2px solid white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            margin: 0.2rem;
            font-weight: bold;
        }
        
        .step-button-active {
            background: #ffd700;
            color: #333;
            border: 2px solid #ffd700;
        }
        
        /* KPI Cards */
        .kpi-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Filter section */
        .filter-section {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Global navigation header
def render_navigation(current_page):
    """Render consistent navigation across all pages"""
    
    st.markdown("""
    <div class="nav-container">
        <div class="nav-title">🏦 Bank Marketing Customer Clustering</div>
        <div class="nav-subtitle">Deep-Based GMM with Adaptive Variable Grouping</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("### 📍 Project Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_page == "data_exploration":
            st.button("📊 Step 1: Data Exploration", disabled=True, use_container_width=True, type="primary")
        else:
            if st.button("📊 Step 1: Data Exploration", use_container_width=True):
                st.switch_page("pages/page1_data_exploration.py")
    
    with col2:
        if current_page == "preprocessing":
            st.button("🔧 Step 2: Preprocessing", disabled=True, use_container_width=True, type="primary")
        else:
            if st.button("🔧 Step 2: Preprocessing", use_container_width=True):
                st.switch_page("pages/page2_preprocessing.py")
    
    with col3:
        if current_page == "baseline_gmm":
            st.button("📈 Step 3: Baseline GMM", disabled=True, use_container_width=True, type="primary")
        else:
            if st.button("📈 Step 3: Baseline GMM", use_container_width=True):
                st.switch_page("pages/page3_baseline_gmm.py")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_page == "innovative_gmm":
            st.button("🚀 Step 4: Innovative GMM", disabled=True, use_container_width=True, type="primary")
        else:
            if st.button("🚀 Step 4: Innovative GMM", use_container_width=True):
                st.switch_page("pages/page4_innovative_gmm.py")
    
    with col2:
        if current_page == "comparison":
            st.button("📊 Step 5: Comparison", disabled=True, use_container_width=True, type="primary")
        else:
            if st.button("📊 Step 5: Comparison", use_container_width=True):
                st.switch_page("pages/page5_comparison.py")
    
    with col3:
        if current_page == "report":
            st.button("📑 Step 6: Report", disabled=True, use_container_width=True, type="primary")
        else:
            if st.button("📑 Step 6: Report", use_container_width=True):
                st.switch_page("pages/page6_report.py")
    
    st.markdown("---")

# Global sidebar filters
def render_sidebar_filters(df):
    """Render global filters in sidebar"""
    
    st.sidebar.markdown("# 🎯 Global Filters")
    st.sidebar.markdown("Apply filters to all visualizations across pages")
    
    # Initialize session state for filters if not exists (using lowercase 'all' for internal logic)
    if 'filter_month' not in st.session_state:
        st.session_state.filter_month = 'all' 
    if 'filter_day' not in st.session_state:
        st.session_state.filter_day = 'all' 
    if 'filter_age_range' not in st.session_state:
        st.session_state.filter_age_range = (int(df['age'].min()), int(df['age'].max()))
    if 'filter_job' not in st.session_state:
        st.session_state.filter_job = 'all' 
    if 'filter_education' not in st.session_state:
        st.session_state.filter_education = 'all' 
    if 'filter_marital' not in st.session_state:
        st.session_state.filter_marital = 'all' 
    
    st.sidebar.markdown("---")
    
    # --- Helper to handle display vs internal value ---
    def get_display_options(col_name):
        # Get unique values, capitalize for display, and add 'All'
        unique_values = sorted(df[col_name].unique().tolist())
        display_options = ['All'] + [v.replace('.', '').replace('_', ' ').title() for v in unique_values]
        return display_options, unique_values

    def get_display_value(internal_value):
        if internal_value == 'all':
            return 'All'
        # Handle job type special case (e.g., 'admin.' -> 'Admin')
        return internal_value.replace('.', '').replace('_', ' ').title()

    def get_internal_value(display_value):
        if display_value == 'All':
            return 'all'
        # Convert display value back to the internal lowercase data format
        # FIX: Corrected syntax error by combining the chained methods
        return display_value.lower().replace(' ', '')
    # -------------------------------------------------

    # Time Filters
    st.sidebar.markdown("### 📅 Time Dimensions")
    
    # Month Filter
    all_months_display, _ = get_display_options('month')
    current_month_display = get_display_value(st.session_state.filter_month)
    selected_month_display = st.sidebar.selectbox(
        "Campaign Month",
        options=all_months_display,
        index=all_months_display.index(current_month_display),
        key="month_filter_display"
    )
    st.session_state.filter_month = selected_month_display.lower()
    
    # Day Filter
    all_days_display, _ = get_display_options('day_of_week')
    current_day_display = get_display_value(st.session_state.filter_day)
    selected_day_display = st.sidebar.selectbox(
        "Day of Week",
        options=all_days_display,
        index=all_days_display.index(current_day_display),
        key="day_filter_display"
    )
    st.session_state.filter_day = selected_day_display.lower()
    
    st.sidebar.markdown("---")
    
    # Demographic Filters
    st.sidebar.markdown("### 👥 Demographics")
    
    # Age Slider (Numerical is fine)
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=st.session_state.filter_age_range,
        key="age_filter"
    )
    st.session_state.filter_age_range = age_range
    
    # Job Filter
    all_jobs_display, _ = get_display_options('job')
    current_job_display = get_display_value(st.session_state.filter_job)
    selected_job_display = st.sidebar.selectbox(
        "Job Type",
        options=all_jobs_display,
        index=all_jobs_display.index(current_job_display),
        key="job_filter_display"
    )
    st.session_state.filter_job = selected_job_display.lower().replace(' ', '') # Match internal cleaning (lower and remove spaces)
    
    # Education Filter
    all_education_display, _ = get_display_options('education')
    current_education_display = get_display_value(st.session_state.filter_education)
    selected_education_display = st.sidebar.selectbox(
        "Education Level",
        options=all_education_display,
        index=all_education_display.index(current_education_display),
        key="education_filter_display"
    )
    st.session_state.filter_education = selected_education_display.lower().replace(' ', '')
    
    # Marital Filter
    all_marital_display, _ = get_display_options('marital')
    current_marital_display = get_display_value(st.session_state.filter_marital)
    selected_marital_display = st.sidebar.selectbox(
        "Marital Status",
        options=all_marital_display,
        index=all_marital_display.index(current_marital_display),
        key="marital_filter_display"
    )
    st.session_state.filter_marital = selected_marital_display.lower().replace(' ', '')
    
    st.sidebar.markdown("---")
    
    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.filter_month = 'all'
        st.session_state.filter_day = 'all'
        st.session_state.filter_age_range = (int(df['age'].min()), int(df['age'].max()))
        st.session_state.filter_job = 'all'
        st.session_state.filter_education = 'all'
        st.session_state.filter_marital = 'all'
        st.rerun()
    
    # Show filter status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filter Status")
    
    filters_applied = []
    if st.session_state.filter_month != 'all':
        filters_applied.append(f"Month: {get_display_value(st.session_state.filter_month)}")
    if st.session_state.filter_day != 'all':
        filters_applied.append(f"Day: {get_display_value(st.session_state.filter_day)}")
    if st.session_state.filter_age_range != (int(df['age'].min()), int(df['age'].max())):
        filters_applied.append(f"Age: {st.session_state.filter_age_range[0]}-{st.session_state.filter_age_range[1]}")
    if st.session_state.filter_job != 'all':
        filters_applied.append(f"Job: {get_display_value(st.session_state.filter_job)}")
    if st.session_state.filter_education != 'all':
        filters_applied.append(f"Education: {get_display_value(st.session_state.filter_education)}")
    if st.session_state.filter_marital != 'all':
        filters_applied.append(f"Marital: {get_display_value(st.session_state.filter_marital)}")
    
    if filters_applied:
        st.sidebar.info("**Active Filters:**\n" + "\n".join([f"- {f}" for f in filters_applied]))
    else:
        st.sidebar.success("No filters applied (showing all data)")

# Apply filters to dataframe
def apply_filters(df):
    """Apply global filters to dataframe"""
    
    df_filtered = df.copy()
    
    # Apply month filter
    if st.session_state.filter_month != 'all':
        df_filtered = df_filtered[df_filtered['month'] == st.session_state.filter_month]
    
    # Apply day filter
    if st.session_state.filter_day != 'all':
        df_filtered = df_filtered[df_filtered['day_of_week'] == st.session_state.filter_day]
    
    # Apply age filter
    df_filtered = df_filtered[
        (df_filtered['age'] >= st.session_state.filter_age_range[0]) &
        (df_filtered['age'] <= st.session_state.filter_age_range[1])
    ]
    
    # Apply job filter
    if st.session_state.filter_job != 'all':
        # NOTE: filter_job now holds the cleaned (lowercase, no space) value
        df_filtered = df_filtered[df_filtered['job'] == st.session_state.filter_job]
    
    # Apply education filter
    if st.session_state.filter_education != 'all':
        df_filtered = df_filtered[df_filtered['education'] == st.session_state.filter_education]
    
    # Apply marital filter
    if st.session_state.filter_marital != 'all':
        df_filtered = df_filtered[df_filtered['marital'] == st.session_state.filter_marital]
    
    return df_filtered

# Display global KPIs
def render_global_kpis(df_filtered, df_original):
    """Render global KPIs that appear on every page"""
    
    st.markdown("### 📊 Global Metrics")
    
    # Define columns for KPIs
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Total Customers KPI
    with col1:
        st.metric(
            "Total Customers",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df_original):,}" if len(df_filtered) != len(df_original) else None,
            help="Number of customers after filtering"
        )
    
    # Conversion Rate KPI
    with col2:
        # Check if 'y_binary' exists before summing
        if 'y_binary' in df_filtered.columns:
            conv_rate = (df_filtered['y_binary'].sum() / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
            orig_conv = (df_original['y_binary'].sum() / len(df_original)) * 100
            st.metric(
                "Conversion Rate",
                f"{conv_rate:.2f}%",
                delta=f"{conv_rate - orig_conv:.2f}%" if len(df_filtered) != len(df_original) else None,
                help="% who subscribed to term deposit"
            )
        else:
            st.metric("Conversion Rate", "N/A", help="Target column 'y' is missing.")

    
    # Avg Age KPI
    with col3:
        avg_age = df_filtered['age'].mean() if len(df_filtered) > 0 else 0
        orig_age = df_original['age'].mean()
        st.metric(
            "Avg Age",
            f"{avg_age:.1f}",
            delta=f"{avg_age - orig_age:.1f}" if len(df_filtered) != len(df_original) else None
        )
    
    # Avg Call Duration KPI
    with col4:
        avg_duration = df_filtered['duration'].mean() if len(df_filtered) > 0 else 0
        orig_duration = df_original['duration'].mean()
        st.metric(
            "Avg Call (sec)",
            f"{avg_duration:.0f}",
            delta=f"{avg_duration - orig_duration:.0f}" if len(df_filtered) != len(df_original) else None
        )
    
    # Avg Contacts KPI
    with col5:
        avg_campaign = df_filtered['campaign'].mean() if len(df_filtered) > 0 else 0
        orig_campaign = df_original['campaign'].mean()
        st.metric(
            "Avg Contacts",
            f"{avg_campaign:.2f}",
            delta=f"{avg_campaign - orig_campaign:.2f}" if len(df_filtered) != len(df_original) else None
        )
    
    # Data Coverage KPI
    with col6:
        filter_pct = (len(df_filtered) / len(df_original)) * 100 if len(df_original) > 0 else 0
        st.metric(
            "Data Coverage",
            f"{filter_pct:.1f}%",
            help="% of original data after filtering"
        )
    
    if len(df_filtered) != len(df_original):
        st.info(f"📌 **Filters Active:** Showing {len(df_filtered):,} of {len(df_original):,} customers ({filter_pct:.1f}%)")
    
    st.markdown("---")