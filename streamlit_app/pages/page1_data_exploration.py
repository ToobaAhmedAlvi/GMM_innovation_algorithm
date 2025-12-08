# pages/page1_data_exploration.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os 
import warnings

warnings.filterwarnings('ignore')

# Set matplotlib style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# === MUST BE FIRST ===
st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

sys.path.append('..')
try:
    from common_components import apply_custom_css, render_navigation, render_sidebar_filters, apply_filters, render_global_kpis
except ImportError:
    st.error("Error loading common components. Please ensure 'common_components.py' is in the parent directory (..).")
    st.stop() 

apply_custom_css()

# --- Feature Definitions ---
NUMERICAL_FEATURES = ['age','duration','campaign','pdays','previous','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
TARGET_COL = 'y'
TARGET_COL_BINARY = 'y_binary'

# --- DATA LOADING ---
@st.cache_data
def load_data_final():
    """Loads data from the .xlsx or .csv file and performs necessary cleaning."""
    paths_to_try = [
        "bank-additional/bank-additional-full.xlsx",
        "bank-additional/bank-additional-full.csv",
        "../bank-additional/bank-additional-full.xlsx",
        "../bank-additional/bank-additional-full.csv",
        "bank-additional\\bank-additional-full.xlsx",
        "bank-additional\\bank-additional-full.csv"
    ]
    df = pd.DataFrame()

    for file_path in paths_to_try:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path, sep=';')
                break
            except Exception:
                continue
    
    if df.empty:
        st.error(f"Could not load data. Please ensure 'bank-additional-full' file is accessible.")
        return pd.DataFrame()
    df.columns = df.columns.str.lower()
    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_', regex=False)
    
    # GUARANTEE NUMERIC TYPE and handle NaNs
    for c in NUMERICAL_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce') 
            df[c] = df[c].fillna(df[c].median()) 
    
    # Clean categorical data
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
            
    # Create Binary Target Column
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
        df[TARGET_COL_BINARY] = (df[TARGET_COL] == 'yes').astype(int)
    
    return df

df_original = load_data_final()

if df_original.empty:
    st.stop()

# --- STREAMLIT UI & FILTERING ---
st.title("📊 Step 1: Data Exploration & Statistical Analysis")

render_navigation("data_exploration")
render_sidebar_filters(df_original)

try:
    df_filtered = apply_filters(df_original) 
except Exception as e:
    st.error(f"Error applying filters: {e}. Displaying unfiltered data.")
    df_filtered = df_original 

render_global_kpis(df_filtered, df_original)

if len(df_filtered) == 0:
    st.warning("⚠️ The current filter selection results in zero data points. Please adjust the filters.")
    st.stop()

st.markdown("---")

# =================================================================
# 1. Dataset Statistics Table (FIXED - No Arrow Error)
# =================================================================
st.markdown("## 📋 Dataset Statistics")

with st.expander("🔍 View Detailed Dataset Information", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Information")
        # Fix: Convert all values to strings to avoid Arrow serialization error
        stats_df = pd.DataFrame({
            'Metric': ['Total Rows', 'Total Columns', 'Numerical Features', 'Categorical Features', 'Target Variable', 'Missing Values'],
            'Value': [
                str(len(df_filtered)),
                str(df_filtered.shape[1]),
                str(len(NUMERICAL_FEATURES)),
                str(len(CATEGORICAL_FEATURES)),
                'y (binary: yes/no)',
                str(df_filtered.isnull().sum().sum())
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Data Types Distribution")
        
        # Count actual feature types correctly
        num_count = len([c for c in NUMERICAL_FEATURES if c in df_filtered.columns])
        cat_count = len([c for c in CATEGORICAL_FEATURES if c in df_filtered.columns])
        target_count = 2  # 'y' and 'y_binary'
        
        # Create proper pie chart with matplotlib
        type_data = pd.DataFrame({
            'Type': ['Numerical', 'Categorical', 'Target'],
            'Count': [num_count, cat_count, target_count]
        })
        
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        wedges, texts, autotexts = ax.pie(
            type_data['Count'], 
            labels=type_data['Type'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax.set_title("Feature Types Distribution", fontsize=12, fontweight='bold')
        
        # Make percentage text bold and white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# =================================================================
# 2. Numerical Feature Analysis (FIXED - Working Plots)
# =================================================================
st.markdown("## 📈 Numerical Feature Distribution & Analysis")

selected_num_feature = st.selectbox(
    "Select Numerical Feature to Analyze:", 
    NUMERICAL_FEATURES, 
    key='num_feature_select'
)

if selected_num_feature in df_filtered.columns:
    
    # Get the data for the selected feature
    feature_data = df_filtered[selected_num_feature].dropna()
    
    if len(feature_data) == 0:
        st.warning(f"No data available for {selected_num_feature} with current filters.")
    else:
        st.subheader(f"Analysis of: {selected_num_feature}")
        
        # Display statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Count", f"{len(feature_data):,}")
        with col2:
            st.metric("Mean", f"{feature_data.mean():.2f}")
        with col3:
            st.metric("Median", f"{feature_data.median():.2f}")
        with col4:
            st.metric("Std Dev", f"{feature_data.std():.2f}")
        with col5:
            st.metric("Range", f"{feature_data.min():.0f} - {feature_data.max():.0f}")
        
        st.markdown("---")
        
        # Create side-by-side visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Histogram with KDE
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(feature_data, bins=40, kde=True, ax=ax, color='#1f77b4')
                ax.set_title(f"Distribution of {selected_num_feature}", fontsize=14, fontweight='bold')
                ax.set_xlabel(selected_num_feature, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error creating histogram: {str(e)}")
        
        with col2:
            try:
                # Boxplot
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x=feature_data, ax=ax, color='#2ca02c')
                ax.set_title(f"Boxplot of {selected_num_feature}", fontsize=14, fontweight='bold')
                ax.set_xlabel(selected_num_feature, fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error creating boxplot: {str(e)}")
        
        st.markdown("---")
        
        # Comparison by Target Variable
        st.subheader(f"{selected_num_feature} by Term Deposit Subscription")
        
        if TARGET_COL in df_filtered.columns:
            try:
                # Create comparison visualization
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Separate data by target
                yes_data = df_filtered[df_filtered[TARGET_COL] == 'yes'][selected_num_feature].dropna()
                no_data = df_filtered[df_filtered[TARGET_COL] == 'no'][selected_num_feature].dropna()
                
                # Side-by-side boxplots
                data_to_plot = [no_data, yes_data]
                positions = [1, 2]
                colors = ['#ff7f0e', '#2ca02c']
                
                bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                               labels=['No', 'Yes'])
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f"{selected_num_feature} by Subscription Status", fontsize=14, fontweight='bold')
                ax.set_xlabel("Subscribed to Term Deposit", fontsize=12)
                ax.set_ylabel(selected_num_feature, fontsize=12)
                ax.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig)
                plt.close()
                
                # Show comparison statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Not Subscribed (No)**")
                    st.write(f"Count: {len(no_data):,}")
                    st.write(f"Mean: {no_data.mean():.2f}")
                    st.write(f"Median: {no_data.median():.2f}")
                
                with col2:
                    st.markdown("**Subscribed (Yes)**")
                    st.write(f"Count: {len(yes_data):,}")
                    st.write(f"Mean: {yes_data.mean():.2f}")
                    st.write(f"Median: {yes_data.median():.2f}")
                
                with col3:
                    st.markdown("**Difference**")
                    mean_diff = yes_data.mean() - no_data.mean()
                    median_diff = yes_data.median() - no_data.median()
                    st.write(f"Mean Δ: {mean_diff:+.2f}")
                    st.write(f"Median Δ: {median_diff:+.2f}")
                    
                    # Statistical significance indicator
                    if abs(mean_diff) > no_data.std() * 0.5:
                        st.success("📊 Notable difference")
                    else:
                        st.info("📊 Similar distributions")
                
            except Exception as e:
                st.error(f"Error creating comparison plot: {str(e)}")
else:
    st.warning(f"Feature '{selected_num_feature}' not found in data.")

st.markdown("---")

# =================================================================
# 3. Categorical Feature Analysis
# =================================================================
st.markdown("## 📊 Categorical Feature Analysis")

selected_cat_feature = st.selectbox(
    "Select Categorical Feature to Analyze:", 
    CATEGORICAL_FEATURES, 
    key='cat_feature_select'
)

if selected_cat_feature in df_filtered.columns:
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Count distribution
            value_counts = df_filtered[selected_cat_feature].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            value_counts.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title(f"Top 10 {selected_cat_feature} Distribution", fontsize=14, fontweight='bold')
            ax.set_xlabel("Count", fontsize=12)
            ax.set_ylabel(selected_cat_feature, fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, v in enumerate(value_counts):
                ax.text(v, i, f' {v:,}', va='center')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Conversion rate by category
            if TARGET_COL_BINARY in df_filtered.columns:
                conversion_by_cat = df_filtered.groupby(selected_cat_feature)[TARGET_COL_BINARY].agg(['sum', 'count'])
                conversion_by_cat['conversion_rate'] = (conversion_by_cat['sum'] / conversion_by_cat['count']) * 100
                conversion_by_cat = conversion_by_cat.sort_values('conversion_rate', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                conversion_by_cat['conversion_rate'].plot(kind='barh', ax=ax, color='coral')
                ax.set_title(f"Conversion Rate by {selected_cat_feature}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Conversion Rate (%)", fontsize=12)
                ax.set_ylabel(selected_cat_feature, fontsize=12)
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, v in enumerate(conversion_by_cat['conversion_rate']):
                    ax.text(v, i, f' {v:.1f}%', va='center')
                
                st.pyplot(fig)
                plt.close()
        
        # Stacked bar chart
        st.subheader(f"Subscription Status by {selected_cat_feature}")
        fig_cat_counts = px.histogram(
            df_filtered, 
            x=selected_cat_feature, 
            color=TARGET_COL,
            title=f'Total Client Counts by {selected_cat_feature} and Deposit Success',
            color_discrete_map={'no': '#ff7f0e', 'yes': '#2ca02c'},
            barmode='stack',
            text_auto=True
        )
        fig_cat_counts.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat_counts, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error plotting Categorical Analysis: {str(e)}")

st.markdown("---")

# =================================================================
# 4. Correlation Heatmap (FIXED - Matplotlib Version)
# =================================================================
st.markdown("## 🕸️ Feature Correlation Analysis")

numerical_df = df_filtered[NUMERICAL_FEATURES + [TARGET_COL_BINARY]].copy()
corr = numerical_df.corr()

if not corr.empty and len(corr.columns) > 1:
    try:
        st.subheader(f"Correlation with Target ('{TARGET_COL}'):")
        
        # Filter correlation values for the target column
        target_corr = corr[TARGET_COL_BINARY].sort_values(ascending=False).drop(TARGET_COL_BINARY)
        
        # Display top correlations
        col_list = st.columns(min(len(target_corr), 5))
        for i, (feat, val) in enumerate(target_corr.items()):
            with col_list[i % 5]:
                st.metric(
                    label=feat.replace('_', ' ').title(), 
                    value=f"{val:+.4f}"
                )
        
        # Correlation Heatmap using Matplotlib/Seaborn
        st.subheader("Full Correlation Matrix Heatmap")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with seaborn
        sns.heatmap(
            corr, 
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error in Correlation Analysis: {str(e)}")

st.markdown("---")

# =================================================================
# 5. Time Dimension Analysis (CONVERSION RATES)
# =================================================================
st.markdown("## 🕰️ Time Dimension Analysis: Conversion Rates")

st.info("""
**📊 What these charts show:**
- **Conversion Rate** = (Number of 'Yes' / Total Contacts) × 100%
- Higher percentage = Better success rate for that time period
- Use filters to analyze specific segments (e.g., "Which day is best for admin jobs?")
""")

if 'month' in df_filtered.columns and 'day_of_week' in df_filtered.columns:
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Conversion Rate by Month")
        try:
            # Calculate conversion rate (not just counts)
            month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            
            # Use ORIGINAL data if no month filter is applied
            data_for_month = df_original if st.session_state.get('filter_month', 'All') == 'All' else df_filtered
            
            data_for_month['month_num'] = data_for_month['month'].str.lower().map(month_order)
            
            # Calculate conversion rate per month
            monthly_stats = data_for_month.groupby(['month', 'month_num']).agg({
                TARGET_COL_BINARY: ['sum', 'count']
            }).reset_index()
            monthly_stats.columns = ['month', 'month_num', 'yes_count', 'total_count']
            monthly_stats['conversion_rate'] = (monthly_stats['yes_count'] / monthly_stats['total_count']) * 100
            monthly_stats = monthly_stats.sort_values('month_num')
            
            # Create bar chart with matplotlib for better control
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(monthly_stats['month'], monthly_stats['conversion_rate'], color='#2ca02c', alpha=0.7)
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, monthly_stats['conversion_rate'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Month', fontsize=12, fontweight='bold')
            ax.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
            ax.set_title('Conversion Rate by Campaign Month', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(monthly_stats['conversion_rate']) * 1.2)  # Add space for labels
            
            # Add average line
            avg_rate = monthly_stats['conversion_rate'].mean()
            ax.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_rate:.1f}%')
            ax.legend()
            
            st.pyplot(fig)
            plt.close()
            
            # Show insight
            best_month = monthly_stats.loc[monthly_stats['conversion_rate'].idxmax(), 'month']
            best_rate = monthly_stats['conversion_rate'].max()
            st.success(f"✅ Best month: **{best_month.upper()}** with {best_rate:.2f}% conversion rate")

        except Exception as e:
            st.error(f"Error in monthly analysis: {str(e)}")

    with col2:
        st.subheader("📆 Conversion Rate by Day of Week")
        try:
            day_order = ['mon', 'tue', 'wed', 'thu', 'fri']
            
            # Use ORIGINAL data if no day filter is applied
            data_for_day = df_original if st.session_state.get('filter_day', 'All') == 'All' else df_filtered
            
            data_for_day['day_of_week_cat'] = pd.Categorical(
                data_for_day['day_of_week'].str.lower(), 
                categories=day_order, 
                ordered=True
            )
            
            # Calculate conversion rate per day
            daily_stats = data_for_day.groupby('day_of_week_cat').agg({
                TARGET_COL_BINARY: ['sum', 'count']
            }).reset_index()
            daily_stats.columns = ['day_of_week', 'yes_count', 'total_count']
            daily_stats['conversion_rate'] = (daily_stats['yes_count'] / daily_stats['total_count']) * 100
            daily_stats = daily_stats.sort_values('day_of_week')
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(
                daily_stats['day_of_week'].astype(str), 
                daily_stats['conversion_rate'], 
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(daily_stats)],
                alpha=0.7
            )
            
            # Add value labels
            for bar, rate in zip(bars, daily_stats['conversion_rate']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Day of Week', fontsize=12, fontweight='bold')
            ax.set_ylabel('Conversion Rate (%)', fontsize=12, fontweight='bold')
            ax.set_title('Conversion Rate by Day of Week', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(daily_stats['conversion_rate']) * 1.2)
            
            # Add average line
            avg_rate = daily_stats['conversion_rate'].mean()
            ax.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_rate:.1f}%')
            ax.legend()
            
            st.pyplot(fig)
            plt.close()
            
            # Show insight
            best_day = daily_stats.loc[daily_stats['conversion_rate'].idxmax(), 'day_of_week']
            best_rate = daily_stats['conversion_rate'].max()
            st.success(f"✅ Best day: **{str(best_day).upper()}** with {best_rate:.2f}% conversion rate")
            
        except Exception as e:
            st.error(f"Error in daily analysis: {str(e)}")
    
    # Additional time insights
    st.markdown("---")
    st.markdown("### 💡 Time-Based Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Best time combinations
        if len(df_filtered) > 100:
            best_combo = df_filtered.groupby(['month', 'day_of_week'])[TARGET_COL_BINARY].mean().sort_values(ascending=False).head(1)
            if len(best_combo) > 0:
                month, day = best_combo.index[0]
                rate = best_combo.values[0] * 100
                st.info(f"""
                **🎯 Best Time Combo:**
                {month.upper()} + {day.upper()}
                
                Conversion: {rate:.1f}%
                """)
    
    with col2:
        # Week pattern
        if 'day_of_week' in df_filtered.columns:
            weekday_avg = df_filtered[df_filtered['day_of_week'].isin(['mon', 'tue', 'wed'])][TARGET_COL_BINARY].mean() * 100
            weekend_avg = df_filtered[df_filtered['day_of_week'].isin(['thu', 'fri'])][TARGET_COL_BINARY].mean() * 100
            
            st.info(f"""
            **📊 Week Pattern:**
            Early Week: {weekday_avg:.1f}%
            Late Week: {weekend_avg:.1f}%
            
            Difference: {abs(weekday_avg - weekend_avg):.1f}%
            """)
    
    with col3:
        # Seasonal pattern
        if 'month' in df_filtered.columns:
            spring = df_filtered[df_filtered['month'].isin(['mar', 'apr', 'may'])][TARGET_COL_BINARY].mean() * 100
            fall = df_filtered[df_filtered['month'].isin(['sep', 'oct', 'nov'])][TARGET_COL_BINARY].mean() * 100
            
            st.info(f"""
            **🍂 Seasonal Pattern:**
            Spring: {spring:.1f}%
            Fall: {fall:.1f}%
            
            Best: {'Spring' if spring > fall else 'Fall'}
            """)

st.markdown("---")

# Business Insights
st.markdown("## 💡 Key Business Insights")

if len(df_filtered) > 0:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **📊 Data Quality:**
        - Clean dataset
        - Well-structured features
        - Suitable for clustering
        """)
    
    with col2:
        conv_rate = df_filtered[TARGET_COL_BINARY].mean() * 100
        st.success(f"""
        **🎯 Conversion Rate:**
        - Current: {conv_rate:.2f}%
        - Clustering will identify high-value segments
        """)
    
    with col3:
        st.warning("""
        **🔍 Next Steps:**
        - Proceed to preprocessing
        - Encode categorical features
        - Determine optimal clusters
        """)