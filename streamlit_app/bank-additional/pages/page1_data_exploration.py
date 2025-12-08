# pages/1_📊_Data_Exploration.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv",sep=",")
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
    df['y_bin'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df = load_data()

# Page header
st.title("📊 Step 1: Data Exploration & Statistical Analysis")
st.markdown("---")

# Overview Section
st.header("1.1 Dataset Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Total Columns", df.shape[1])
with col3:
    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display sample
with st.expander("📋 View Sample Data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# Data types summary
st.subheader("1.2 Data Types Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Numerical Columns**")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_df = pd.DataFrame({
        'Column': num_cols,
        'Data Type': [str(df[col].dtype) for col in num_cols],
        'Non-Null': [df[col].notna().sum() for col in num_cols],
        'Null': [df[col].isna().sum() for col in num_cols]
    })
    st.dataframe(num_df, use_container_width=True)
    st.info(f"**Total Numerical Features: {len(num_cols)}**")

with col2:
    st.markdown("**Categorical Columns**")
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in cat_cols:
        cat_cols.remove('y')  # Remove target
    cat_df = pd.DataFrame({
        'Column': cat_cols,
        'Unique Values': [df[col].nunique() for col in cat_cols],
        'Non-Null': [df[col].notna().sum() for col in cat_cols],
        'Null': [df[col].isna().sum() for col in cat_cols]
    })
    st.dataframe(cat_df, use_container_width=True)
    st.info(f"**Total Categorical Features: {len(cat_cols)}**")

# Missing values analysis
st.subheader("1.3 Missing Values Analysis")
missing_data = df.isnull().sum()
if missing_data.sum() == 0:
    st.success("✅ **No missing values found in the dataset!**")
else:
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Percentage': (missing_data.values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    st.dataframe(missing_df, use_container_width=True)

# Statistical Summary
st.header("1.4 Statistical Summary")

tab1, tab2 = st.tabs(["📊 Numerical Statistics", "📋 Categorical Statistics"])

with tab1:
    st.subheader("Descriptive Statistics for Numerical Features")
    st.dataframe(df[num_cols].describe().T.style.format("{:.2f}"), use_container_width=True)
    
    # Interactive numerical distribution
    st.subheader("Distribution Analysis")
    selected_num = st.selectbox("Select numerical feature to analyze:", num_cols, key='num_dist')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=selected_num, nbins=50, 
                          title=f'Distribution of {selected_num}',
                          marginal='box')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=selected_num, 
                     title=f'Box Plot of {selected_num}')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics for selected feature
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{df[selected_num].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[selected_num].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[selected_num].std():.2f}")
    with col4:
        st.metric("Range", f"{df[selected_num].max() - df[selected_num].min():.2f}")

with tab2:
    st.subheader("Frequency Analysis for Categorical Features")
    
    selected_cat = st.selectbox("Select categorical feature to analyze:", cat_cols, key='cat_dist')
    
    value_counts = df[selected_cat].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                     title=f'Frequency Distribution of {selected_cat}',
                     labels={'x': selected_cat, 'y': 'Count'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                     title=f'Proportion of {selected_cat}')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Frequency table
    st.subheader("Frequency Table")
    freq_df = pd.DataFrame({
        'Category': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / len(df) * 100).round(2)
    })
    st.dataframe(freq_df, use_container_width=True)

# Correlation Analysis
st.header("1.5 Correlation Analysis")

correlation_matrix = df[num_cols].corr()

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.imshow(correlation_matrix, 
                    text_auto='.2f',
                    title='Correlation Heatmap',
                    color_continuous_scale='RdBu',
                    aspect='auto')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Highly Correlated Pairs")
    st.markdown("**Correlation > 0.7 or < -0.7**")
    
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append({
                    'Feature 1': correlation_matrix.columns[i],
                    'Feature 2': correlation_matrix.columns[j],
                    'Correlation': f"{corr_val:.3f}"
                })
    
    if high_corr:
        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
    else:
        st.info("No highly correlated feature pairs found.")

# Target Variable Analysis
st.header("1.6 Target Variable Analysis")

target_counts = df['y'].value_counts()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    fig = go.Figure(data=[go.Pie(labels=target_counts.index, 
                                  values=target_counts.values,
                                  hole=0.3)])
    fig.update_layout(title='Target Distribution (Subscription)',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Subscribed (Yes)", f"{target_counts.get('yes', 0):,}")
    st.metric("Not Subscribed (No)", f"{target_counts.get('no', 0):,}")

with col3:
    yes_pct = target_counts.get('yes', 0) / len(df) * 100
    no_pct = target_counts.get('no', 0) / len(df) * 100
    st.metric("Yes %", f"{yes_pct:.2f}%")
    st.metric("No %", f"{no_pct:.2f}%")

# Feature Distribution by Target
st.subheader("Feature Distribution by Target")

selected_feature = st.selectbox("Select feature to compare with target:", 
                                num_cols + cat_cols, key='target_comp')

if selected_feature in num_cols:
    fig = px.box(df, x='y', y=selected_feature, 
                 color='y',
                 title=f'{selected_feature} by Subscription Status')
    st.plotly_chart(fig, use_container_width=True)
else:
    # Categorical comparison
    cross_tab = pd.crosstab(df[selected_feature], df['y'], normalize='index') * 100
    fig = px.bar(cross_tab, 
                 title=f'Subscription Rate by {selected_feature}',
                 labels={'value': 'Percentage', 'variable': 'Subscription'},
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# Summary Statistics Table
st.header("1.7 Complete Summary Table")

summary_data = {
    'Metric': ['Total Records', 'Features', 'Numerical Features', 'Categorical Features', 
               'Missing Values', 'Duplicate Rows', 'Memory Usage (MB)', 'Subscription Rate (%)'],
    'Value': [
        f"{df.shape[0]:,}",
        df.shape[1],
        len(num_cols),
        len(cat_cols),
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
        f"{df['y_bin'].mean() * 100:.2f}"
    ]
}

st.table(pd.DataFrame(summary_data))

# Download processed data
st.markdown("---")
st.subheader("💾 Download Data")

col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv,
        file_name="bank_marketing_data.csv",
        mime="text/csv",
    )

with col2:
    summary_csv = df.describe().to_csv()
    st.download_button(
        label="📥 Download Statistical Summary (CSV)",
        data=summary_csv,
        file_name="statistical_summary.csv",
        mime="text/csv",
    )

st.success("✅ Data exploration complete! Proceed to **Business Context** page.")
