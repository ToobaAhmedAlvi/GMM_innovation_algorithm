# pages/page1_data_exploration.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Exploration", page_icon="Chart", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional\\bank-additional-full.csv",sep=",")
    df.columns = df.columns.str.strip()
    df['y_binary'] = df['y'].map({'yes': 1, 'no': 0})
    return df

df = load_data()

st.title("Step 1: Data Exploration")

# Sidebar filters for interactive time dimensions
st.sidebar.markdown("## Interactive Filters")
st.sidebar.markdown("Filter data by time dimensions:")

# Month filter
all_months = ['All'] + sorted(df['month'].unique().tolist())
selected_month = st.sidebar.selectbox(
    "Select Month",
    options=all_months,
    index=0,
    help="Filter data by campaign month"
)

# Day of week filter
all_days = ['All'] + sorted(df['day_of_week'].unique().tolist())
selected_day = st.sidebar.selectbox(
    "Select Day of Week",
    options=all_days,
    index=0,
    help="Filter data by day of week"
)

# Apply filters
df_filtered = df.copy()
if selected_month != 'All':
    df_filtered = df_filtered[df_filtered['month'] == selected_month]
if selected_day != 'All':
    df_filtered = df_filtered[df_filtered['day_of_week'] == selected_day]

# Show filter status
if selected_month != 'All' or selected_day != 'All':
    filter_text = []
    if selected_month != 'All':
        filter_text.append(f"Month: {selected_month}")
    if selected_day != 'All':
        filter_text.append(f"Day: {selected_day}")
    st.info(f"Filtered by: {' | '.join(filter_text)} | Showing {len(df_filtered):,} of {len(df):,} records")

# Top KPIs
st.markdown("## Key Performance Indicators")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Customers", f"{len(df_filtered):,}", 
             help="Total number of customer records")

with col2:
    conversion_rate = (df_filtered['y_binary'].sum() / len(df_filtered)) * 100
    st.metric("Conversion Rate", f"{conversion_rate:.2f}%",
             help="Percentage who subscribed")

with col3:
    avg_age = df_filtered['age'].mean()
    st.metric("Avg Customer Age", f"{avg_age:.1f} yrs",
             help="Average age of customers")

with col4:
    avg_duration = df_filtered['duration'].mean()
    st.metric("Avg Call Duration", f"{avg_duration:.0f} sec",
             help="Average contact duration")

with col5:
    total_features = df.shape[1] - 2
    st.metric("Total Features", total_features,
             help="Number of attributes")

with col6:
    avg_campaign = df_filtered['campaign'].mean()
    st.metric("Avg Contacts", f"{avg_campaign:.2f}",
             help="Avg contacts per customer")

st.markdown("---")

# Target Distribution
st.markdown("## Target Variable Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    target_counts = df_filtered['y'].value_counts()
    yes_count = target_counts.get('yes', 0)
    no_count = target_counts.get('no', 0)
    
    total = yes_count + no_count
    yes_pct = (yes_count / total) * 100
    no_pct = (no_count / total) * 100
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['No', 'Yes'],
            values=[no_count, yes_count],
            hole=0.4,
            marker_colors=['#ff7f0e', '#2ca02c'],
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value:,}<br>(%{percent:.1%})',
            textfont_size=14
        )
    ])
    fig.update_layout(
        title="Subscription Distribution",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Target Statistics")
    st.write(f"**Subscribed (Yes):** {yes_count:,}")
    st.write(f"**Not Subscribed (No):** {no_count:,}")
    
    if yes_count > 0:
        ratio = no_count / yes_count
        st.write(f"**Class Ratio:** 1:{ratio:.2f}")

# Feature Analysis Tabs
st.markdown("## Feature Analysis")

tab1, tab2, tab3 = st.tabs(["Numerical Features", "Categorical Features", "Correlations"])

numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'day_of_week', 'poutcome']

with tab1:
    col1, col2 = st.columns(2)
    
    selected_num = st.selectbox("Select Numerical Feature", options=numerical_cols, index=0)
    
    with col1:
        fig = px.histogram(df_filtered, x=selected_num, 
                           title=f'Distribution of {selected_num}',
                           color_discrete_sequence=['#1f77b4'],
                           marginal='box')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df_filtered, x='y', y=selected_num,
                     title=f'{selected_num} by Subscription Status',
                     color='y',
                     color_discrete_map={'no': '#ff7f0e', 'yes': '#2ca02c'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    selected_cat = st.selectbox("Select Categorical Feature", options=categorical_cols, index=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        value_counts = df_filtered[selected_cat].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    title=f'Top 10 {selected_cat} Distribution',
                    labels={'x': selected_cat, 'y': 'Count'},
                    text=value_counts.values,
                    color_continuous_scale='Blues')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        conversion_by_cat = df_filtered.groupby(selected_cat)['y_binary'].agg(['sum', 'count'])
        conversion_by_cat['conversion_rate'] = (conversion_by_cat['sum'] / conversion_by_cat['count']) * 100
        conversion_by_cat = conversion_by_cat.sort_values('conversion_rate', ascending=False).head(10)
        
        fig = px.bar(x=conversion_by_cat.index, 
                    y=conversion_by_cat['conversion_rate'],
                    title=f'Conversion Rate by {selected_cat}',
                    labels={'x': selected_cat, 'y': 'Conversion Rate (%)'},
                    text=[f"{v:.2f}%" for v in conversion_by_cat['conversion_rate']],
                    color=conversion_by_cat['conversion_rate'],
                    color_continuous_scale='RdYlGn')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Feature Correlation Matrix")
    
    numeric_cols_for_corr = ['age', 'duration', 'campaign', 'pdays', 'previous',
                             'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                             'euribor3m', 'nr.employed', 'y_binary']
    
    corr_matrix = df_filtered[numeric_cols_for_corr].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=700,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Time dimension analysis
st.markdown("## Time Dimension Analysis")

col1, col2 = st.columns(2)

with col1:
    month_conv = df.groupby('month')['y_binary'].agg(['sum', 'count'])
    month_conv['conversion_rate'] = (month_conv['sum'] / month_conv['count']) * 100
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_conv = month_conv.reindex([m for m in month_order if m in month_conv.index])
    
    fig = px.bar(x=month_conv.index, y=month_conv['conversion_rate'],
                title='Conversion Rate by Month',
                labels={'x': 'Month', 'y': 'Conversion Rate (%)'},
                text=[f"{v:.2f}%" for v in month_conv['conversion_rate']],
                color=month_conv['conversion_rate'],
                color_continuous_scale='Viridis')
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    day_conv = df.groupby('day_of_week')['y_binary'].agg(['sum', 'count'])
    day_conv['conversion_rate'] = (day_conv['sum'] / day_conv['count']) * 100
    day_order = ['mon', 'tue', 'wed', 'thu', 'fri']
    day_conv = day_conv.reindex([d for d in day_order if d in day_conv.index])
    
    fig = px.bar(x=day_conv.index, y=day_conv['conversion_rate'],
                title='Conversion Rate by Day of Week',
                labels={'x': 'Day', 'y': 'Conversion Rate (%)'},
                text=[f"{v:.2f}%" for v in day_conv['conversion_rate']],
                color=day_conv['conversion_rate'],
                color_continuous_scale='Plasma')
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Navigation
if st.button("Proceed to Preprocessing & Cluster Selection", 
            use_container_width=True, type="primary"):
    st.switch_page("pages/page2_preprocessing.py")