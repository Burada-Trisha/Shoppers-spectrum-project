import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Shopper Spectrum - E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Dark glassmorphic styling */
    .reportview-container {
        background: #0f172a;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
    }
    .segment-card {
        border-radius: 12px;
        padding: 20px;
        color: white;
        margin-bottom: 15px;
        border-left: 6px solid;
    }
    .segment-high {
        background: rgba(16, 185, 129, 0.15);
        border-color: #10b981;
    }
    .segment-regular {
        background: rgba(59, 130, 246, 0.15);
        border-color: #3b82f6;
    }
    .segment-occasional {
        background: rgba(245, 158, 11, 0.15);
        border-color: #f59e0b;
    }
    .segment-atrisk {
        background: rgba(239, 68, 68, 0.15);
        border-color: #ef4444;
    }
    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0px 0px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
        border-color: #6366f1 !important;
    }
</style>
""", unsafe_allowed_html=True)

# Helper function to load pickled models/data
@st.cache_resource
def load_models():
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('cluster_labels.pkl', 'rb') as f:
            cluster_labels = pickle.load(f)
        with open('product_recommendations.pkl', 'rb') as f:
            recs_dict = pickle.load(f)
        return kmeans, scaler, cluster_labels, recs_dict
    except Exception as e:
        return None, None, None, None

# Helper function to load datasets
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_transactions.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        rfm = pd.read_csv('customer_rfm.csv')
        return df, rfm
    except Exception as e:
        return None, None

# Load models and data
kmeans, scaler, cluster_labels, recs_dict = load_models()
df_transactions, df_rfm = load_data()

# App Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Shopper Spectrum
    </h1>
    <p style="color: #94a3b8; font-size: 1.15rem; margin-top: 10px;">
        Customer Segmentation & Product Recommendations in E-Commerce
    </p>
</div>
""", unsafe_allowed_html=True)

if kmeans is None or df_transactions is None:
    st.error("⚠️ Model assets or datasets not found! Please run the pipeline script first to generate files.")
    st.info("Execute: `python pipeline.py` in your terminal to build the models.")
    st.stop()

# Sidebar Setup
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="font-size: 1.5rem; color: #818cf8;">📊 Project Control Center</h2>
    <p style="color: #94a3b8; font-size: 0.85rem;">Manage and analyze customer profiles & recommendations</p>
</div>
""", unsafe_allowed_html=True)

# Add sidebar filters
st.sidebar.subheader("🔍 Transactions Filters")
countries = ["All"] + sorted(df_transactions['Country'].unique().tolist())
selected_country = st.sidebar.selectbox("Filter Dashboard by Country", countries)

# Filter transactions based on selection
if selected_country != "All":
    filtered_df = df_transactions[df_transactions['Country'] == selected_country]
else:
    filtered_df = df_transactions

# App Tabs
tab_dash, tab_segment, tab_recs, tab_insights, tab_data = st.tabs([
    "📊 Analytics Dashboard", 
    "🎯 Customer Segmenter", 
    "🛍️ Product Recommendations", 
    "🔬 Clustering Insights",
    "📂 Dataset Preview"
])

# ----------------- TAB 1: ANALYTICS DASHBOARD -----------------
with tab_dash:
    st.subheader("📈 E-Commerce Business Performance")
    
    # Calculate KPIs
    total_revenue = filtered_df['TotalSpend'].sum()
    total_tx = filtered_df['InvoiceNo'].nunique()
    total_customers = filtered_df['CustomerID'].nunique()
    avg_order_val = total_revenue / total_tx if total_tx > 0 else 0
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value">£{total_revenue:,.2f}</div>
        </div>
        """, unsafe_allowed_html=True)
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Transactions</div>
            <div class="metric-value">{total_tx:,}</div>
        </div>
        """, unsafe_allowed_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Unique Customers</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allowed_html=True)
    with col_kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Order Value</div>
            <div class="metric-value">£{avg_order_val:,.2f}</div>
        </div>
        """, unsafe_allowed_html=True)

    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🛒 Top 10 Selling Products")
        top_products = filtered_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_prod = px.bar(
            top_products, 
            x='Quantity', 
            y='Description', 
            orientation='h',
            labels={'Quantity': 'Total Quantity Sold', 'Description': 'Product Name'},
            color='Quantity',
            color_continuous_scale='purples',
            template='plotly_dark'
        )
        fig_prod.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with col_chart2:
        st.subheader("🌍 Transaction Share by Country (Top 10)")
        country_shares = filtered_df['Country'].value_counts().head(10).reset_index()
        country_shares.columns = ['Country', 'Count']
        fig_country = px.pie(
            country_shares,
            values='Count',
            names='Country',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu,
            template='plotly_dark'
        )
        fig_country.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_country, use_container_width=True)

    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        st.subheader("📅 Monthly Sales Trend")
        filtered_df['YearMonth'] = filtered_df['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_sales = filtered_df.groupby('YearMonth')['TotalSpend'].sum().reset_index()
        fig_trend = px.line(
            monthly_sales,
            x='YearMonth',
            y='TotalSpend',
            labels={'TotalSpend': 'Revenue (£)', 'YearMonth': 'Month'},
            markers=True,
            template='plotly_dark'
        )
        fig_trend.update_traces(line_color='#818cf8', line_width=3, marker=dict(size=8, color='#c084fc'))
        fig_trend.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_chart4:
        st.subheader("👥 Customer Segments Breakdown")
        segment_counts = df_rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig_segment = px.bar(
            segment_counts,
            x='Segment',
            y='Count',
            color='Segment',
            color_discrete_map={
                'High-Value': '#10b981',
                'Regular': '#3b82f6',
                'Occasional': '#f59e0b',
                'At-Risk': '#ef4444'
            },
            template='plotly_dark'
        )
        fig_segment.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_segment, use_container_width=True)

# ----------------- TAB 2: CUSTOMER SEGMENTER -----------------
with tab_segment:
    st.subheader("🎯 Real-Time RFM Customer Classifier")
    st.write("Enter the customer's behavioral attributes below to determine their marketing segment:")
    
    col_inp1, col_inp2, col_inp3 = st.columns(3)
    with col_inp1:
        recency = st.number_input(
            "Recency (Days since last purchase)", 
            min_value=1, max_value=365, value=30, step=1,
            help="How many days ago did the customer purchase? Smaller is more recent."
        )
    with col_inp2:
        frequency = st.number_input(
            "Frequency (Total number of invoices)", 
            min_value=1, max_value=200, value=5, step=1,
            help="How many transactions has the customer made?"
        )
    with col_inp3:
        monetary = st.number_input(
            "Monetary (Total spending in £)", 
            min_value=1.0, max_value=100000.0, value=250.0, step=10.0,
            help="What is the sum total of purchases by this customer?"
        )
        
    if st.button("🚀 Predict Customer Segment", use_container_width=True):
        # Apply same preprocessing (log1p + scale)
        input_data = np.log1p([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        segment_label = cluster_labels[cluster]
        
        # Mapping labels to classes for design
        card_class = "segment-regular"
        desc = ""
        action = ""
        
        if segment_label == "High-Value":
            card_class = "segment-high"
            desc = "This customer buys very frequently, spends highly, and has purchased extremely recently."
            action = "Offer exclusive VIP access, early product launches, and high-tier rewards program."
        elif segment_label == "Regular":
            card_class = "segment-regular"
            desc = "This customer is a steady purchaser who buys consistently but at average monetary values."
            action = "Cross-sell related products, offer milestone loyalty points, and run multi-buy discounts."
        elif segment_label == "Occasional":
            card_class = "segment-occasional"
            desc = "This customer makes purchases rarely or has recently engaged but has low overall transaction counts."
            action = "Send personalized recommendations based on past purchases, offer discount codes to boost frequency."
        elif segment_label == "At-Risk":
            card_class = "segment-atrisk"
            desc = "This customer hasn't purchased in a long time and has low purchase frequency."
            action = "Run win-back re-engagement email campaigns, provide high-discount promo codes, or conduct feedback surveys."
            
        st.markdown(f"""
        <div class="segment-card {card_class}">
            <h3 style="margin-top: 0; font-size: 1.6rem; font-weight: 700;">{segment_label} Customer</h3>
            <p style="font-size: 1rem; margin-bottom: 10px;"><strong>Description:</strong> {desc}</p>
            <p style="font-size: 1rem; margin-bottom: 0;"><strong>Recommended Action:</strong> {action}</p>
        </div>
        """, unsafe_allowed_html=True)

# ----------------- TAB 3: PRODUCT RECOMMENDATIONS -----------------
with tab_recs:
    st.subheader("🛍️ Collaborative Product Recommendation System")
    st.write("Select a product description below to discover matching items recommended by item-based collaborative filtering:")
    
    unique_products = sorted(list(recs_dict.keys()))
    selected_product = st.selectbox("Choose a Product Name:", unique_products)
    
    if st.button("🎁 Generate Recommendations", use_container_width=True):
        recs = recs_dict.get(selected_product, [])
        if not recs:
            st.warning("No recommendations found for this product. Try another description.")
        else:
            st.success(f"Top 5 recommended products matching **{selected_product}**:")
            
            # Display recommendations in cards
            cols = st.columns(5)
            for idx, (rec_name, score) in enumerate(recs[:5]):
                with cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; min-height: 200px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div style="font-size: 1.25rem; font-weight: bold; color: #818cf8; margin-bottom: 10px;">#{idx+1}</div>
                        <div style="font-size: 0.9rem; font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; line-height: 1.2; height: 3.6em;">{rec_name}</div>
                        <div style="margin-top: 15px;">
                            <span style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8;">Similarity Match</span>
                            <br>
                            <span style="font-size: 1.1rem; font-weight: 700; color: #10b981;">{score * 100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allowed_html=True)
                    
            # Let's also print them in a table format below
            st.markdown("### Details Table")
            recs_df = pd.DataFrame(recs[:5], columns=['Product Name', 'Similarity Score'])
            recs_df['Similarity Score'] = recs_df['Similarity Score'].map(lambda x: f"{x * 100:.2f}%")
            st.table(recs_df)

# ----------------- TAB 4: CLUSTERING INSIGHTS -----------------
with tab_insights:
    st.subheader("🔬 Clustering Methodology & Insights")
    
    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown("""
        ### RFM Segmentation Methodology
        Customer clustering was performed on three core metrics:
        1. **Recency (R)**: Days since last purchase. Low recency is better (active customer).
        2. **Frequency (F)**: Total number of unique orders placed. High frequency represents customer loyalty.
        3. **Monetary (M)**: Total financial spend. High monetary signifies high financial value.
        
        **Model Pipeline**:
        * Apply `log1p` transformation to normalise right-skewed RFM variables.
        * Apply standard scaling (`StandardScaler`) to give equal weight to each attribute.
        * Fit a **KMeans Clustering** algorithm with **k=4** (selected using the elbow method and silhouette analysis).
        """)
    with col_ins2:
        # Cluster Profile Table
        st.markdown("### Cluster Characteristics Profiles (Averages)")
        cluster_profiles = df_rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
        cluster_profiles.columns = ['Segment', 'Avg Recency (Days)', 'Avg Frequency (Orders)', 'Avg Monetary (£)']
        # format table
        cluster_profiles['Avg Recency (Days)'] = cluster_profiles['Avg Recency (Days)'].round(1)
        cluster_profiles['Avg Frequency (Orders)'] = cluster_profiles['Avg Frequency (Orders)'].round(1)
        cluster_profiles['Avg Monetary (£)'] = cluster_profiles['Avg Monetary (£)'].map(lambda x: f"£{x:,.2f}")
        st.dataframe(cluster_profiles, hide_index=True)
        
    st.markdown("---")
    st.subheader("📊 Cluster Visualisation (Recency vs Monetary)")
    
    # 2D scatter of RFM (colored by cluster)
    fig_scatter = px.scatter(
        df_rfm,
        x='Recency',
        y='Monetary',
        color='Segment',
        log_y=True,
        title="Customer Segments (Monetary vs Recency)",
        color_discrete_map={
            'High-Value': '#10b981',
            'Regular': '#3b82f6',
            'Occasional': '#f59e0b',
            'At-Risk': '#ef4444'
        },
        labels={'Recency': 'Recency (Days)', 'Monetary': 'Monetary (£, Log Scale)'},
        template='plotly_dark',
        opacity=0.7
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- TAB 5: DATASET PREVIEW -----------------
with tab_data:
    st.subheader("📂 Cleaned Transactional Dataset")
    st.write(f"Displaying a sample of the cleaned transaction dataset (Total rows: {len(df_transactions):,}):")
    st.dataframe(df_transactions.head(100), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📈 Summary Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.write("### Numerical Columns Summary")
        st.dataframe(df_transactions[['Quantity', 'UnitPrice', 'TotalSpend']].describe())
    with col_stat2:
        st.write("### Dataset Metadata")
        meta_df = pd.DataFrame({
            "Metric": ["Unique Invoices", "Unique StockCodes", "Unique Customers", "Total Sales Count", "Earliest Invoice", "Latest Invoice"],
            "Value": [
                df_transactions['InvoiceNo'].nunique(),
                df_transactions['StockCode'].nunique(),
                df_transactions['CustomerID'].nunique(),
                len(df_transactions),
                df_transactions['InvoiceDate'].min().strftime('%Y-%m-%d %H:%M:%S'),
                df_transactions['InvoiceDate'].max().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })
        st.dataframe(meta_df, hide_index=True)
