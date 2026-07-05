import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

def run_pipeline():
    print("Step 1: Loading dataset...")
    csv_path = "online_retail.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    print(f"Dataset shape: {df.shape}")
    
    print("Step 2: Cleaning dataset...")
    # Remove missing CustomerID
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(float).astype(int).astype(str)
    
    # Exclude cancelled invoices
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Remove non-positive quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Clean product descriptions (strip and convert to uppercase)
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()
    df = df[df['Description'] != '']
    
    # Parse date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print(f"Cleaned dataset shape: {df.shape}")
    
    print("Step 3: Calculating RFM metrics...")
    # Reference date (1 day after the last transaction)
    ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Compute RFM
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSpend': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSpend': 'Monetary'
    })
    
    print(f"Number of unique customers: {len(rfm)}")
    
    # We will log-transform RFM to handle skewness before scaling
    rfm_log = np.log1p(rfm)
    
    print("Step 4: Standardizing and Clustering RFM...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # Run KMeans with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm['Cluster'] = clusters
    
    # Programmatic cluster labeling based on RFM profiles
    cluster_means = rfm.groupby('Cluster').mean()
    print("Cluster Means before labeling:\n", cluster_means)
    
    labels = {}
    remaining_clusters = list(range(4))
    
    # Find High-Value: cluster with highest Monetary
    high_value_c = cluster_means['Monetary'].idxmax()
    labels[high_value_c] = "High-Value"
    remaining_clusters.remove(high_value_c)
    
    # Find At-Risk: out of remaining, the one with highest Recency
    at_risk_c = cluster_means.loc[remaining_clusters, 'Recency'].idxmax()
    labels[at_risk_c] = "At-Risk"
    remaining_clusters.remove(at_risk_c)
    
    # Find Occasional vs Regular:
    c1, c2 = remaining_clusters
    if cluster_means.loc[c1, 'Monetary'] < cluster_means.loc[c2, 'Monetary']:
        labels[c1] = "Occasional"
        labels[c2] = "Regular"
    else:
        labels[c1] = "Regular"
        labels[c2] = "Occasional"
        
    print("Mapped Cluster Labels:", labels)
    
    # Map cluster names to rfm dataframe
    rfm['Segment'] = rfm['Cluster'].map(labels)
    
    # Save model, scaler, and labeled cluster profiles
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # Also save the mapping dictionary
    with open('cluster_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
        
    print("Saved Clustering Models successfully.")
    
    print("Step 5: Building Collaborative Filtering Recommendations...")
    # Filter to top products to keep memory footprint small and recommendations high-quality
    product_counts = df['Description'].value_counts()
    valid_products = product_counts[product_counts >= 10].index
    df_filtered = df[df['Description'].isin(valid_products)]
    
    # Customer-Product pivot matrix
    customer_product = df_filtered.groupby(['CustomerID', 'Description']).size().unstack(fill_value=0)
    customer_product = customer_product.clip(upper=1) # Binary interaction
    
    print(f"Pivot matrix shape: {customer_product.shape}")
    
    # Compute Item-Item Cosine Similarity
    col_norms = np.linalg.norm(customer_product.values, axis=0)
    col_norms[col_norms == 0] = 1e-9
    norm_matrix = customer_product.values / col_norms
    
    similarity_matrix = np.dot(norm_matrix.T, norm_matrix)
    
    print("Generating top recommendations dictionary...")
    products = list(customer_product.columns)
    recommendations_dict = {}
    
    for idx, product in enumerate(products):
        sim_scores = similarity_matrix[idx]
        sorted_indices = np.argsort(sim_scores)[::-1]
        
        top_matches = []
        for match_idx in sorted_indices:
            match_prod = products[match_idx]
            if match_prod == product:
                continue
            score = float(sim_scores[match_idx])
            if score > 0:
                top_matches.append((match_prod, score))
            if len(top_matches) >= 10:
                break
        
        recommendations_dict[product] = top_matches
        
    with open('product_recommendations.pkl', 'wb') as f:
        pickle.dump(recommendations_dict, f)
        
    print("Saved Recommendations successfully.")
    
    # Save a small subset of the cleaned dataset for dashboard visualization
    df_clean_dash = df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country', 'TotalSpend']].copy()
    df_clean_dash.to_csv('cleaned_transactions.csv', index=False)
    rfm.reset_index().to_csv('customer_rfm.csv', index=False)
    
    print("Pipeline Complete! All assets generated.")

if __name__ == '__main__':
    run_pipeline()
