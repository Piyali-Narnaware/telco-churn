import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEGMENT_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges",
                     "ServiceCount", "HasDependents"]

SEGMENT_NAMES = {
    0: "Price-Sensitive",
    1: "High-Value Loyal",
    2: "New / Short-Tenure",
    3: "Low Engagement",
    4: "Premium Power Users",
}

def prepare_segmentation_data(df):
    seg_df = df[SEGMENT_FEATURES].copy()
    seg_df.fillna(0, inplace=True)
    scaler = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_df)
    return seg_scaled, scaler

def find_optimal_k(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

def run_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(data)
    return components, pca

def label_segments(cluster_labels):
    return [SEGMENT_NAMES.get(label, f"Segment {label}") for label in cluster_labels]
