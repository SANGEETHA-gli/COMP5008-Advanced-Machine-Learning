import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


data_path = r"C:\Users\dsang\source\repos\Machine_Learning_Assignment\Dataset\preprocessed_winequality.csv"
data = pd.read_csv(data_path)


X = data.drop(columns=["quality", "type"])
labels_true = data["quality"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)


kmeans = KMeans(n_clusters=6, random_state=42, init="k-means++", n_init=10, max_iter=300)
kmeans_labels = kmeans.fit_predict(X_pca)


gmm = GaussianMixture(n_components=6, random_state=42, covariance_type="full", max_iter=300, n_init=5)
gmm_labels = gmm.fit_predict(X_pca)


kmeans_silhouette = silhouette_score(X_pca, kmeans_labels)
kmeans_ch_score = calinski_harabasz_score(X_pca, kmeans_labels)
kmeans_db_score = davies_bouldin_score(X_pca, kmeans_labels)
print("KMeans Clustering:")
print(f"Silhouette Score: {kmeans_silhouette:.4f}")
print(f"Calinski-Harabasz Index: {kmeans_ch_score:.4f}")
print(f"Davies-Bouldin Index: {kmeans_db_score:.4f}")


gmm_silhouette = silhouette_score(X_pca, gmm_labels)
gmm_ch_score = calinski_harabasz_score(X_pca, gmm_labels)
gmm_db_score = davies_bouldin_score(X_pca, gmm_labels)
print("GMM Clustering:")
print(f"Silhouette Score: {gmm_silhouette:.4f}")
print(f"Calinski-Harabasz Index: {gmm_ch_score:.4f}")
print(f"Davies-Bouldin Index: {gmm_db_score:.4f}")


def plot_clustering(X, labels, title):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="viridis", legend=None)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

plot_clustering(X_pca, kmeans_labels, "KMeans Clustering")
plot_clustering(X_pca, gmm_labels, "GMM Clustering")
