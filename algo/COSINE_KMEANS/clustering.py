from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import joblib


class LogCluster:
    def __init__(self, max_clusters=10, model_file="cluster_model_cosine_kmeans.pkl"):
        self.max_clusters = max_clusters
        self.model_file = model_file
        self.vectorizer = TfidfVectorizer()
        self.cluster_model = None


    def fit(self, logs):
        """
        Clusters the logs using K-means and cosine similarity.

        Args:
        - logs (list): List of log lines.

        Returns:
        - dict: Dictionary, where keys are cluster numbers, values are a list of logs in each cluster.
        """
        X = self.vectorizer.fit_transform(logs)
        cosine_distances_matrix = cosine_similarity(X)
        optimal_num_clusters = self.determine_optimal_clusters(cosine_distances_matrix)
        self.cluster_model = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10,
                                    random_state=0)
        self.cluster_model.fit(X)
        joblib.dump(self.cluster_model, self.model_file)
        clusters = self._collect_clusters(logs)
        return clusters


    def load_model(self):
        """Loads the clustering model from a file."""
        self.cluster_model = joblib.load(self.model_file)


    def determine_optimal_clusters(self, distances_matrix):
        """
        Determines the optimal number of clusters using the elbow method.

        Args:
        - distances_matrix (ndarray): Matrix of pairwise distances between vectors.

        Returns:
        - int: Optimal number of clusters.
        """
        wcss = []
        for i in range(1, self.max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(distances_matrix)
            wcss.append(kmeans.inertia_)
        optimal_num_clusters = np.argmin(np.diff(np.diff(wcss))) + 1
        return optimal_num_clusters


    def _collect_clusters(self, logs):
        """
        Collects logs into clusters based on the fitted model.

        Args:
        - logs (list): List of log lines.

        Returns:
        - dict: Dictionary, where keys are cluster numbers, values are a list of logs in each cluster.
        """
        clusters = {}
        for i, label in enumerate(self.cluster_model.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(logs[i])
        return clusters


    def pretty_print_clusters(self, clusters):
        """
        Prints clusters in a readable format.

        Args:
        - clusters (dict): Dictionary of clusters.
        """
        for cluster_id, logs in clusters.items():
            print(f"Cluster {cluster_id}:")
            for log in logs:
                print(f"  {log}")
            print()
