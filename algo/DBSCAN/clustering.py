from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import numpy as np


class LogCluster:
    def __init__(self, model_file="cluster_model_dbscan.pkl", eps_candidates=[0.1, 0.3, 0.5, 0.7, 1.0], min_samples=5):
        self.model_file = model_file
        self.eps_candidates = eps_candidates
        self.min_samples = min_samples
        self.vectorizer = TfidfVectorizer()
        self.cluster_model = None


    def fit(self, logs):
        """
        Clusters the logs using DBSCAN algorithm.

        Args:
        - logs (list): List of log lines.

        Returns:
        - dict: Dictionary, where keys are cluster labels, values are a list of logs in each cluster.
        """
        X = self.vectorizer.fit_transform(logs)
        distances = euclidean_distances(X)
        optimal_eps = self._find_optimal_eps(distances)
        self.cluster_model = DBSCAN(eps=optimal_eps, min_samples=self.min_samples, metric='precomputed')
        self.cluster_model.fit(distances)
        joblib.dump(self.cluster_model, self.model_file)
        clusters = self._collect_clusters(logs)
        return clusters


    def load_model(self):
        """Loads the clustering model from a file."""
        self.cluster_model = joblib.load(self.model_file)


    def _find_optimal_eps(self, distances):
        """
        Determines the optimal epsilon value for DBSCAN.

        Args:
        - distances (ndarray): Matrix of pairwise distances between vectors.

        Returns:
        - float: Optimal epsilon value.
        """
        for eps_candidate in self.eps_candidates:
            dbscan = DBSCAN(eps=eps_candidate, min_samples=self.min_samples, metric='precomputed')
            dbscan.fit(distances)
            if len(set(dbscan.labels_)) >= 2:
                return eps_candidate
        return self.eps_candidates[-1]


    def _collect_clusters(self, logs):
        """
        Collects logs into clusters based on the fitted model.

        Args:
        - logs (list): List of log lines.

        Returns:
        - dict: Dictionary, where keys are cluster labels, values are a list of logs in each cluster.
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
