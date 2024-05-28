# clustering.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


class LogClusterKMeans:
    def __init__(self, model_file="cluster_model.pkl"):
        self.model_file = model_file
        self.vectorizer = TfidfVectorizer()
        self.kmeans = None


    def determine_optimal_clusters(self, X):
        n_samples = X.shape[0]
        if n_samples <= 1:
            return 1
        max_clusters = min(n_samples - 1, 10)  # Maximum number of clusters - 10
        scores = []
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X)
            score = silhouette_score(X, kmeans.labels_)
            scores.append(score)
        return scores.index(max(scores)) + 2  # Return the optimal number of clusters


    def fit(self, logs):
        X = self.vectorizer.fit_transform(logs)
        n_clusters = self.determine_optimal_clusters(X)
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.kmeans.fit(X)
        self.save_model()
        return self._collect_clusters(logs)


    def _collect_clusters(self, logs):
        clusters = {}
        for i, label in enumerate(self.kmeans.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(logs[i])
        return clusters


    def save_model(self):
        joblib.dump(self.kmeans, self.model_file)


    def load_model(self):
        self.kmeans = joblib.load(self.model_file)


    def predict(self, logs):
        X = self.vectorizer.transform(logs)
        return self.kmeans.predict(X)


    def pretty_print_clusters(self, clusters):
        for cluster_id, logs in clusters.items():
            print(f"Cluster {cluster_id}:")
            for log in logs:
                print(f"  {log}")
            print()
