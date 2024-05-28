from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import joblib

class LogCluster:
    def __init__(self, n_clusters=5, merge_threshold=0.8, model_file="cluster_model_agglomerative.pkl"):
        self.n_clusters = n_clusters
        self.merge_threshold = merge_threshold
        self.model_file = model_file
        self.vectorizer = TfidfVectorizer()
        self.cluster_model = None


    def fit(self, logs):
        X = self.vectorizer.fit_transform(logs)
        self.cluster_model = AgglomerativeClustering(linkage='ward', n_clusters=self.n_clusters)
        self.cluster_model.fit(X.toarray())
        clusters = self._create_clusters(logs)
        clusters = self._merge_similar_clusters(clusters)
        best_clusters = self._extract_best_clusters(clusters)
        joblib.dump(self.cluster_model, self.model_file)
        return best_clusters


    def load_model(self):
        self.cluster_model = joblib.load(self.model_file)


    def _create_clusters(self, logs):
        clusters = {}
        for i, label in enumerate(self.cluster_model.labels_):
            if label not in clusters:
                clusters[label] = {"logs": [], "categories": []}
            log_with_category, category = self.determine_category(logs[i])
            clusters[label]["logs"].append(log_with_category)
            clusters[label]["categories"].append(category)
        return clusters


    def _merge_similar_clusters(self, clusters):
        cluster_keys = list(clusters.keys())
        merged_clusters = {}
        cluster_texts = [' '.join(clusters[key]["logs"]) for key in cluster_keys]
        X_clusters = self.vectorizer.fit_transform(cluster_texts)
        similarities = cosine_similarity(X_clusters)
        merged = set()

        for i, key1 in enumerate(cluster_keys):
            if key1 in merged:
                continue
            merged_clusters[key1] = clusters[key1]
            for j, key2 in enumerate(cluster_keys):
                if i != j and key2 not in merged:
                    if similarities[i][j] >= self.merge_threshold:
                        merged_clusters[key1]["logs"].extend(clusters[key2]["logs"])
                        merged_clusters[key1]["categories"].extend(clusters[key2]["categories"])
                        merged.add(key2)
        return merged_clusters


    def _extract_best_clusters(self, clusters):
        best_clusters = {}
        for label, cluster_data in clusters.items():
            category_counter = Counter(cluster_data["categories"])
            most_common_category = next(
                (category for category, count in category_counter.most_common() if category != 'Unknown'), 'Unknown')
            best_clusters[label] = {
                "logs": cluster_data["logs"],
                "most_common_category": most_common_category,
                "all_categories": category_counter
            }
        return best_clusters


    @staticmethod
    def determine_category(log):
        keywords = {
            "ERROR": ["fail", "exception", "critical", "unavailable", "unable", "invalid", "error", "crash"],
            "INFO": ["started", "completed", "running", "connected", "scanning", "initializing", "api", "registering",
                     "setting", "info", "ACPI", "using", "cache", "checking", "connection", "startup", "succeeded"],
            "DEBUG": ["debugging", "variable", "trace", "step", "debug", "using", "checking", "opened", "closed"],
            "WARNING": ["high", "low", "degraded", "exceeded", "threshold", "warning"]
        }

        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                if keyword in log.lower():
                    return f"{log} [{category}]", category
        return f"{log} [Unknown]", "Unknown"


    def pretty_print_clusters(self, clusters):
        for cluster_id, cluster_data in clusters.items():
            print(f"Cluster {cluster_id} (Most Common Category: {cluster_data['most_common_category']}):")
            print(f"  Total Logs: {len(cluster_data['logs'])}")
            for category, count in cluster_data['all_categories'].items():
                print(f"    {category}: {count} logs")
            print("  Logs:")
            for log in cluster_data["logs"]:
                print(f"    {log}")
            print()
