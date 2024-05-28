from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from algo.AGGLOMERATIVE.clustering import LogCluster


class LogClassifier:
    def __init__(self, model_file='log_classifier.pkl'):
        self.model_file = model_file
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        self.model = None


    def train_classifier(self, logs, categories):
        """
        Trains a Naive Bayes classifier.

        Args:
        - logs (list): List of log lines.
        - categories (list): List of categories.

        Returns:
        - Pipeline: Trained classification model.
        """
        self.pipeline.fit(logs, categories)
        joblib.dump(self.pipeline, self.model_file)
        self.model = self.pipeline


    def load_model(self):
        """Loads the classification model from a file."""
        self.model = joblib.load(self.model_file)


    def classify(self, logs, clusters):
        """
        Classifies logs based on clusters obtained using DBSCAN.

        Args:
        - logs (list): List of log lines.
        - clusters (dict): A dictionary containing log clusters.

        Returns:
        - dict: Dictionary with log categories and their probabilities for each cluster.
        """
        training_logs = []
        training_categories = []
        for cluster_id, log_group in clusters.items():
            for log in log_group:
                category = LogCluster.determine_category(log)
                training_logs.append(log)
                training_categories.append(category)

        self.train_classifier(training_logs, training_categories)

        cluster_categories = {}
        for cluster_id, log_group in clusters.items():
            if cluster_id == -1:
                continue

            predictions = self.model.predict(log_group)
            probabilities = self.model.predict_proba(log_group)
            cluster_counter = Counter(predictions)

            if len(cluster_counter) > 1:
                most_common_category = cluster_counter.most_common(1)[0][0]
                probability = np.max(probabilities, axis=1).mean()
            else:
                most_common_log = Counter(log_group).most_common(1)[0][0]
                most_common_category = LogCluster.determine_category(most_common_log)
                probability = 1.0

            cluster_categories[cluster_id] = (most_common_category, probability)

        return cluster_categories


    def classify_single_log(self, log):
        """
        Classifies a single log.

        Args:
        - log (str): Log string.

        Returns:
        - str: Log category.
        - float: Probability of the predicted category.
        """
        if not self.model:
            self.load_model()

        prediction = self.model.predict([log])[0]
        probability = self.model.predict_proba([log]).max()
        return prediction, probability


    def pretty_print_classification(self, classification):
        """
        Prints classification results in a readable format.

        Args:
        - classification (dict): Dictionary with classification details for each cluster.
        """
        for cluster_id, (category, probability) in classification.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Category: {category}")
            print(f"  Probability: {probability:.2f}")
            print()
