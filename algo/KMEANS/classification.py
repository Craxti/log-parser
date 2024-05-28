# classification.py
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import numpy as np


class LogClassifier:
    def __init__(self, model_file='log_classifier.pkl'):
        self.model_file = model_file
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        self.model = None


    def determine_category(self, log):
        if "ERROR" in log:
            return "ERROR"
        elif "INFO" in log:
            return "INFO"
        elif "DEBUG" in log:
            return "DEBUG"
        elif "WARNING" in log:
            return "WARNING"
        else:
            return "Unknown"


    def train_classifier(self, logs, categories):
        self.pipeline.fit(logs, categories)
        joblib.dump(self.pipeline, self.model_file)
        self.model = self.pipeline


    def load_model(self):
        self.model = joblib.load(self.model_file)


    def classify(self, logs, clusters):
        training_logs = []
        training_categories = []
        for cluster_id, log_group in clusters.items():
            for log in log_group:
                category = self.determine_category(log)
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
                most_common_category = self.determine_category(most_common_log)
                probability = 1.0

            cluster_categories[cluster_id] = (most_common_category, probability)

        return cluster_categories


    def classify_single_log(self, log):
        if not self.model:
            self.load_model()
        prediction = self.model.predict([log])[0]
        probability = self.model.predict_proba([log]).max()
        return prediction, probability


    def pretty_print_classification(self, classification):
        for cluster_id, (category, probability) in classification.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Category: {category}")
            print(f"  Probability: {probability:.2f}")
            print()
