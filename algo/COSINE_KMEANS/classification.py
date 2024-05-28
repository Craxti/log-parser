import re
import math
from collections import Counter

class LogClassifier:
    def __init__(self):
        self.timestamp_regexes = [
            r'\b\d{1,2}:\d{2}:\d{2}\b',
            r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b',
            r'\b\d{2}:\d{2}:\d{2}\.\d{3}\b',
            r'\b\d{2}:\d{2}:\d{2},\d{2}.\d{2}.\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\b',
            r'\b\d{2}.\d{2}.\d{4} \d{2}:\d{2}:\d{2}\b',
            r'\b\d{2}:\d{2}\b',
            r'\b\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\b'
        ]


    def classify(self, logs, clusters):
        """
        Classifies logs based on clusters obtained using the k-means method with cosine distance.

        Args:
        - logs (list): List of log lines.
        - clusters (dict): A dictionary containing log clusters.

        Returns:
        - dict: Dictionary with classification details for each cluster.
        """
        cluster_categories = {}

        for cluster_id, log_group in clusters.items():
            cluster_category_counter = Counter()
            total_weight = 0

            for log in log_group:
                timestamps = self._find_timestamps(log)
                if timestamps:
                    weight = self._calculate_weight(log)
                    total_weight += weight
                    cluster_category_counter[log] += weight
                else:
                    total_weight *= math.log(cluster_category_counter[log] + 2)
                    cluster_category_counter[log] += 1

            for log, count in cluster_category_counter.items():
                cluster_category_counter[log] = count / total_weight

            most_common_log = cluster_category_counter.most_common(1)[0][0]

            cluster_categories[cluster_id] = {
                "most_common_log": most_common_log,
                "num_logs": len(log_group),
                "total_logs": len(logs),
                "percentage_of_total_logs": len(log_group) / len(logs) * 100
            }

        return cluster_categories


    def _find_timestamps(self, log):
        """Finds timestamps in a log line using predefined regex patterns."""
        for regex in self.timestamp_regexes:
            timestamps = re.findall(regex, log)
            if timestamps:
                return timestamps
        return None


    def _calculate_weight(self, log):
        """Calculates the weight of a log line based on its unique characters."""
        import string
        return len(set(log)) - len(set(log).intersection(set(string.whitespace)))


    def pretty_print_classification(self, classification):
        """
        Prints classification results in a readable format.

        Args:
        - classification (dict): Dictionary with classification details for each cluster.
        """
        for cluster_id, cluster_info in classification.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Most Common Log: {cluster_info['most_common_log']}")
            print(f"  Number of Logs: {cluster_info['num_logs']}")
            print(f"  Percentage of Total Logs: {cluster_info['percentage_of_total_logs']:.2f}%")
            print()
