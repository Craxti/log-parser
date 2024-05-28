from collections import Counter
from algo.AGGLOMERATIVE.clustering import LogCluster


class LogClassifier:
    def __init__(self):
        pass

    def classify(self, clusters):
        cluster_categories = {}

        for cluster_id, cluster_data in clusters.items():
            log_group = cluster_data["logs"]
            cluster_category_counter = Counter()

            for log in log_group:
                category = LogCluster.determine_category(log)[1]
                cluster_category_counter[category] += 1

            if cluster_category_counter:
                most_common_category, most_common_count = cluster_category_counter.most_common(1)[0]
            else:
                most_common_category = 'Unknown'
                most_common_count = 0

            cluster_info = {
                "category": most_common_category,
                "log_count": len(log_group),
                "log_distribution": dict(cluster_category_counter),
                "most_common_category_count": most_common_count,
                "logs": log_group,
                "cluster_id": cluster_id
            }

            cluster_categories[cluster_id] = cluster_info

        return cluster_categories

    @staticmethod
    def determine_best_category(log_group):
        category_counter = Counter()

        for log in log_group:
            category = LogCluster.determine_category(log)[1]
            category_counter[category] += 1

        most_common_category, frequency = category_counter.most_common(1)[0]
        probability = frequency / len(log_group)
        return most_common_category, probability

    def pretty_print_classification(self, classification):
        for cluster_id, cluster_info in classification.items():
            print(f"Cluster {cluster_id}:")
            print(f"  Category: {cluster_info['category']}")
            print(f"  Total Logs: {cluster_info['log_count']}")
            print(f"  Most Common Category Count: {cluster_info['most_common_category_count']}")
            print("  Log Distribution:")
            for category, count in cluster_info['log_distribution'].items():
                print(f"    {category}: {count}")
            print("  Logs:")
            for log in cluster_info["logs"]:
                print(f"    {log}")
            print()
