from utils.parser_log import parse_log, remove_stopwords
from algo.KMEANS.clustering import LogClusterKMeans
from algo.KMEANS.classification import LogClassifier
from algo.DBSCAN.clustering import LogCluster
from algo.DBSCAN.classification import LogClassifier
from algo.AGGLOMERATIVE.clustering import LogCluster
from algo.AGGLOMERATIVE.classification import LogClassifier
from algo.COSINE_KMEANS.clustering import LogCluster
from algo.COSINE_KMEANS.classification import LogClassifier
from algo.BRAIN.clustering import LogParser
from algo.AEL.clustering import LogParser
from anomaly.anomal_IsolationForest import find_anomalies, classify_anomalies
from anomaly.anomal_One_Class_SVM import find_anomalies_svm
from anomaly.anomaly_Local_Outlier_Factor import find_anomalies_lof
from algo.BRAIN.clustering import LogParser


def main(algorithm="KMEANS",anomaly=None, keyword_dict=None, stopwords=None, log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'):
    """
     Function for running the log processing algorithm.

     Args:
     - algorithm (str): Algorithm name (default "KMEANS").

     Returns:
     - None
    """
    import uuid
#    model_file = str(uuid.uuid4()) + algorithm
    # Path to the log file
    log_file = "log.txt"

    # Parsing logs
    logs = parse_log(log_file)

    # Remove stop words (optional)
    logs_cleaned = remove_stopwords(logs, stopwords)

    if algorithm == "KMEANS":
        log_cluster = LogClusterKMeans()
        clustered_logs = log_cluster.fit(logs)
        log_cluster.pretty_print_clusters(clustered_logs)

        log_classifier = LogClassifier()
        classified_logs = log_classifier.classify(logs, clustered_logs)
        log_classifier.pretty_print_classification(classified_logs)

    if algorithm == "DBSCAN":
        log_cluster = LogCluster()
        clustered_logs = log_cluster.fit(logs)
        log_cluster.pretty_print_clusters(clustered_logs)

        # Классификация логов
        log_classifier = LogClassifier()
        classified_logs = log_classifier.classify(logs, clustered_logs)
        log_classifier.pretty_print_classification(classified_logs)

    if algorithm == "AGGLOMERATIVE":
        log_cluster = LogCluster(n_clusters=3)
        clustered_logs = log_cluster.fit(logs)
        log_cluster.pretty_print_clusters(clustered_logs)

        log_classifier = LogClassifier()
        classified_logs = log_classifier.classify(clustered_logs)
        log_classifier.pretty_print_classification(classified_logs)

    if algorithm == "COSINE_KMEANS":
        log_cluster = LogCluster(max_clusters=5)
        clustered_logs = log_cluster.fit(logs)
        log_cluster.pretty_print_clusters(clustered_logs)

        log_classifier = LogClassifier()
        classified_logs = log_classifier.classify(logs, clustered_logs)
        log_classifier.pretty_print_classification(classified_logs)

    if algorithm == "BRAIN":
        parser = LogParser(log_format)
        parsed_logs, templates = parser.parse(logs)

        print("Parsed Logs:")
        for log in parsed_logs:
            print(log)

    if algorithm == "AEL":
        parser = LogParser(log_format)
        parsed_logs = parser.parse(logs)

        print("Parsed Logs:")
        for log in parsed_logs:
            print(log)

    if anomaly == "IsolationForest":
        anomalies, all_logs_with_labels, logs = find_anomalies(log_file)
        print("=== Extended result report ===")
        print(f"Total log lines: {len(logs)}")
        print(f"Number of anomalous lines: {len(anomalies)}")
        print(f"Proportion of anomalies: {len(anomalies) / len(logs):.2%}")

        if keyword_dict == None:
            keyword_dict = {
                "Error": ['high load', 'performance issue', 'failed', 'Authentication failed', 'error', 'invalid'],
                "Warning": ['disk space low', 'out of memory', 'invalid'],
                "Info": ['completed', 'ready', 'connection', 'PID', 'cache', 'starting', 'enabling'],
                "Timing": ['timed out', 'timeout', 'timed']
            }

        classified_anomalies = classify_anomalies(anomalies, keyword_dict)
        print("=== Classification of anomalous strings ===")
        for category, logs in classified_anomalies.items():
            print(f"{category}:")
            for log in logs:
                print(log)

    if anomaly == "OneClassSVM":
        anomalies, all_logs_with_labels, logs = find_anomalies_svm(log_file)

        print("=== Extended result report ===")
        print(f"Total log lines: {len(logs)}")
        print(f"Number of anomalous lines: {len(anomalies)}")
        print(f"Proportion of anomalies: {len(anomalies) / len(logs):.2%}")

        if keyword_dict == None:
            keyword_dict = {
                "Error": ['high load', 'performance issue', 'failed', 'Authentication failed', 'error', 'invalid'],
                "Warning": ['disk space low', 'out of memory', 'invalid'],
                "Info": ['completed', 'ready', 'connection', 'PID', 'cache', 'starting', 'enabling'],
                "Timing": ['timed out', 'timeout', 'timed']
            }
        classified_anomalies = classify_anomalies(anomalies, keyword_dict)
        print("=== Classification of anomalous strings ===")
        for category, logs in classified_anomalies.items():
            print(f"{category}:")
            for log in logs:
                print(log)

    if anomaly == "LocalOutlierFactor":
        anomalies, all_logs_with_labels, logs = find_anomalies_lof(log_file)

        print("=== Extended result report ===")
        print(f"Total log lines: {len(logs)}")
        print(f"Number of anomalous lines: {len(anomalies)}")
        print(f"Proportion of anomalies: {len(anomalies) / len(logs):.2%}")


        if keyword_dict == None:
            keyword_dict = {
                "Error": ['high load', 'performance issue', 'failed', 'Authentication failed', 'error', 'invalid'],
                "Warning": ['disk space low', 'out of memory', 'invalid'],
                "Info": ['completed', 'ready', 'connection', 'PID', 'cache', 'starting', 'enabling'],
                "Timing": ['timed out', 'timeout', 'timed']
            }
        classified_anomalies = classify_anomalies(anomalies, keyword_dict)
        print("=== Classification of anomalous strings ===")
        for category, logs in classified_anomalies.items():
            print(f"{category}:")
            for log in logs:
                print(log)

    if anomaly or algorithm == None or anomaly or algorithm not in ['KMEANS', 'AEL', 'BRAIN', 'COSINE_KMEANS',
                                                                    'AGGLOMERATIVE', 'DBSCAN', 'OneClassSVM',
                                                                    'LocalOutlierFactor', 'IsolationForest']:
        print("Invalid algorithm or anomaly name.")

if __name__ == "__main__":
    main(anomaly="IsolationForest")
