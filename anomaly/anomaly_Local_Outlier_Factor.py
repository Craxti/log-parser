from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import re
from utils.proccess_log_file import preprocess_logs
from utils.analyze_log_text import analyze_log_text


def find_anomalies_lof(log_file, contamination=0.1, n_neighbors=20, algorithm='auto',
                       vectorizer_params=None, lof_params=None):
    """
    Function for searching for anomalies in logs using Local Outlier Factor.

    Args:
    - log_file (str): Path to the log file.
    - contamination (float): Level of contamination (proportion of anomalies). Default is 0.1.
    - n_neighbors (int): Number of neighbors for LOF. Default is 20.
    - algorithm (str): Algorithm for LOF. Default is "auto".
    - vectorizer_params (dict): Parameters for TfidfVectorizer. Default is None.
    - lof_params (dict): Parameters for Local Outlier Factor. Default is None.

    Returns:
    - tuple: A tuple containing three elements:
             1. List of anomalous log lines.
             2. A list of all log lines with labels indicating whether the line is anomalous or not.
             3. List of all log lines.
    """
    try:
        with open(log_file, 'r') as file:
            logs = file.readlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл {log_file} не найден.")
        return [], [], []

    preprocessed_logs = preprocess_logs(logs)

    vectorizer_params = vectorizer_params or {}
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X = vectorizer.fit_transform(preprocessed_logs)

    lof_params = lof_params or {}
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=n_neighbors, algorithm=algorithm, **lof_params)
    predictions = lof.fit_predict(X)

    anomalies_indices = [i for i, prediction in enumerate(predictions) if prediction == -1]
    anomalies = [logs[i].strip() for i in anomalies_indices]

    all_logs_with_labels = [(log.strip(), 'Anomalous' if i in anomalies_indices else 'Normal') for i, log in
                            enumerate(logs)]

    return anomalies, all_logs_with_labels, logs


def generate_keyword_dict(anomalies, top_n=10):
    """
    Generates a dictionary of keywords for anomaly classification.

    Args:
    - anomalies (list): List of anomalous log lines.
    - top_n (int): Number of top keywords for each category.

    Returns:
    - dict: Dictionary of keywords for classifying anomalies.
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(anomalies)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    keywords = []
    for i in range(tfidf_matrix.shape[0]):
        tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
        keywords.extend([keyword for keyword, score in sorted_keywords])

    return keywords


def classify_anomalies(anomalies, keyword_dict):
    classified_anomalies = {status: [] for status in keyword_dict.keys()}
    classified_anomalies["Other"] = []

    for anomaly in anomalies:
        category = analyze_log_text(anomaly, keyword_dict)
        if category in classified_anomalies:
            classified_anomalies[category].append(anomaly)
        else:
            classified_anomalies["Other"].append(anomaly)

    return classified_anomalies