import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from utils.proccess_log_file import preprocess_logs
from utils.analyze_log_text import analyze_log_text

def find_anomalies_svm(log_file, nu=0.1, kernel="rbf", gamma="scale"):
    """
   Function for searching for anomalies in logs using One-Class SVM.

    Args:
    - log_file (str): Path to the log file.
    - nu (float): nu parameter for One-Class SVM. Default is 0.1.
    - kernel (str): Kernel for One-Class SVM. Default is "rbf".
    - gamma (str): gamma parameter for One-Class SVM. Default is "scale".

    Returns:
    - tuple: A tuple containing three elements:
             1. List of anomalous log lines.
             2. A list of all log lines with labels indicating whether the line is anomalous or not.
             3. List of all log lines.
    """
    with open(log_file, 'r') as file:
        logs = file.readlines()

    preprocessed_logs = preprocess_logs(logs)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_logs)

    oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    oc_svm.fit(X)

    anomalies_indices = [i for i, decision in enumerate(oc_svm.decision_function(X)) if decision < 0]
    anomalies = [logs[i].strip() for i in anomalies_indices]

    all_logs_with_labels = [(log.strip(), 'Anomalous' if i in anomalies_indices else 'Normal') for i, log in enumerate(logs)]

    return anomalies, all_logs_with_labels, logs

def classify_anomalies(anomalies, keyword_dict):
    classified_anomalies = {status: [] for status in keyword_dict.keys()}
    classified_anomalies["Other"] = []

    for anomaly in anomalies:
        category = analyze_log_text(anomaly, keyword_dict)
        classified_anomalies[category].append(anomaly)

    return classified_anomalies
