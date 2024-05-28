import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from utils.proccess_log_file import preprocess_logs


def find_anomalies(log_file, contamination=0.1, random_state=42):
    """
    Функция для поиска аномалий в логах с использованием Isolation Forest.

    Args:
    - log_file (str): Путь к файлу с логами.
    - contamination (float): Уровень контаминации (доля аномалий). По умолчанию 0.1.
    - random_state (int): Семя для генерации случайных чисел. По умолчанию None.

    Returns:
    - tuple: Кортеж, содержащий три элемента:
             1. Список аномальных строк логов.
             2. Список всех строк логов с метками о том, является ли строка аномальной или нет.
             3. Список всех строк логов.
    """
    # Читаем содержимое файла логов
    try:
        with open(log_file, 'r') as file:
            logs = file.readlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл {log_file} не найден.")
        return [], [], []

    # Предобрабатываем логи
    preprocessed_logs = preprocess_logs(logs)

    # Преобразуем предобработанные логи в числовые векторы с помощью TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_logs)

    # Обучаем модель Isolation Forest на данных логов
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
    isolation_forest.fit(X)

    # Предсказываем аномалии
    anomalies_indices = [i for i, score in enumerate(isolation_forest.decision_function(X)) if score < 0]
    anomalies = [logs[i].strip() for i in anomalies_indices]

    # Создаем список всех строк логов с метками о том, является ли строка аномальной
    all_logs_with_labels = [(log.strip(), 'Anomalous' if i in anomalies_indices else 'Normal') for i, log in enumerate(logs)]

    return anomalies, all_logs_with_labels, logs


def classify_anomalies(anomalies, keyword_dict):
    """
    Function for classifying anomalous log lines.

    Args:
    - anomalies (list): List of anomalous log lines.

    Returns:
    - dict: Dictionary containing the classification of anomalous strings.
    """
    classified_anomalies = {'Error': [], 'Warning': [], 'Info': [], 'Timing': []}
    for anomaly in anomalies:
        if re.search(r'error|exception|fail', anomaly, re.IGNORECASE):
            classified_anomalies['Error'].append(anomaly)
        elif re.search(r'warning|alert', anomaly, re.IGNORECASE):
            classified_anomalies['Warning'].append(anomaly)
        elif re.search(r'info|information', anomaly, re.IGNORECASE):
            classified_anomalies['Info'].append(anomaly)
        elif re.search(r'timeout|timed out', anomaly, re.IGNORECASE):
            classified_anomalies['Timing'].append(anomaly)
        else:
            category = analyze_log_text(anomaly, keyword_dict)
            if category:
                classified_anomalies[category].append(anomaly)
            else:
                classified_anomalies.setdefault('Other', []).append(anomaly)
    return classified_anomalies


def analyze_log_text(log_text, keyword_dict):
    """
    Function for analyzing the text of an anomalous log line.

    Args:
    - log_text (str): Text of the anomalous log line.
    - keyword_dict (dict): A dictionary containing keywords for classification by status.

    Returns:
    - str: Category of the anomaly or None if classification failed.
    """
    for status, keywords in keyword_dict.items():
        for keyword in keywords:
            if keyword.lower() in log_text.lower():
                return status
    return None
