from algo.COSINE_KMEANS.clustering import cosine_kmeans_cluster_logs
from algo.COSINE_KMEANS.classification import classify_logs_cosine_kmeans
from utils.parser_log import parse_log, remove_stopwords

def test_dbscan_cluster_logs(stopwords=None):
    log_file = "log.txt"

    # Парсинг логов
    logs = parse_log(log_file)

    # Удаление стоп-слов (необязательно)
    logs_cleaned = remove_stopwords(logs, stopwords)
    clusters = cosine_kmeans_cluster_logs(logs_cleaned)

    # Классификация логов по алгоритму KMEANS
    log_category = classify_logs_cosine_kmeans(logs_cleaned)

    assert len(clusters) > 0, "No clusters were generated"

test_dbscan_cluster_logs()
