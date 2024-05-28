from algo.AGGLOMERATIVE.clustering import cluster_logs_agglomerative as linlage_ward_cluster_logs
from algo.AGGLOMERATIVE.classification import classify_logs_agglomerative as linlage_ward_classify_logs
from utils.parser_log import parse_log, remove_stopwords

def test_dbscan_cluster_logs(stopwords=None):
    log_file = "log.txt"

    # Парсинг логов
    logs = parse_log(log_file)

    # Удаление стоп-слов (необязательно)
    logs_cleaned = remove_stopwords(logs, stopwords)
    clusters = linlage_ward_cluster_logs(logs_cleaned)

    # Классификация логов по алгоритму KMEANS
    log_category = linlage_ward_classify_logs(logs_cleaned)

    assert len(clusters) > 0, "No clusters were generated"

test_dbscan_cluster_logs()
