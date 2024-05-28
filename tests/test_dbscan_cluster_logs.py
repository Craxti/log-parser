from algo.DBSCAN.clustering import cluster_logs_dbscan as dbase_cluster_logs
from algo.KMEANS.classification import classify_logs as dbase_classify_logs
from utils.parser_log import parse_log, remove_stopwords

def test_dbscan_cluster_logs(stopwords=None):
    log_file = "log.txt"

    # Парсинг логов
    logs = parse_log(log_file)

    # Удаление стоп-слов (необязательно)
    logs_cleaned = remove_stopwords(logs, stopwords)
    clusters = dbase_cluster_logs(logs_cleaned)

    # Классификация логов по алгоритму KMEANS
    log_category = dbase_classify_logs(logs_cleaned)

    assert len(clusters) > 0, "No clusters were generated"

test_dbscan_cluster_logs()
