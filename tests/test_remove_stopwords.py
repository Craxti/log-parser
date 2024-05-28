from utils.parser_log import remove_stopwords

def test_remove_stopwords():
    logs = ["This is a test log", "Another test log"]
    stopwords = ["this", "is", "a", "test"]
    cleaned_logs = remove_stopwords(logs, stopwords)
    assert cleaned_logs == ["log", "Another log"], "Stopwords were not removed correctly"

test_remove_stopwords()
