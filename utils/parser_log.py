import re
from datetime import datetime


def parse_log(log_file):
    """
    Функция для парсинга логов из файла.

    Args:
    - log_file (str): Путь к файлу с логами.

    Returns:
    - list: Список строк логов.
    """
    with open(log_file, 'r') as file:
        logs = file.readlines()
    return logs


def normalize_log(log):
    """
    Функция для нормализации логов.

    Args:
    - log (str): Строка лога.

    Returns:
    - str: Нормализованная строка лога.
    """
    # Пример шаблонов для различных форматов логов
    patterns = [
        r'(?P<timestamp>\w{3}\s+\d+\s+\d+:\d+:\d+)\s+(?P<message>.+)',  # формат "Jun 14 15:16:01"
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\s*(?P<message>.+)',  # формат "2016-09-28 04:30:30, Info"
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\s*(?P<message>.+)',  # формат "2024-05-21 10:00:00,INFO"
    ]

    for pattern in patterns:
        match = re.match(pattern, log)
        if match:
            timestamp_str = match.group('timestamp')
            message = match.group('message')

            # Преобразуем временную метку в единый формат
            try:
                if re.match(r'\w{3}\s+\d+\s+\d+:\d+:\d+', timestamp_str):
                    timestamp = datetime.strptime(timestamp_str, '%b %d %H:%M:%S')
                    timestamp = timestamp.replace(year=datetime.now().year)
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue

            normalized_log = f"{timestamp_str}, {message}"
            return normalized_log

    return log  # Вернуть оригинальный лог, если не удалось нормализовать


def remove_stopwords(logs, stopwords=None):
    """
    Функция для удаления стоп-слов из логов.

    Args:
    - logs (list): Список строк логов.
    - stopwords (list): Список стоп-слов. Если не указан, будет использован список по умолчанию.

    Returns:
    - list: Список строк логов без стоп-слов.
    """
    if stopwords is None:
        stopwords = []
    cleaned_logs = []
    for log in logs:
        cleaned_words = []
        for word in log.split():
            if word.lower() not in stopwords:
                cleaned_words.append(word)
        cleaned_log = ' '.join(cleaned_words)
        cleaned_logs.append(cleaned_log)
    return cleaned_logs



def parse_and_normalize_logs(log_file, stopwords=None):
    """
    Функция для парсинга и нормализации логов из файла с опциональным удалением стоп-слов.

    Args:
    - log_file (str): Путь к файлу с логами.
    - stopwords (list): Список стоп-слов для удаления (необязательно).

    Returns:
    - list: Список нормализованных строк логов без стоп-слов.
    """
    logs = parse_log(log_file)
    normalized_logs = [normalize_log(log) for log in logs]
    cleaned_logs = remove_stopwords(normalized_logs, stopwords)
    return cleaned_logs
