import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_logs(logs):
    """
    Функция для предобработки логов: удаление стоп-слов и приведение к нижнему регистру.

    Args:
    - logs (list): Список строк логов.

    Returns:
    - list: Список предобработанных строк логов.
    """
    # Получаем список стоп-слов из библиотеки NLTK
    stop_words = set(stopwords.words("english"))

    # Инициализируем список для хранения предобработанных логов
    preprocessed_logs = []

    # Проходимся по каждой строке лога
    for log in logs:
        # Приводим строку к нижнему регистру
        log = log.lower()

        # Токенизируем строку на слова
        words = word_tokenize(log)

        # Удаляем стоп-слова и пунктуацию из токенизированной строки
        filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]

        # Объединяем токены обратно в строку
        preprocessed_log = ' '.join(filtered_words)

        # Добавляем предобработанную строку в список
        preprocessed_logs.append(preprocessed_log)

    return preprocessed_logs