from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException
import urllib.parse
import sqlite3
import os
from typing import List, Tuple
import time
from functools import wraps

# Импорты для векторной базы данных (если они нужны позже)
# from sympy.printing.pytorch import torch
# from transformers import AutoTokenizer, T5EncoderModel
# import chromadb
# import numpy as np
# from tqdm import tqdm


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Функция {func.__name__} начала выполняться.")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Функция {func.__name__} завершила выполнение. Время выполнения: {elapsed_time:.4f} сек.")
        return result
    return wrapper


def create_content_db(db_path: str = "data/wiki_content.db"):
    """
    Создает SQLite базу данных и таблицу для хранения содержимого страниц.

    Args:
        db_path (str): Путь к файлу базы данных.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wiki_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"База данных содержимого создана: {db_path}")


def save_page_content_to_db(url: str, content: str, db_path: str = "data/wiki_content.db"):
    """
    Сохраняет содержимое одной страницы в базу данных.

    Args:
        url (str): URL страницы.
        content (str): Содержимое страницы.
        db_path (str): Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO wiki_pages (url, content) VALUES (?, ?)",
            (url, content)
        )
        conn.commit()
        print(f"Содержимое страницы сохранено в БД: {url}")
    except sqlite3.Error as e:
        print(f"Ошибка SQLite при сохранении {url}: {e}")
    finally:
        conn.close()


def get_all_content_from_db(db_path: str = "data/wiki_content.db") -> List[Tuple[int, str, str]]:
    """
    Извлекает все содержимое из базы данных.

    Args:
        db_path (str): Путь к файлу базы данных.

    Returns:
        List[Tuple[int, str, str]]: Список кортежей (id, url, content).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, content FROM wiki_pages")
    results = cursor.fetchall()
    conn.close()
    return results


def get_content_by_url(url: str, db_path: str = "data/wiki_content.db") -> Tuple[int, str, str] | None:
    """
    Извлекает содержимое страницы по URL из базы данных.

    Args:
        url (str): URL страницы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Tuple[int, str, str] | None: Кортеж (id, url, content) или None, если не найдено.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, content FROM wiki_pages WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result


def parse_wiki():
    @log_execution
    def _save_all_links(urls: list):
        with open('data/links.txt', 'w', encoding='utf-8') as f:
            for item in urls:
                f.write(str(item) + '\n')

    @log_execution
    def _get_all_links() -> list:
        """
        Функция для получения ссылок на страницы вики
        :return:
        list: Массив ссылок на страницы вики
        """

        # ИСПРАВЛЕНО: Убраны лишние пробелы в URL
        base_page_url = "https://outer-wilds.fandom.com/ru/wiki/%D0%A1%D0%BB%D1%83%D0%B6%D0%B5%D0%B1%D0%BD%D0%B0%D1%8F:%D0%92%D1%81%D0%B5_%D1%81%D1%82%D1%80%D0%B0%D0%BD%D0%B8%D1%86%D1%8B"
        urls = []
        while True:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.get(base_page_url)
                    if response.status_code == 429 or "ratelimited" in response.text:
                        print(f"Rate limited for {base_page_url}, waiting...")
                        time.sleep(5)  # Ждём 5 секунд перед повтором
                        continue
                    response.raise_for_status()  # Проверяем, не было ли HTTP-ошибок
                    break
                except RequestException as e:
                    print(f"Request failed for {base_page_url} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        print(f"Failed to fetch {base_page_url} after {max_retries} attempts.")
                        return []

            soup = BeautifulSoup(response.content, "html.parser")
            # Получаем все ссылки на странице
            try:
                all_links = soup.find('div', class_='mw-allpages-body').find_all('a')
                for link in all_links:
                    # ИСПРАВЛЕНО: Убраны лишние пробелы в URL
                    full_url = urllib.parse.urljoin("https://outer-wilds.fandom.com/ru/wiki/", link.get('href'))
                    urls.append(full_url)
                    print(f"Found link: {full_url}")
            except AttributeError:
                print("Could not find page links, stopping.")
                return urls

            # Получаем ссылку на следующую страницу
            try:
                nav_links = soup.find('div', class_='mw-allpages-nav').find_all('a')
                if not nav_links:
                     print("No navigation links found, stopping.")
                     return urls
                # Предполагаем, что последняя ссылка - это "следующая страница"
                # Лучше проверить текст ссылки
                next_link = None
                for a in nav_links:
                    if "Следующая страница" in a.text or "Next page" in a.text or a.text.strip() == ">": # Проверка на русский, английский или символ
                        next_link = a
                        break

                if next_link:
                    # ИСПРАВЛЕНО: Убраны лишние пробелы в URL
                    next_page_href = next_link.get('href')
                    if next_page_href:
                        base_page_url = urllib.parse.urljoin("https://outer-wilds.fandom.com/ru/wiki/", next_page_href)
                        print(f"Next page URL: {base_page_url}")
                    else:
                        print("Next page href is None, stopping.")
                        return urls
                else:
                    print("No 'Next page' link found, stopping.")
                    return urls
            except AttributeError:
                print("Could not find next page link, stopping.")
                return urls


    def _get_data_of_pages(urls: list, start_url: int, max_urls: int, db_path: str) -> str:
        """
        Функция для получения информации со страниц и сохранения в БД
        :param urls: Массив ссылок на страницы
        :param start_url: Индекс начальной страницы
        :param max_urls: Максимальное количество страниц для обработки за вызов
        :param db_path: Путь к базе данных содержимого
        :return:
        str: URL последней обработанной страницы
        """

        for url in urls[start_url:start_url + max_urls]:
            print(f"Processing URL: {url}")
            content = _get_data_of_page(url)
            if content: # Если контент успешно получен
                save_page_content_to_db(url, content, db_path)
            else:
                print(f"Failed to get content for {url}, stopping.")
                return url # Возвращаем URL, на котором произошла ошибка
            time.sleep(1.0) # Сон между запросами

        # Возвращаем URL последней успешно обработанной страницы
        last_processed_url = urls[start_url + max_urls - 1] if len(urls) > start_url + max_urls - 1 else urls[-1]
        return last_processed_url


    def _get_data_of_page(url: str) -> str | None:
        """
        Функция для получения информации со страницы
        :param url: Ссылка на страницу
        :return:
        str | None: Текстовое содержимое страницы или None в случае ошибки
        """
        print(f"Fetching page content: {url}")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 429 or "ratelimited" in response.text:
                    print(f"Rate limited for {url}, waiting...")
                    time.sleep(5)  # Ждём 5 секунд перед повтором
                    continue
                response.raise_for_status()  # Проверяем, не было ли HTTP-ошибок
                break
            except RequestException as e:
                print(f"Request failed for {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"Failed to fetch {url} after {max_retries} attempts.")
                    return None

        soup = BeautifulSoup(response.content, "html.parser")

        try:
            # Пытаемся получить заголовок страницы
            title_element = soup.find('span', class_='mw-page-title-main')
            title = title_element.text.strip() if title_element else "No Title"
            print(f"Page title: {title}")

            # Пытаемся получить основное содержимое
            content_div = soup.find('div', class_='mw-content-ltr')
            if not content_div:
                 print(f"Content div not found for {url}")
                 return None

            # Извлекаем текст из всех дочерних элементов
            content_text = content_div.get_text(separator=' ', strip=True)
            # Убираем лишние символы и шаблоны
            content_text = content_text.replace('(Издания для ПК, консолей, консолей старого поколения и мобильных устройств)', '')
            content_text = content_text.replace('(Издания для ПК, консолей и мобильных устройств)', '')
            content_text = content_text.replace(' View or edit this template ', '')
            content_text = content_text.replace('  ', ' ').strip() # Убираем двойные пробелы

            if not content_text:
                print(f"Content is empty for {url}")
                return None

            # Формируем итоговое содержимое
            full_content = f"Заголовок: {title}\n\nСодержимое:\n{content_text}"

            return full_content

        except AttributeError as e:
            print(f"Error parsing content for {url}: {e}")
            return None

    @log_execution
    def _load_links() -> list:
        """
        Функция для извлечения ссылок на страницы вики
        :return:
        list - список ссылок
        """
        if not os.path.exists("data/links.txt"):
            print("Файл data/links.txt не найден. Сначала нужно получить ссылки.")
            return []
        with open("data/links.txt", 'r', encoding='utf-8') as f: # Убедитесь в кодировке
            return [urllib.parse.unquote(link.strip()) for link in f.readlines() if link.strip()] # Используйте strip() для удаления \n и пробелов

    # --- Основной процесс ---
    # 1. Создаем базу данных содержимого
    content_db_path = "data/wiki_content.db"
    create_content_db(content_db_path)

    # 2. Получаем или загружаем список ссылок
    # urls = _get_all_links() # Используйте это, если нужно переполучить ссылки
    # _save_all_links(urls)
    urls = _load_links() # Используйте это, если ссылки уже сохранены

    if not urls:
        print("Список URL-адресов пуст. Завершение.")
        return

    print(f"Загружено {len(urls)} ссылок.")

    # 3. Обрабатываем ссылки порциями
    last_url_index = 0
    batch_size = 5 # Количество страниц за итерацию

    while last_url_index < len(urls):
        print(f"Обработка с индекса {last_url_index}, размер пачки {batch_size}")
        try:
            last_processed_url = _get_data_of_pages(urls, last_url_index, batch_size, content_db_path)
            # Найдем индекс последней обработанной URL в списке
            try:
                last_url_index = urls.index(last_processed_url) + 1
            except ValueError:
                # Если URL не найден в списке (например, из-за изменения), увеличиваем индекс вручную
                last_url_index += batch_size
                if last_url_index >= len(urls):
                    break
        except Exception as e:
            print(f"Произошла ошибка при обработке пачки: {e}")
            # Возможно, стоит сохранить прогресс и выйти или пропустить ошибочную страницу
            break

    print("Парсинг завершен.")


# --- Пример использования новых функций ---
def example_usage():
    db_path = "data/wiki_content.db"

    # 1. Создать базу данных
    create_content_db(db_path)

    # 2. Сохранить пример содержимого (если нужно вручную)
    # save_page_content_to_db("https://example.com", "Это пример содержимого.", db_path)

    # 3. Получить все содержимое
    all_content = get_all_content_from_db(db_path)
    print(f"Всего записей в БД: {len(all_content)}")
    for item in all_content[10:14]:
        print(f"ID: {item[0]}, URL: {item[1][:50]}...") # Выводим первые 50 символов URL
        print(f"Content preview: {item[2]}...\n") # Выводим первые 100 символов контента

    # 4. Получить содержимое по конкретному URL
    # example_url = "https://example.com"
    # specific_content = get_content_by_url(example_url, db_path)
    # if specific_content:
    #     print(f"Найдено содержимое для {example_url}: ID={specific_content[0]}, Content preview={specific_content[2][:100]}...")
    # else:
    #     print(f"Содержимое для {example_url} не найдено.")


if __name__ == "__main__":
    # Запуск основного парсинга
    # parse_wiki()

    # Пример использования функций работы с БД (опционально)
    example_usage()