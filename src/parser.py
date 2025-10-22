from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException
import urllib.parse
import sqlite3
import os
from typing import List, Tuple, Optional
import time
from functools import wraps


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
                save_or_update_page_by_url(url, content, db_path)
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


def create_content_db(db_path: str = "data/wiki_content.db"):
    """
    Создает SQLite базу данных и таблицу для хранения содержимого страниц.
    Также создает таблицу для хранения чанков, связанную с основной таблицей.

    Args:
        db_path (str): Путь к файлу базы данных.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Основная таблица для страниц
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wiki_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL
        )
    ''')

    # Таблица для чанков, связанная с wiki_pages через page_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wiki_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_order INTEGER, -- Порядковый номер чанка в документе (опционально)
            FOREIGN KEY(page_id) REFERENCES wiki_pages(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print(f"База данных содержимого и чанков создана: {db_path}")


def save_or_update_page_by_url(url: str, content: str, db_path: str = "data/wiki_content.db") -> Optional[int]:
    """
    Сохраняет (или обновляет, если URL уже существует) содержимое одной страницы в базу данных по URL.
    Возвращает ID сохраненной/обновленной страницы.

    Args:
        url (str): URL страницы.
        content (str): Содержимое страницы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Optional[int]: ID сохраненной/обновленной страницы или None в случае ошибки.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    page_id = None
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO wiki_pages (url, content) VALUES (?, ?)",
            (url, content)
        )
        # Получаем ID вставленной или замененной строки
        cursor.execute("SELECT id FROM wiki_pages WHERE url = ?", (url,))
        row = cursor.fetchone()
        if row:
            page_id = row[0]
            conn.commit()
            print(f"Содержимое страницы сохранено/обновлено в БД: {url}, ID: {page_id}")
        else:
            # Теоретически маловероятно при INSERT OR REPLACE, но на всякий случай
            print(f"Ошибка: Не удалось получить ID после INSERT OR REPLACE для {url}.")
            page_id = None
    except sqlite3.Error as e:
        print(f"Ошибка SQLite при сохранении/обновлении {url}: {e}")
        page_id = None # Возвращаем None в случае ошибки
    finally:
        conn.close()
    return page_id # Возвращаем ID страницы


def save_or_update_page_by_id(page_id: int, content: str, db_path: str = "data/wiki_content.db") -> Optional[bool]:
    """
    Обновляет содержимое одной страницы в базу данных по её ID.
    Возвращает True, если обновление прошло успешно, False, если запись не найдена или произошла ошибка.

    Args:
        page_id (int): ID страницы.
        content (str): Новое содержимое страницы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Optional[bool]: True если обновлено, False если не найдено или ошибка.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    success = False
    try:
        cursor.execute(
            "UPDATE wiki_pages SET content = ? WHERE id = ?",
            (content, page_id)
        )
        # Проверяем, была ли затронута хотя бы одна строка
        if cursor.rowcount > 0:
            conn.commit()
            print(f"Содержимое страницы с ID {page_id} обновлено.")
            success = True
        else:
            print(f"Предупреждение: Страница с ID {page_id} не найдена. Нечего обновлять.")
            success = False # Возвращаем False, если запись не найдена
    except sqlite3.Error as e:
        print(f"Ошибка SQLite при обновлении содержимого для ID {page_id}: {e}")
        success = False # Возвращаем False в случае ошибки
    finally:
        conn.close()
    return success


def save_chunks_to_db(page_id: int, chunks: List[str], db_path: str = "data/wiki_content.db"):
    """
    Сохраняет список чанков в базу данных, связывая их с ID страницы.

    Args:
        page_id (int): ID страницы из таблицы wiki_pages.
        chunks (List[str]): Список строк, представляющих чанки текста.
        db_path (str): Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Удаляем существующие чанки для этой страницы (если нужно обновить)
        cursor.execute("DELETE FROM wiki_chunks WHERE page_id = ?", (page_id,))

        # Вставляем новые чанки
        for i, chunk_text in enumerate(chunks):
            cursor.execute(
                "INSERT INTO wiki_chunks (page_id, chunk_text, chunk_order) VALUES (?, ?, ?)",
                (page_id, chunk_text, i) # Сохраняем порядок
            )
        conn.commit()
        print(f"Сохранено {len(chunks)} чанков для страницы ID {page_id}")
    except sqlite3.Error as e:
        print(f"Ошибка SQLite при сохранении чанков для ID {page_id}: {e}")
    finally:
        conn.close()


def get_all_pages(db_path: str = "data/wiki_content.db") -> List[Tuple[int, str, str]]:
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


def get_all_chunks(db_path: str = "data/wiki_content.db") -> List[Tuple[int, int, str, int]]:
    """
    Извлекает все чанки из базы данных.

    Args:
        db_path (str): Путь к файлу базы данных.

    Returns:
        List[Tuple[int, int, str, int]]: Список всех кортежей (chunk_id, page_id, chunk_text, chunk_order).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, page_id, chunk_text, chunk_order FROM wiki_chunks ORDER BY page_id, chunk_order")
    results = cursor.fetchall()
    conn.close()
    return results


def get_page_by_url(url: str, db_path: str = "data/wiki_content.db") -> Tuple[int, str, str] | None:
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


def get_page_by_id(page_id: int, db_path: str = "data/wiki_content.db") -> Tuple[int, str, str] | None:
    """
    Извлекает содержимое страницы по ID из базы данных.

    Args:
        page_id (int): ID страницы.
        db_path (str): Путь к файлу базы данных.

    Returns:
        Tuple[int, str, str] | None: Кортеж (id, url, content) или None, если не найдено.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, url, content FROM wiki_pages WHERE id = ?", (page_id,))
    result = cursor.fetchone()
    conn.close()
    return result


def get_chunks_by_page_id(page_id: int, db_path: str = "data/wiki_content.db") -> List[Tuple[int, int, str, int]]:
    """
    Извлекает все чанки для заданного ID страницы.

    Args:
        page_id (int): ID страницы из таблицы wiki_pages.
        db_path (str): Путь к файлу базы данных.

    Returns:
        List[Tuple[int, int, str, int]]: Список кортежей (chunk_id, page_id, chunk_text, chunk_order).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, page_id, chunk_text, chunk_order FROM wiki_chunks WHERE page_id = ? ORDER BY chunk_order", (page_id,))
    results = cursor.fetchall()
    conn.close()
    return results


def delete_page_and_chunks(page_id: int, delete_original: bool = False, db_path: str = "data/wiki_content.db"):
    """
    Удаляет чанки, связанные с указанным ID страницы.
    При необходимости также удаляет оригинальный документ из wiki_pages.

    Args:
        page_id (int): ID страницы, чанки которой нужно удалить.
        delete_original (bool): Если True, удаляет также запись из wiki_pages.
                                Если False, удаляет только чанки.
        db_path (str): Путь к файлу базы данных.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Удаляем чанки
        cursor.execute("DELETE FROM wiki_chunks WHERE page_id = ?", (page_id,))
        deleted_chunks_count = cursor.rowcount
        print(f"Удалено {deleted_chunks_count} чанков для страницы ID {page_id}.")

        # Удаляем оригинальный документ, если нужно
        if delete_original:
            cursor.execute("DELETE FROM wiki_pages WHERE id = ?", (page_id,))
            deleted_pages_count = cursor.rowcount
            if deleted_pages_count > 0:
                print(f"Оригинальный документ (ID {page_id}) также удален.")
            else:
                print(f"Предупреждение: Оригинальный документ с ID {page_id} не найден для удаления.")
        else:
            print(f"Оригинальный документ (ID {page_id}) оставлен в базе данных.")

        conn.commit()
    except sqlite3.Error as e:
        print(f"Ошибка SQLite при удалении для ID {page_id}: {e}")
    finally:
        conn.close()


def form_chunks(content: str) -> List[str]:
    """
    Формирует чанки. Разделение осуществляется по точкам и ограничению длины полученных предложений.

    Args:
        content (str): Содержимое страницы.

    Returns:
        List[str]: Список полученных предложений.
    """
    total_chunks = []
    current_chunk = ''
    length_of_current_chunk = 0
    splitted_content = content.split('.')
    for chunk in splitted_content:
        if length_of_current_chunk + len(chunk) < 768:
            current_chunk = current_chunk + chunk
            length_of_current_chunk += len(chunk)
        else:
            total_chunks.append(current_chunk)
            current_chunk = ''
            length_of_current_chunk = 0

    return total_chunks


def example_usage():
    data = get_all_pages()
    ids = [row[0] for row in data]
    for page_id in ids:
        _, url, unchunked_content = get_page_by_id(page_id)
        chunked_content = form_chunks(unchunked_content)
        save_chunks_to_db(page_id, chunked_content)

if __name__ == "__main__":
    example_usage()
