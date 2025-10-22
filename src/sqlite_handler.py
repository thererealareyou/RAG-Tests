import os
import sqlite3
from typing import List, Tuple, Optional


def create_content(db_path: str = "data/wiki_content.db"):
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


def save_chunks(page_id: int, chunks: List[str], db_path: str = "data/wiki_content.db"):
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

