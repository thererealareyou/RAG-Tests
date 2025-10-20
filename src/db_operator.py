import chromadb

def remove_duplicate_documents(client_path, collection_name):
    """
    Удаляет дубликаты из коллекции ChromaDB на основе содержимого документов (documents).

    Args:
        client_path (str): Путь к PersistentClient.
        collection_name (str): Имя коллекции.
    """
    # 1. Подключаемся к клиенту
    client = chromadb.PersistentClient(path=client_path)
    collection = client.get_collection(collection_name)

    # 2. Получаем все документы, ID и (опционально) метаданные/эмбеддинги
    # 'ids' возвращаются всегда, не нужно указывать в include
    print(f"Получение всех документов из коллекции '{collection_name}'...")
    all_data = collection.get(include=['documents', 'metadatas', 'embeddings'])

    if not all_data['ids']:
        print("Коллекция пуста.")
        return

    ids = all_data['ids']
    documents = all_data['documents']
    # metadatas = all_data['metadatas'] # Не используется в этом примере
    # embeddings = all_data['embeddings'] # Не используется в этом примере

    print(f"Найдено {len(ids)} записей.")

    # 3. Определяем дубликаты на основе содержимого документов
    seen_documents = set()
    duplicate_ids = []
    unique_ids_to_keep = []

    for doc_id, doc_content in zip(ids, documents):
        if doc_content in seen_documents:
            duplicate_ids.append(doc_id)
        else:
            seen_documents.add(doc_content)
            unique_ids_to_keep.append(doc_id)

    print(f"Найдено {len(duplicate_ids)} дубликатов для удаления.")
    print(f"Останется {len(unique_ids_to_keep)} уникальных записей.")

    if not duplicate_ids:
        print("Дубликатов не найдено.")
        return

    # 4. Удаляем дубликаты
    print(f"Удаление {len(duplicate_ids)} дубликатов...")
    try:
        collection.delete(ids=duplicate_ids)
        print("Дубликаты успешно удалены.")
    except Exception as e:
        print(f"Ошибка при удалении дубликатов: {e}")
        return

    # 5. Опционально: Проверка
    remaining_data = collection.get(include=['ids']) # 'ids' всегда возвращаются
    print(f"После удаления в коллекции {len(remaining_data['ids'])} записей.")


# --- Пример использования ---
if __name__ == "__main__":
    CLIENT_PATH = "C://Users//thererealareyou\PycharmProjects\LLMPhilosophy\src\data"  # Укажите ваш путь к базе данных
    COLLECTION_NAME = "outer_wilds_wiki" # Укажите имя вашей коллекции

    # Запускаем функцию удаления дубликатов
    remove_duplicate_documents(
        client_path=CLIENT_PATH,
        collection_name=COLLECTION_NAME
    )
