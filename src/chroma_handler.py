import chromadb
import numpy as np
from typing import List, Tuple, Dict


def create_vector(embeddings: List[np.ndarray], chunks: List[str], overwrite_collection: bool = False):
    """
    Создаёт или пересоздаёт векторную базу данных ChromaDB.

    Args:
        embeddings (List[np.ndarray]): Список эмбеддингов для чанков.
        chunks (List[str]): Список текстовых чанков.
        overwrite_collection (bool): Если True, удалит существующую коллекцию перед созданием новой.
                                     Если False и коллекция существует, будет ошибка или поведение ChromaDB по умолчанию.
    """
    client = chromadb.PersistentClient(path="./data")
    collection_name = 'outer_wilds_wiki'

    if overwrite_collection:
        existing_collections = [col.name for col in client.list_collections()]
        if collection_name in existing_collections:
            print(f"Коллекция '{collection_name}' существует. Удаляем её...")
            client.delete_collection(collection_name)
            print(f"Коллекция '{collection_name}' удалена.")
        else:
            print(f"Коллекция '{collection_name}' не существует, создаём новую.")

    collection = client.create_collection(collection_name)

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"{i}" for i in range(len(chunks))]
    )
    print(f"Коллекция '{collection_name}' успешно создана с {len(chunks)} элементами.")


def remove_duplicate(client_path, collection_name):
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


def load_collection(
    path: str, collection_name: str
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Подключается к Chroma, загружает документы, метаданные и ID.
    """
    client = chromadb.PersistentClient(path=path)
    collection = client.get_collection(collection_name)
    all_docs_result = collection.get(include=["documents", "metadatas"])
    corpus = all_docs_result["documents"]
    doc_metadatas = all_docs_result.get("metadatas", [{}] * len(corpus))
    doc_ids = all_docs_result["ids"]
    return corpus, doc_metadatas, doc_ids
