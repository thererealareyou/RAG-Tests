import re
from typing import List, Dict, Any, Tuple

import chromadb
import torch
import torch.nn.functional as f
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, T5EncoderModel

from src.inference import ROWInferencer


def pool(hidden_state, mask, pooling_method="cls"):
    """
    Пуллинг скрытых состояний.
    """
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")


def get_embedding(text: str, tokenizer, model) -> np.ndarray:
    """
    Генерирует эмбеддинг для текста с помощью заданной модели.
    """
    tokenized = tokenizer(
        text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**tokenized)

    embeddings = pool(
        outputs.last_hidden_state,
        tokenized["attention_mask"],
        pooling_method="cls",
    )
    embeddings = f.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def preprocess_text(text: str) -> List[str]:
    """
    Простая предобработка текста для BM25.
    """
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def initialize_embedding_model(model_name: str) -> Tuple[AutoTokenizer, T5EncoderModel]:
    """
    Загружает токенизатор и модель для эмбеддингов.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=True
    )
    model = T5EncoderModel.from_pretrained(
        model_name, local_files_only=True
    )
    return tokenizer, model


def load_chroma_collection(
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


def create_bm25_index(corpus: List[str]) -> BM25Okapi:
    """
    Создаёт индекс BM25 на основе корпуса.
    """
    tokenized_corpus = [preprocess_text(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(
    query: str, bm25: BM25Okapi, corpus: List[str], doc_ids: List[str], n: int
) -> List[Dict[str, Any]]:
    """
    Выполняет лексический поиск (BM25).
    """
    tokenized_query = preprocess_text(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:n]

    results = []
    for i in top_n_indices:
        results.append({
            "id": doc_ids[i],
            "document": corpus[i],
            "score": bm25_scores[i],
        })
    return results


def semantic_search(
    query_embedding: np.ndarray,
    collection,
    n: int
) -> List[Dict[str, Any]]:
    """
    Выполняет семантический (векторный) поиск.
    """
    vector_results = collection.query(
        query_embeddings=query_embedding, n_results=n
    )

    vector_docs = vector_results["documents"][0]
    vector_doc_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]
    vector_scores = [1 - d for d in vector_distances]

    results = []
    for doc_id, doc_text, v_score, dist in zip(
        vector_doc_ids, vector_docs, vector_scores, vector_distances
    ):
        results.append({
            "id": doc_id,
            "document": doc_text,
            "score": v_score,
            "distance": dist,
        })
    return results


def combine_search_results(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    doc_ids_from_get: List[str],
    bm25_scores_full: List[float],
) -> List[Dict[str, Any]]:
    """
    Объединяет результаты векторного и BM25 поиска.
    """
    combined_results = {}

    for item in vector_results:
        doc_id = item["id"]
        doc_text = item["document"]
        v_score = item["score"]

        if doc_id in combined_results:
            existing = combined_results[doc_id]
            existing["vector_score"] = max(v_score, existing["vector_score"])
        else:
            try:
                doc_idx_in_corpus = doc_ids_from_get.index(doc_id)
                b_score = bm25_scores_full[doc_idx_in_corpus]
            except ValueError:
                b_score = 0.0

            combined_results[doc_id] = {
                "document": doc_text,
                "vector_score": v_score,
                "bm25_score": b_score,
            }

    for item in bm25_results:
        doc_id = item["id"]
        doc_text = item["document"]
        b_score = item["score"]

        if doc_id in combined_results:
            existing = combined_results[doc_id]
            existing["bm25_score"] = max(b_score, existing["bm25_score"])
        else:
            combined_results[doc_id] = {
                "document": doc_text,
                "vector_score": 0.0,
                "bm25_score": b_score,
            }

    return list(combined_results.values())


def calculate_hybrid_score(
    result: Dict[str, float], w_vector: float = 0.4, w_bm25: float = 0.6
) -> float:
    """
    Рассчитывает гибридный скор.
    """
    v_score = result["vector_score"]
    b_score = result["bm25_score"]
    return w_vector * v_score + w_bm25 * b_score


def rank_and_filter_results(
    combined_results: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """
    Ранжирует объединённые результаты и возвращает топ-K.
    """
    sorted_results = sorted(
        combined_results,
        key=lambda x: calculate_hybrid_score(x),
        reverse=True,
    )
    return sorted_results[:k]


def build_prompt(context: str, query: str) -> str:
    """
    Формирует промпт для LLM.
    """
    return f"Контекст: {context}\n\nВопрос: {query}\n\nОтвет:"


def main():
    """
    Основная функция для выполнения гибридного поиска и генерации ответа.
    """

    EMBEDDING_MODEL_NAME = "ai-forever/FRIDA"
    CHROMA_PATH = "src/data"
    COLLECTION_NAME = "outer_wilds_wiki"
    N_BM25 = 10
    N_VECTOR = 10
    K_FINAL = 10
    QUERY = "Расскажи концовку Outer Wilds"

    # 1. Инициализация модели эмбеддингов
    tokenizer_emb, model_emb = initialize_embedding_model(EMBEDDING_MODEL_NAME)

    # 2. Загрузка коллекции из Chroma
    corpus, doc_metadatas, doc_ids_from_get = load_chroma_collection(
        CHROMA_PATH, COLLECTION_NAME
    )

    # 3. Создание индекса BM25
    bm25 = create_bm25_index(corpus)

    # 4. Лексический поиск (BM25)
    bm25_results = lexical_search(
        QUERY, bm25, corpus, doc_ids_from_get, N_BM25
    )

    # 5. Семантический поиск
    query_embedding = get_embedding(QUERY, tokenizer_emb, model_emb)
    # Необходимо получить объект collection снова для семантического поиска
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    vector_results = semantic_search(query_embedding, collection, N_VECTOR)

    # 6. Объединение результатов
    # Для корректного объединения нужен полный список bm25_scores
    tokenized_query_full = preprocess_text(QUERY)
    full_bm25_scores = bm25.get_scores(tokenized_query_full)
    combined_results = combine_search_results(
        vector_results, bm25_results, doc_ids_from_get, full_bm25_scores
    )

    # 7. Ранжирование и фильтрация
    ranked_results = rank_and_filter_results(combined_results, K_FINAL)

    # 8. Формирование контекста и промпта
    context = "\n".join([res["document"] for res in ranked_results])
    prompt = build_prompt(context, QUERY)

    # 9. Генерация ответа
    spi = ROWInferencer()
    answer = spi.generate_response(user_prompt=prompt)
    print(answer["response"])


if __name__ == "__main__":
    main()