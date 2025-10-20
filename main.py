import chromadb
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from rank_bm25 import BM25Okapi
import re

from src.inference import SarcasmPhilosopherInferencer


def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]

# === 1. Загружаем модель для эмбеддингов ===
tokenizer_emb = AutoTokenizer.from_pretrained("ai-forever/FRIDA", local_files_only=True)
model_emb = T5EncoderModel.from_pretrained("ai-forever/FRIDA", local_files_only=True)

def get_embedding(text):
    tokenized = tokenizer_emb(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_emb(**tokenized)
    embeddings = pool(outputs.last_hidden_state, tokenized["attention_mask"], pooling_method="cls")
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def preprocess_text(text):
    """Простая предобработка текста для BM25: приведение к нижнему регистру и токенизация по пробелам."""
    tokens = re.findall(r'\w+', text.lower())
    return tokens

# === 2. Подключаемся к векторной базе ===
client = chromadb.PersistentClient(path="src/data")
collection = client.get_collection("outer_wilds_wiki")

# === 2.1. Загружаем все документы и их ID из базы для BM25 ===
all_docs_result = collection.get(include=['documents', 'metadatas'])
corpus = all_docs_result['documents']
doc_metadatas = all_docs_result.get('metadatas', [{}] * len(corpus))
doc_ids_from_get = all_docs_result['ids']
print(f"Загружено {len(corpus)} документов для BM25 из базы Chroma.")

# === 2.2. Создаём индекс BM25 на основе загруженного корпуса ===
tokenized_corpus = [preprocess_text(doc) for doc in corpus]
print(f"Создан индекс BM25 для {len(tokenized_corpus)} документов.")
# Отладка: посмотрим первые токенизированные документы
# print("Примеры токенизированных документов:", tokenized_corpus[:2])

bm25 = BM25Okapi(tokenized_corpus)

# === 3. Тестовый запрос ===
query = "Кто такой удильщик?"  #  Запрос
print(f"Запрос: '{query}'")

# === 4. Лексический поиск (BM25) ===
tokenized_query = preprocess_text(query)
print(f"Токенизированный запрос: {tokenized_query}")

bm25_scores = bm25.get_scores(tokenized_query)
print(f"Получено {len(bm25_scores)} оценок BM25 (должно совпадать с количеством документов).")

# === 4.1. Получаем топ-N результатов из BM25 ===
N_BM25 = 10
top_n_indices_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:N_BM25]
top_n_docs_bm25 = [corpus[i] for i in top_n_indices_bm25]
top_n_ids_bm25 = [doc_ids_from_get[i] for i in top_n_indices_bm25]
top_n_scores_bm25 = [bm25_scores[i] for i in top_n_indices_bm25]

print(f"Топ-{N_BM25} документов по BM25:")
for i, (idx, score) in enumerate(zip(top_n_indices_bm25, top_n_scores_bm25)):
    print(f"  {i+1}. ID: {top_n_ids_bm25[i]}, Score: {score:.4f}, Doc: '{top_n_docs_bm25[i][:100]}...'") # Печатаем первые 100 символов

# === 5. Получаем эмбеддинг запроса ===
query_embedding = get_embedding(query)

# === 6. Выполняем семантический (векторный) поиск ===
N_VECTOR = 10
vector_results = collection.query(
    query_embeddings=query_embedding,
    n_results=N_VECTOR
)

# === 6.1. Извлекаем результаты векторного поиска ===
vector_docs = vector_results['documents'][0]
vector_doc_ids = vector_results['ids'][0]
vector_distances = vector_results['distances'][0]
vector_scores = [1 - d for d in vector_distances]

print(f"Топ-{N_VECTOR} документов по векторному поиску:")
for i, (doc_id, v_score, dist) in enumerate(zip(vector_doc_ids, vector_scores, vector_distances)):
    print(f"  {i+1}. ID: {doc_id}, Vector Score: {v_score:.4f}, Distance: {dist:.4f}, Doc: '{vector_docs[i][:100]}...'") # Печатаем первые 100 символов

# === 7. Объединение результатов ===
combined_results = {}

# Добавляем результаты векторного поиска
print("\n--- Добавляем результаты векторного поиска ---")
for doc_text, doc_id, v_score in zip(vector_docs, vector_doc_ids, vector_scores):
    print(f"  Обрабатываю векторный результат: ID={doc_id}, Vector Score={v_score:.4f}")
    if doc_id in combined_results:
        combined_results[doc_id]['vector_score'] = max(v_score, combined_results[doc_id]['vector_score'])
        print(f"    Обновлён векторный скор для ID {doc_id}")
    else:
        # Нужно найти BM25 оценку для этого документа
        # Найдём индекс документа в corpus по его ID
        try:
            doc_idx_in_corpus = doc_ids_from_get.index(doc_id)
            print(f"    Найден индекс в корпусе: {doc_idx_in_corpus}")
            # Получим оценку BM25 для этого индекса
            b_score = bm25_scores[doc_idx_in_corpus]
            print(f"    BM25 Score для этого документа: {b_score:.4f}")
        except ValueError:
            print(f"    ОШИБКА: ID {doc_id} из векторного поиска не найден в списке ID из collection.get()!")
            b_score = 0.0 # Если ID не найден, присваиваем 0

        combined_results[doc_id] = {
            'document': doc_text,
            'vector_score': v_score,
            'bm25_score': b_score,
        }
        print(f"    Добавлен документ в combined_results с BM25 Score: {b_score:.4f}")

# Добавляем результаты BM25 (только если их нет в векторном поиске)
print("\n--- Добавляем результаты BM25 (если ID не в векторном поиске) ---")
for doc_text, doc_id, b_score in zip(top_n_docs_bm25, top_n_ids_bm25, top_n_scores_bm25):
    print(f"  Обрабатываю BM25 результат: ID={doc_id}, BM25 Score={b_score:.4f}")
    if doc_id in combined_results:
        # Обновляем BM25 score, если он выше
        if b_score > combined_results[doc_id]['bm25_score']:
             combined_results[doc_id]['bm25_score'] = b_score
             print(f"    Обновлён BM25 скор для ID {doc_id} до {b_score:.4f}")
        else:
             print(f"    BM25 скор ({b_score:.4f}) не выше существующего ({combined_results[doc_id]['bm25_score']:.4f}) для ID {doc_id}")
    else:
        print(f"    Добавлен новый документ из BM25 в combined_results")
        combined_results[doc_id] = {
            'document': doc_text,
            'vector_score': 0.0,
            'bm25_score': b_score,
        }

# === 8. Ранжирование объединённых результатов ===
WEIGHT_VECTOR = 0.4
WEIGHT_BM25 = 0.6

def calculate_hybrid_score(result):
    v_score = result['vector_score']
    b_score = result['bm25_score']
    return WEIGHT_VECTOR * v_score + WEIGHT_BM25 * b_score

sorted_results_with_id = sorted(combined_results.items(), key=lambda item: calculate_hybrid_score(item[1]), reverse=True)

# === 9. Выбираем топ-K финальных результатов ===
K_FINAL = 10
final_results_data = [item[1] for item in sorted_results_with_id[:K_FINAL]]
final_docs = [res['document'] for res in final_results_data]

print("\n🔍 Найденные документы (гибридный поиск):")
for i, (doc_id, res_data) in enumerate(sorted_results_with_id[:K_FINAL]):
    score = calculate_hybrid_score(res_data)
    print(f"Doc {i+1} (ID: {doc_id}): Score={score:.4f}, Vector={res_data['vector_score']:.4f}, BM25={res_data['bm25_score']:.4f}")
    print(f"Text: {res_data['document']}")
    print("-" * 50)

# === 10. Формируем промпт для LLM ===
context = "\n".join(final_docs)

prompt = f"""
Контекст: {context}

Вопрос: {query}

Ответ:
"""

SPI = SarcasmPhilosopherInferencer()
answer = SPI.generate_response(user_prompt=prompt)
print(answer['response'])