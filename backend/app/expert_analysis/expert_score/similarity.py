import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer, util
from app.models.article import Article
from app.config.settings import QUERY_LENGTH_THRESHOLD
from fastapi import Request
from app.utils.text_processing import extract_abstract_text
from typing import List, Tuple, Dict, Any, Optional

model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')

def compute_semantic_similarity(article, query_embedding):
    if article.embedding is None:
        return 0.0

    # Перетворення у Tensor, якщо потрібно
    if not isinstance(article.embedding, torch.Tensor):
        article_embedding = torch.tensor(article.embedding, dtype=torch.float32)
    else:
        article_embedding = article.embedding.clone().detach()

    # Перевірка розмірностей
    if article_embedding.shape[0] != query_embedding.shape[0]:
        print("❗ Розмірність ембедінгів не збігається")
        return 0.0

    # Нормалізація
    query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)
    article_vec = F.normalize(article_embedding.view(1, -1), p=2, dim=1)

    score = F.cosine_similarity(query_vec, article_vec).item()
    return score


def compute_query_similarity(query: str, articles: List[Any]) -> float:
    """
    Обчислює середню семантичну схожість запиту до ембедінгів статей.
    """
    if not articles:
        return 0.0

    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
        query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

        similarities = []
        for article in articles:
            emb = article.embedding
            if emb is None:
                continue

            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)

            if emb.shape[0] != query_embedding.shape[0]:
                continue

            article_vec = F.normalize(emb.view(1, -1), p=2, dim=1)
            sim = F.cosine_similarity(query_vec, article_vec).item()
            similarities.append(sim)

        if similarities:
            return float(np.mean(similarities))
        else:
            return 0.0

    except Exception as e:
        print(f"Помилка при обчисленні схожості з ембедінгами: {e}")
        return 0.0


# def compute_semantic_similarity(article, query_embedding):
#     """
#     Обчислює семантичну схожість між запитом і статтею,
#     розбиваючи текст статті на chunks (по ~200 символів).
#     """
#     try:
#         # Витягуємо назву
#         title = getattr(article, "display_name", "")
        
#         # Обробляємо abstract_inverted_index
#         abstract = ""
#         inverted_index = getattr(article, "abstract_inverted_index", None)
#         if isinstance(inverted_index, dict):
#             try:
#                 positions = {}
#                 for word, pos_list in inverted_index.items():
#                     for pos in pos_list:
#                         positions[pos] = word
#                 max_pos = max(positions.keys()) if positions else -1
#                 abstract_words = [positions.get(i, "") for i in range(max_pos + 1)]
#                 abstract = " ".join(abstract_words)
#             except Exception as e:
#                 print(f"❗ Помилка обробки abstract_inverted_index: {e}")
        
#         # Формуємо повний текст
#         full_text = f"{title} {abstract}".strip()
#         if not full_text:
#             return 0.0

#         # Якщо текст короткий — просто порівнюємо
#         if len(full_text) <= 200:
#             chunks = [full_text]
#         else:
#             # Ділимо на шматки по 200 символів (по словах)
#             words = full_text.split()
#             chunks = [' '.join(words[i:i + 40]) for i in range(0, len(words), 40)]

#         if not chunks:
#             return 0.0

#         # Кодування та обчислення схожості
#         chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#         chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
#         query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

#         similarities = F.cosine_similarity(query_vec, chunk_embeddings)
#         return torch.max(similarities).item()

#     except Exception as e:
#         print(f"❗ Помилка в compute_semantic_similarity: {e}")
#         return 0.0



# def compute_query_similarity(query: str, articles: List[Any]) -> float:
#     """
#     Обчислює середню максимальну семантичну схожість між запитом і кожною статтею.
#     Для кожної статті береться найрелевантніший chunk (максимальна подібність).
#     """
#     if not articles:
#         return 0.0

#     try:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

#         similarities = []

#         for article in articles:
#             try:
#                 # 1. Витягуємо назву
#                 title = getattr(article, "display_name", "")

#                 # 2. Витягуємо abstract_inverted_index
#                 abstract = ""
#                 inverted_index = getattr(article, "abstract_inverted_index", None)
#                 if isinstance(inverted_index, dict):
#                     positions = {}
#                     for word, pos_list in inverted_index.items():
#                         for pos in pos_list:
#                             positions[pos] = word
#                     max_pos = max(positions.keys()) if positions else -1
#                     abstract_words = [positions.get(i, "") for i in range(max_pos + 1)]
#                     abstract = " ".join(abstract_words)

#                 # 3. Обʼєднання тексту
#                 full_text = f"{title} {abstract}".strip()
#                 if not full_text:
#                     continue

#                 # 4. Поділ на chunks по ~40 слів
#                 if len(full_text) <= 200:
#                     chunks = [full_text]
#                 else:
#                     words = full_text.split()
#                     chunks = [' '.join(words[i:i + 40]) for i in range(0, len(words), 40)]

#                 if not chunks:
#                     continue

#                 # 5. Кодування та обчислення максимального score
#                 chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#                 chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

#                 sims = F.cosine_similarity(query_vec, chunk_embeddings)
#                 max_sim = torch.max(sims).item()
#                 similarities.append(max_sim)

#             except Exception as inner_e:
#                 print(f"❗ Проблема зі статтею: {inner_e}")
#                 continue

#         return float(np.mean(similarities)) if similarities else 0.0

#     except Exception as e:
#         print(f"❗ Помилка при обчисленні схожості: {e}")
#         return 0.0




































# def compute_query_similarity(query: str, articles: List[Any]) -> float:
#     if not articles:
#         return 0.0

#     try:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

#         similarities = []

#         for article in articles:
#             emb = article.embedding
#             if emb is None:
#                 continue

#             try:
#                 # Якщо це один вектор
#                 if isinstance(emb, (list, np.ndarray)):
#                     emb = torch.tensor(emb, dtype=torch.float32)
#                     article_vec = F.normalize(emb.view(1, -1), p=2, dim=1)
#                     if article_vec.shape[1] != query_vec.shape[1]:
#                         continue
#                     sim = F.cosine_similarity(query_vec, article_vec).item()
#                     similarities.append(sim)

#                 # Якщо це вже тензор
#                 elif isinstance(emb, torch.Tensor):
#                     if emb.ndim == 1:
#                         article_vec = F.normalize(emb.view(1, -1), p=2, dim=1)
#                         if article_vec.shape[1] != query_vec.shape[1]:
#                             continue
#                         sim = F.cosine_similarity(query_vec, article_vec).item()
#                         similarities.append(sim)

#                     elif emb.ndim == 2:
#                         # Це набір embedding-ів chunk'ів
#                         chunk_embeddings = F.normalize(emb, p=2, dim=1)
#                         sims = F.cosine_similarity(query_vec, chunk_embeddings)
#                         max_sim = torch.max(sims).item()
#                         similarities.append(max_sim)

#             except Exception as inner_e:
#                 print(f"❗ Стаття з проблемним embedding: {inner_e}")
#                 continue

#         return float(np.mean(similarities)) if similarities else 0.0

#     except Exception as e:
#         print(f"❗ Помилка при обчисленні схожості з ембедінгами: {e}")
#         return 0.0



# def compute_query_similarity(query: str, articles: List[Any]) -> float:
#     """
#     Обчислює середню максимальну семантичну схожість між запитом і кожною статтею.
#     Для кожної статті береться найрелевантніший chunk (максимальна подібність).
#     """
#     if not articles:
#         return 0.0

#     try:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

#         similarities = []

#         for article in articles:
#             try:
#                 # 1. Витягуємо назву
#                 title = getattr(article, "display_name", "")

#                 # 2. Витягуємо abstract_inverted_index
#                 abstract = ""
#                 inverted_index = getattr(article, "abstract_inverted_index", None)
#                 if isinstance(inverted_index, dict):
#                     positions = {}
#                     for word, pos_list in inverted_index.items():
#                         for pos in pos_list:
#                             positions[pos] = word
#                     max_pos = max(positions.keys()) if positions else -1
#                     abstract_words = [positions.get(i, "") for i in range(max_pos + 1)]
#                     abstract = " ".join(abstract_words)

#                 # 3. Обʼєднання тексту
#                 full_text = f"{title} {abstract}".strip()
#                 if not full_text:
#                     continue

#                 # 4. Поділ на chunks по ~40 слів
#                 if len(full_text) <= 200:
#                     chunks = [full_text]
#                 else:
#                     words = full_text.split()
#                     chunks = [' '.join(words[i:i + 40]) for i in range(0, len(words), 40)]

#                 if not chunks:
#                     continue

#                 # 5. Кодування та обчислення максимального score
#                 chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#                 chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

#                 sims = F.cosine_similarity(query_vec, chunk_embeddings)
#                 max_sim = torch.max(sims).item()
#                 similarities.append(max_sim)

#             except Exception as inner_e:
#                 print(f"❗ Проблема зі статтею: {inner_e}")
#                 continue

#         return float(np.mean(similarities)) if similarities else 0.0

#     except Exception as e:
#         print(f"❗ Помилка при обчисленні схожості: {e}")
#         return 0.0





# def compute_semantic_similarity(article, query_embedding, model, max_tokens=100):
#     """
#     Обчислює семантичну схожість між запитом і статтею, розбитою на частини.
#     Працює з abstract_inverted_index.
#     """

#     # Отримуємо повний текст (title + abstract)
#     full_text = full_text_from_article(article)
#     if not full_text:
#         return 0.0

#     # Ділимо текст на частини по max_tokens слів
#     words = full_text.split()
#     chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
#     if not chunks:
#         return 0.0

#     try:
#         # Кодуємо всі частини
#         chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#         chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

#         # Нормалізуємо запит
#         query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

#         # Косинусна подібність до кожного фрагменту
#         similarities = util.pytorch_cos_sim(query_vec, chunk_embeddings)
#         return torch.max(similarities).item()
#     except Exception as e:
#         print(f"❗ Помилка під час обчислення similarity: {e}")
#         return 0.0
    

# def compute_query_similarity(query: str, articles: List[Any], model, max_tokens=100) -> float:
#     """
#     Обчислює середню максимальну семантичну схожість між запитом і кожною статтею.
#     Кожна стаття ділиться на chunks (назва + абстракт).
#     """
#     if not articles:
#         return 0.0

#     try:
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

#         max_similarities = []

#         for article in articles:
#             # Отримуємо повний текст
#             full_text = full_text_from_article(article)
#             if not full_text:
#                 continue

#             # Розбиваємо на chunks
#             words = full_text.split()
#             chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
#             if not chunks:
#                 continue

#             # Кодуємо chunks
#             chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
#             chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

#             # Обчислюємо cosine similarity запиту до кожного chunk
#             similarities = util.pytorch_cos_sim(query_vec, chunk_embeddings)

#             max_sim = torch.max(similarities).item()
#             max_similarities.append(max_sim)

#         # Повертаємо середнє значення max similarity по всіх статтях
#         return float(np.mean(max_similarities)) if max_similarities else 0.0

#     except Exception as e:
#         print(f"❗ Помилка при обчисленні семантичної схожості (chunk-based): {e}")
#         return 0.0