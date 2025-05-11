import asyncio
import concurrent.futures
import torch
import torch.nn.functional as F
from sentence_transformers import util

# async def batch_encode_embeddings(model, texts, batch_size=32):
#     """Обробка ембедінгів партіями з використанням ThreadPoolExecutor для паралелізації"""
#     all_embeddings = []
    
#     # Обробка порожнього списку текстів
#     if not texts:
#         return all_embeddings
    
#     # Очистка текстів перед обробкою - видаляємо порожні тексти
#     valid_texts = [text for text in texts if text and isinstance(text, str)]
#     if not valid_texts:
#         return [None] * len(texts)
    
#     # Розбиваємо тексти на партії
#     batches = [valid_texts[i:i+batch_size] for i in range(0, len(valid_texts), batch_size)]
    
#     # ThreadPoolExecutor для паралельної обробки партій
#     loop = asyncio.get_event_loop()
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Функція для безпечного кодування партії
#         def safe_encode(batch):
#             try:
#                 embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
#                 # Додатково нормалізуємо ембедінги для коректного порівняння косинусом
#                 return F.normalize(embeddings, p=2, dim=1)
#             except Exception as e:
#                 print(f"Помилка під час кодування партії: {e}")
#                 return [None] * len(batch)
        
#         # Створюємо завдання для кожної партії
#         futures = [loop.run_in_executor(executor, safe_encode, batch) for batch in batches]
        
#         # Очікуємо завершення всіх завдань
#         for future in asyncio.as_completed(futures):
#             try:
#                 batch_embeddings = await future
#                 if batch_embeddings is not None:
#                     all_embeddings.extend(batch_embeddings)
#             except Exception as e:
#                 print(f"Помилка при обробці результатів партії: {e}")
        
#     # Переконуємося, що кількість ембедінгів відповідає кількості текстів
#     while len(all_embeddings) < len(texts):
#         all_embeddings.append(None)
    
#     return all_embeddings[:len(texts)]

import torch
import torch.nn.functional as F
import concurrent.futures
import asyncio
from typing import List, Any

async def batch_encode_embeddings(model, texts: List[str], batch_size: int = 32) -> List[Any]:
    """
    Асинхронне кодування ембедінгів для списку текстів, з урахуванням їх позиції у вхідному списку.
    
    Порожні або невалідні тексти ігноруються, але їх позиції у фінальному списку заповнюються `None`.
    """

    if not texts:
        return []

    # Зберігаємо індекси та тексти
    indexed_texts = [(i, text) for i, text in enumerate(texts) if text and isinstance(text, str)]
    if not indexed_texts:
        return [None] * len(texts)

    index_map = [i for i, _ in indexed_texts]
    valid_texts = [t for _, t in indexed_texts]

    # Розбиваємо на партії
    batches = [valid_texts[i:i + batch_size] for i in range(0, len(valid_texts), batch_size)]

    loop = asyncio.get_event_loop()
    all_embeddings = []

    def safe_encode(batch):
        try:
            embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            return F.normalize(embeddings, p=2, dim=1)
        except Exception as e:
            print(f"❗ Помилка при кодуванні партії: {e}")
            return [None] * len(batch)

    # Обробка в ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [loop.run_in_executor(executor, safe_encode, batch) for batch in batches]

        for future in asyncio.as_completed(futures):
            try:
                batch_embeddings = await future
                if batch_embeddings is not None:
                    if isinstance(batch_embeddings, torch.Tensor):
                        all_embeddings.extend(batch_embeddings)
                    else:
                        all_embeddings.extend([None] * len(batch_embeddings))  # на випадок помилки
            except Exception as e:
                print(f"❗ Помилка при обробці партії: {e}")

    # Відновлюємо порядок відповідно до `texts`
    final_embeddings = [None] * len(texts)
    for idx, emb in zip(index_map, all_embeddings):
        final_embeddings[idx] = emb

    return final_embeddings



def semantic_sort(articles, query_embedding, top_k):
    scored = []
    for art in articles:
        if art.embedding is None:
            continue
        try:
            article_embedding = torch.tensor(art.embedding)
            article_embedding = F.normalize(article_embedding, p=2, dim=0)
            score = util.pytorch_cos_sim(query_embedding.unsqueeze(0), article_embedding.unsqueeze(0)).item()
            scored.append((art, score))
        except Exception as e:
            print(f"Помилка cosine similarity: {e}")
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:top_k]]