import asyncio
import concurrent.futures
import torch
import torch.nn.functional as F
from sentence_transformers import util
import torch
import torch.nn.functional as F
import concurrent.futures
import asyncio
from typing import List, Any

async def batch_encode_embeddings(model, texts: List[str], batch_size: int = 32) -> List[Any]:
    if not texts:
        return []

    # Зберігання індексів та текстів
    indexed_texts = [(i, text) for i, text in enumerate(texts) if text and isinstance(text, str)]
    if not indexed_texts:
        return [None] * len(texts)

    index_map = [i for i, _ in indexed_texts]
    valid_texts = [t for _, t in indexed_texts]

    # Розбивиття на партії
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

    # Відновлення порядку відповідно до `texts`
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