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
    try:
        title = getattr(article, "display_name", "")

        abstract = ""
        inverted_index = getattr(article, "abstract_inverted_index", None)
        if isinstance(inverted_index, dict):
            try:
                positions = {}
                for word, pos_list in inverted_index.items():
                    for pos in pos_list:
                        positions[pos] = word
                max_pos = max(positions.keys()) if positions else -1
                abstract_words = [positions.get(i, "") for i in range(max_pos + 1)]
                abstract = " ".join(abstract_words)
            except Exception as e:
                print(f"❗ Помилка обробки abstract_inverted_index: {e}")
        
        # Формуємо повний текст
        full_text = f"{title} {abstract}".strip()
        if not full_text:
            return 0.0

        if len(full_text) <= 200:
            chunks = [full_text]
        else:
            words = full_text.split()
            chunks = [' '.join(words[i:i + 40]) for i in range(0, len(words), 40)]

        if not chunks:
            return 0.0

        # Кодування та обчислення схожості
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
        chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
        query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

        similarities = F.cosine_similarity(query_vec, chunk_embeddings)
        return torch.max(similarities).item()

    except Exception as e:
        print(f"❗ Помилка в compute_semantic_similarity: {e}")
        return 0.0



def compute_query_similarity(query: str, articles: List[Any]) -> float:
    if not articles:
        return 0.0

    try:
        query_embedding = model.encode(query, convert_to_tensor=True)
        query_vec = F.normalize(query_embedding.view(1, -1), p=2, dim=1)

        similarities = []

        for article in articles:
            try:
                title = getattr(article, "display_name", "")

                abstract = ""
                inverted_index = getattr(article, "abstract_inverted_index", None)
                if isinstance(inverted_index, dict):
                    positions = {}
                    for word, pos_list in inverted_index.items():
                        for pos in pos_list:
                            positions[pos] = word
                    max_pos = max(positions.keys()) if positions else -1
                    abstract_words = [positions.get(i, "") for i in range(max_pos + 1)]
                    abstract = " ".join(abstract_words)

                # 3. Обʼєднання тексту
                full_text = f"{title} {abstract}".strip()
                if not full_text:
                    continue

                # 4. Поділ на chunks по ~40 слів
                if len(full_text) <= 200:
                    chunks = [full_text]
                else:
                    words = full_text.split()
                    chunks = [' '.join(words[i:i + 40]) for i in range(0, len(words), 40)]

                if not chunks:
                    continue

                # 5. Кодування та обчислення максимального score
                chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
                chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)

                sims = F.cosine_similarity(query_vec, chunk_embeddings)
                max_sim = torch.max(sims).item()
                similarities.append(max_sim)

            except Exception as inner_e:
                print(f"❗ Проблема зі статтею: {inner_e}")
                continue

        return float(np.mean(similarities)) if similarities else 0.0

    except Exception as e:
        print(f"❗ Помилка при обчисленні схожості: {e}")
        return 0.0
