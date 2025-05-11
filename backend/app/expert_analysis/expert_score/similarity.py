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

# def compute_semantic_similarity(article, query_embedding: torch.Tensor) -> float:
#     """Обчислює косинусну подібність між embedding запиту та статтею."""

#     # def extract_abstract_text(inverted_index: dict) -> str:
#     #     """Відновлює текст абстракту з inverted index."""
#     #     if not inverted_index:
#     #         return ""
#     #     sorted_words = sorted(inverted_index.items(), key=lambda x: min(x[1]))
#     #     return " ".join(word for word, _ in sorted_words)
#     try:
#         # Якщо є embedding
#         if hasattr(article, 'embedding') and article.embedding is not None:
#             article_embedding = article.embedding.clone().detach()
#         else:
#             # Створюємо текст: заголовок + абстракт
#             title = getattr(article, 'display_name', '') or ''
#             abstract_raw = getattr(article, 'abstract_inverted_index', {})
#             abstract = extract_abstract_text(abstract_raw)
#             full_text = (title + ". " + abstract).strip()
#             if not full_text:
#                 return 0.0
#             article_embedding = model.encode(full_text, convert_to_tensor=True)

#         # Нормалізуємо обидва вектори
#         query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
#         article_norm = F.normalize(article_embedding.unsqueeze(0), p=2, dim=1)
#         similarity = util.pytorch_cos_sim(query_norm, article_norm).item()
#         return float(np.clip(similarity, 0.0, 1.0))

#     except Exception as e:
#         print(f"Помилка compute_semantic_similarity: {e}")
#         return 0.0

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


# def compute_query_similarity(query: str, articles: List[Any]) -> float:
#     """
#     Обчислює середню семантичну схожість запиту до заголовків статей.

#     Args:
#         query: пошуковий запит
#         articles: список статей
#         model: SentenceTransformer модель

#     Returns:
#         Середній показник схожості (0.0-1.0)
#     """
#     if not articles:
#         return 0.0

#     try:
#         titles = [getattr(article, "display_name", "") for article in articles if getattr(article, "display_name", "")]
#         if not titles:
#             return 0.0

#         # Кодування запиту та заголовків
#         query_embedding = model.encode(query, convert_to_tensor=True)
#         title_embeddings = model.encode(titles, convert_to_tensor=True)

#         # Нормалізація
#         query_embedding = F.normalize(query_embedding, p=2, dim=0)
#         title_embeddings = F.normalize(title_embeddings, p=2, dim=1)

#         # Обчислення схожості
#         similarities = util.pytorch_cos_sim(query_embedding.unsqueeze(0), title_embeddings)
#         return torch.mean(similarities).item()

#     except Exception as e:
#         print(f"Помилка при обчисленні семантичної схожості: {e}")
#         return 0.0

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



# from typing import List, Dict, Any
# import numpy as np
# from app.models.article import Article
# from app.config.settings import CURRENT_YEAR
# from app.utils.text_processing import extract_abstract_text

# def get_author_influence_score(authorships: List[Dict[str, Any]]) -> float:
#     """Обчислює вплив авторів на основі їх h-індексу, affiliated_institutions та кількості публікацій
#     з покращеною диференціацією."""
#     if not authorships:
#         return 0.0
    
#     # Даємо більше значення першим 3-5 авторам, особливо першому та corresponding автору
#     # Сортуємо авторів за їх позицією - перший, кореспондуючий, потім всі інші
#     position_priority = {"first": 0, "corresponding": 1}
#     sorted_authors = sorted(authorships, key=lambda a: position_priority.get(a.get("author_position", ""), 999))
    
#     top_author_score = 0.0
#     additional_authors_score = 0.0
    
#     # Список престижних освітніх інституцій (можна розширити)
#     prestigious_institutions = [
#         "harvard", "stanford", "mit", "cambridge", "oxford", "california", "princeton", 
#         "yale", "chicago", "tokyo", "berkeley", "eth zurich", "caltech", "imperial"
#     ]
    
#     # Обробляємо кожного автора
#     for i, author in enumerate(sorted_authors):
#         # Базовий бал залежно від позиції автора (вище для перших авторів)
#         if i == 0:  # Перший автор
#             author_score = 0.3
#         elif i == 1 and author.get("author_position") == "corresponding":  # Кореспондуючий автор
#             author_score = 0.25
#         elif i < 3:  # Інші перші автори (топ-3)
#             author_score = 0.15
#         else:  # Всі інші автори
#             author_score = 0.05
        
#         # Враховуємо h-індекс автора, якщо він є, з нелінійною залежністю
#         author_h_index = author.get("author", {}).get("h_index", 0)
#         if author_h_index:
#             h_index_score = 1 - np.exp(-author_h_index / 30)  # Експоненційне насичення
#             author_score += h_index_score * 0.4
        
#         # Враховуємо інституції з підвищеною чутливістю до престижних установ
#         institutions = author.get("institutions", [])
#         if institutions:
#             # Базовий бал за наявність інституцій
#             institution_count = len(institutions)
#             author_score += min(institution_count * 0.08, 0.25)
            
#             # Підвищений бал за престижні інституції
#             for inst in institutions:
#                 inst_name = inst.get("display_name", "").lower()
                
#                 # Навчальні заклади
#                 if inst.get("type") == "education" and inst_name:
#                     # Перевірка на престижні університети
#                     if any(univ in inst_name for univ in prestigious_institutions):
#                         author_score += 0.2  # Значний бонус за престижний університет
#                     else:
#                         author_score += 0.05  # Базовий бонус за освітню установу
                
#                 # Дослідницькі установи
#                 elif inst.get("type") == "facility" and inst_name:
#                     author_score += 0.08  # Бонус за дослідні установи
        
#         # Враховуємо роль автора (якщо головний автор)
#         if author.get("author_position") in ["first", "corresponding"]:
#             author_score *= 1.6  # Більший множник для посилення ваги головних авторів
        
#         # Додаємо бал цього автора до загального
#         if i == 0:  # Зберігаємо найвищий бал першого автора
#             top_author_score = author_score
#         else:  # Інші автори дають зменшений внесок
#             additional_authors_score += author_score * (0.9 ** (i-1))  # Експоненційне зменшення впливу
    
#     # Загальний бал - це сума найкращого автора та зважених внесків інших
#     score = top_author_score + 0.5 * min(additional_authors_score, 1.0)
    
#     # Нормалізуємо фінальний бал
#     return min(score, 1.0)

# def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
#     """Оцінює релевантність концептів статті до запиту з покращеною диференціацією."""
#     if not concepts:
#         return 0.0
    
#     query_terms = set(query.lower().split())
#     score = 0.0
#     match_found = False
    
#     # Сортуємо концепти за рівнем (щоб спочатку обробити найвищі рівні)
#     # та за оцінкою (щоб спочатку обробити найбільш релевантні концепти)
#     sorted_concepts = sorted(concepts, key=lambda c: (c.get("level", 999), -c.get("score", 0)))
    
#     for concept in sorted_concepts:
#         concept_name = concept.get("display_name", "").lower()
#         concept_score = concept.get("score", 0)
#         level = concept.get("level", 0)
        
#         # Перевірка на пряме співпадіння з запитом
#         term_match = False
#         for term in query_terms:
#             if term in concept_name:
#                 # Значно збільшуємо бал за пряме співпадіння з запитом
#                 term_match = True
#                 match_found = True
#                 direct_match_score = 0.3 * concept_score
#                 score += direct_match_score
        
#         # Додатковий бонус за повне співпадіння
#         if concept_name in query_terms:
#             score += 0.2
#             match_found = True
        
#         # Врахування загального рівня концепту (лише якщо не було прямого співпадіння)
#         # Використовуємо експоненційне затухання за рівнями
#         level_factor = np.exp(-level * 0.5)  # e^0 = 1 для level=0, ~0.6 для level=1, ~0.37 для level=2
        
#         # Додаємо бал на основі рівня навіть якщо немає прямого співпадіння
#         score += 0.2 * concept_score * level_factor
    
#     # Бонус, якщо знайдено хоча б одне співпадіння (заохочує статті з будь-яким релевантним концептом)
#     if match_found:
#         score *= 1.2
    
#     return min(score, 1.0)

# def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
#     """Оцінює якість цитування з урахуванням віку публікації з покращеною диференціацією."""
#     if not cited_by_count or not publication_year:
#         return 0.0
    
#     years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
#     # Обчислюємо середню кількість цитувань за рік
#     citations_per_year = cited_by_count / years_since_publication
    
#     # Використовуємо більш розтягнуту логарифмічну шкалу для кращої диференціації
#     # Збільшуємо максимальний бал до 1.0 замість 0.3 для ширшого діапазону
#     return min(np.log1p(citations_per_year) / np.log1p(30), 1.0)

# def get_keyword_match_score(keywords: List[Any], query: str) -> float:
#     """Оцінює збіг ключових слів статті із запитом з покращеною диференціацією."""
#     if not keywords:
#         return 0.0
    
#     query_terms = set(query.lower().split())
    
#     # Відстеження збігів для кожного ключового слова в запиті
#     term_matches = {term: 0 for term in query_terms}
#     exact_matches = 0
#     partial_matches = 0
    
#     # Розширена оцінка для кожного ключового слова
#     for keyword in keywords:
#         # Обробляємо випадок, коли keyword може бути словником
#         if isinstance(keyword, dict):
#             # Шукаємо потрібні поля у словнику (display_name, name або value)
#             keyword_text = keyword.get('display_name', keyword.get('name', keyword.get('value', '')))
#             if isinstance(keyword_text, str):
#                 keyword_lower = keyword_text.lower()
#             else:
#                 continue
#         elif isinstance(keyword, str):
#             keyword_lower = keyword.lower()
#         else:
#             continue
        
#         # Перевірка на точний збіг (коли ключове слово повністю співпадає з терміном запиту)
#         if keyword_lower in query_terms:
#             exact_matches += 1
#             term_matches[keyword_lower] += 2  # Даємо більшу вагу точним збігам
#             continue
            
#         # Перевірка на частковий збіг (коли термін запиту міститься в ключовому слові)
#         for term in query_terms:
#             if term in keyword_lower:
#                 partial_matches += 1
#                 term_matches[term] += 1
#                 break
    
#     # Відсоток знайдених термінів у запиті (використовуємо для розрахунку повноти збігу)
#     terms_covered = sum(1 for term, count in term_matches.items() if count > 0)
#     coverage_ratio = terms_covered / len(query_terms) if query_terms else 0
    
#     # Обчислюємо загальний бал на основі кількості та якості збігів
#     base_score = 0.4 * min(exact_matches * 0.2, 0.6) + 0.2 * min(partial_matches * 0.1, 0.4)
    
#     # Додатковий бонус за широке охоплення термінів запиту
#     coverage_bonus = coverage_ratio * 0.4
    
#     # Комбінуємо бали
#     final_score = base_score + coverage_bonus
    
#     return min(final_score, 1.0)

# def get_language_relevance(language: str, query: str) -> float:
#     """Оцінює релевантність мови статті відносно запиту."""
#     # Визначаємо мову запиту (спрощений підхід)
#     query_lang = "en"  # За замовчуванням англійська
    
#     # Спрощена логіка визначення мови запиту
#     cyrillic_chars = set('абвгдеєжзиіїйклмнопрстуфхцчшщьюя')
#     if any(c in cyrillic_chars for c in query.lower()):
#         query_lang = "uk"
    
#     # Якщо мова статті співпадає з мовою запиту або стаття англійською
#     if language and ((language == query_lang) or (language == "en" and query_lang != "en")):
#         return 0.1
    
#     return 0.0