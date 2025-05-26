import numpy as np
from app.expert_analysis.expert_score.metrics import (
    get_author_influence_score,
    get_concept_relevance_score,
    get_citation_quality_score,
    get_keyword_match_score,
    get_language_relevance
)
from app.expert_analysis.expert_score.similarity import compute_semantic_similarity
from app.config.settings import CURRENT_YEAR
from app.models.article import Article

def compute_expert_score(article: Article, similarity: float, query: str) -> float:
    """Розрахунок комплексної експертної оцінки релевантності статті з розширенням діапазону."""
    raw_score = 0.0
    weights = {
        'semantic': 0.20,
        'citation': 0.20,
        'recency': 0.10,
        'concepts': 0.15,
        'keywords': 0.15,
        'authorship': 0.10,
        'accessibility': 0.05,
        'language': 0.05,
    }

    # 1. Семантична подібність
    # raw_score += similarity * weights['semantic']
    raw_score += similarity

    # 2. Цитування
    if getattr(article, 'cited_by_count', None) and getattr(article, 'publication_year', None):
        raw_score += get_citation_quality_score(article.cited_by_count, article.publication_year) * weights['citation']

    # 3. Актуальність
    if getattr(article, 'publication_year', None):
        recency_score = max(0.0, 1.0 - max(0, (CURRENT_YEAR - article.publication_year)) / 10)
        raw_score += recency_score * weights['recency']

    # 4. Концепти
    if getattr(article, 'concepts', None):
        raw_score += get_concept_relevance_score(article.concepts, query) * weights['concepts']

    # 5. Ключові слова
    if getattr(article, 'keywords', None):
        raw_score += get_keyword_match_score(article.keywords, query) * weights['keywords']

    # 6. Автори
    if getattr(article, 'authorships', None):
        raw_score += get_author_influence_score(article.authorships) * weights['authorship']

    # 7. Доступність
    accessibility_score = 0.0
    if getattr(article, "open_access", None) and article.open_access.get("is_oa"):
        accessibility_score += 0.6
    if getattr(article, "has_fulltext", False):
        accessibility_score += 0.4
    raw_score += min(accessibility_score, 1.0) * weights['accessibility']

    # 8. Мова
    if getattr(article, "language", None):
        raw_score += get_language_relevance(article.language, query) * weights['language']

    # 9. Штраф за відкликану статтю
    if getattr(article, "is_retracted", False):
        raw_score *= 0.2

    # 10. Розширення діапазону — сигмоїда
    # Параметр 10 — крутість функції, 0.5 — центр
    score = 1 / (1 + np.exp(-10 * (raw_score - 0.5)))
    return round(float(score), 4)
