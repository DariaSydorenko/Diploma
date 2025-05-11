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






# def compute_expert_score(article: Article, similarity: float, query: str) -> float:
#     """Розрахунок комплексної експертної оцінки релевантності статті з покращеним розподілом балів."""
#     # Перевіримо, чи similarity знаходиться в діапазоні [0, 1]
#     similarity = max(0.0, min(similarity, 1.0))
    
#     # Ініціалізуємо ваги та загальний бал
#     weights = {
#         'semantic': 0.30,       # Підвищуємо вагу семантичної подібності
#         'citation': 0.20,       # Якість цитування
#         'recency': 0.10,        # Свіжість публікації
#         'concepts': 0.15,       # Релевантність концептів
#         'keywords': 0.15,       # Релевантність ключових слів
#         'authorship': 0.05,     # Зменшуємо вплив авторів
#         'accessibility': 0.03,  # Зменшуємо вагу доступності
#         'language': 0.02,       # Зменшуємо вагу мови
#     }
    
#     # Перевіряємо, що сума ваг дорівнює 1.0
#     assert abs(sum(weights.values()) - 1.0) < 1e-9, "Сума ваг повинна бути рівна 1.0"
    
#     # Обчислюємо бали для кожного компонента
#     scores = {}
    
#     # 1. Семантична подібність (найважливіший фактор) - застосовуємо степеневу функцію для підсилення різниці
#     scores['semantic'] = similarity ** 0.9  # Степенева функція для кращої диференціації
    
#     # 2. Якість цитування з урахуванням віку публікації - збільшуємо вплив високоцитованих статей
#     if getattr(article, 'cited_by_count', None) and getattr(article, 'publication_year', None):
#         citation_score = get_citation_quality_score(article.cited_by_count, article.publication_year)
#         # Застосовуємо нелінійну функцію, щоб підвищити значущість високих значень
#         scores['citation'] = citation_score ** 0.8
#     else:
#         scores['citation'] = 0.0
    
#     # 3. Свіжість публікації (актуальність) - робимо більший акцент на нових публікаціях
#     if getattr(article, 'publication_year', None):
#         # Використовуємо експоненційну функцію для підвищення ваги нових публікацій
#         years_difference = max(0, CURRENT_YEAR - article.publication_year)
#         scores['recency'] = np.exp(-years_difference / 3) if years_difference < 10 else 0.0
#     else:
#         scores['recency'] = 0.0
    
#     # 4. Релевантність концептів - підсилюємо вплив високорелевантних концептів
#     if getattr(article, 'concepts', None):
#         concept_score = get_concept_relevance_score(article.concepts, query)
#         scores['concepts'] = concept_score ** 0.8  # Нелінійна функція для підсилення високих оцінок
#     else:
#         scores['concepts'] = 0.0
    
#     # 5. Релевантність ключових слів - підсилюємо вплив багатьох збігів
#     if getattr(article, 'keywords', None):
#         keyword_score = get_keyword_match_score(article.keywords, query)
#         scores['keywords'] = keyword_score ** 0.7  # Нелінійна функція для підсилення високих оцінок
#     else:
#         scores['keywords'] = 0.0
    
#     # 6. Авторський вплив - вирівнюємо вплив
#     if getattr(article, 'authorships', None):
#         scores['authorship'] = get_author_influence_score(article.authorships)
#     else:
#         scores['authorship'] = 0.0
    
#     # 7. Доступність
#     accessibility_score = 0.0
#     if getattr(article, "open_access", None) and article.open_access.get("is_oa"):
#         accessibility_score += 0.7  # Підвищуємо бонус за відкритий доступ
#     if getattr(article, "has_fulltext", False):
#         accessibility_score += 0.5  # Підвищуємо бонус за повний текст
#     scores['accessibility'] = min(accessibility_score, 1.0)
    
#     # 8. Мовна релевантність
#     if getattr(article, "language", None):
#         scores['language'] = get_language_relevance(article.language, query)
#     else:
#         scores['language'] = 0.0
    
#     # Обчислюємо зважений бал
#     weighted_score = sum(scores[key] * weights[key] for key in weights)
    
#     # Застосовуємо нелінійну трансформацію для розширення діапазону оцінок
#     # Сигмоїдальна функція з параметрами, що розширюють середину діапазону
#     enhanced_score = 1.0 / (1.0 + np.exp(-10 * (weighted_score - 0.5)))
    
#     # 9. Штрафи
#     if getattr(article, "is_retracted", False):
#         enhanced_score *= 0.1  # Ще більший штраф за відкликані статті
    
#     # Нормалізуємо до діапазону [0, 1]
#     final_score = max(0.0, min(enhanced_score, 1.0))
    
#     # Для відлагодження можна додати логування
#     # print(f"Raw scores: {scores}")
#     # print(f"Weighted score: {weighted_score}, Enhanced score: {enhanced_score}, Final score: {final_score}")
    
#     return final_score