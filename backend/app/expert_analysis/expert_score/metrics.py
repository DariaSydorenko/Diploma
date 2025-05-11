# from typing import List, Dict, Any
# import numpy as np
# import torch
# from app.models.article import Article
# from app.config.settings import CURRENT_YEAR
# from app.utils.text_processing import extract_abstract_text

# def get_author_influence_score(authorships: List[Dict[str, Any]]) -> float:
#     """Обчислює вплив авторів на основі їх h-індексу, affiliated_institutions та кількості публікацій."""
#     if not authorships:
#         return 0.0
    
#     score = 0.0
#     for author in authorships:
#         # Базовий бал за автора
#         author_score = 0.1
        
#         # Враховуємо h-індекс автора, якщо він є
#         author_h_index = author.get("author", {}).get("h_index", 0)
#         if author_h_index:
#             author_score += min(author_h_index / 50, 1.0) * 0.3
        
#         # Враховуємо інституції
#         institutions = author.get("institutions", [])
#         if institutions:
#             institution_count = len(institutions)
#             author_score += min(institution_count * 0.05, 0.2)
            
#             # Перевіряємо престижність інституцій (можна розширити)
#             for inst in institutions:
#                 if inst.get("type") == "education" and inst.get("display_name"):
#                     author_score += 0.05
        
#         # Враховуємо роль автора (якщо головний автор)
#         if author.get("author_position") in ["first", "corresponding"]:
#             author_score *= 1.5
            
#         score += author_score
    
#     # Нормалізуємо фінальний бал
#     return min(score, 1.0)

# def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
#     """Оцінює релевантність концептів статті до запиту."""
#     if not concepts:
#         return 0.0
    
#     query_terms = set(query.lower().split())
#     score = 0.0
    
#     for concept in concepts:
#         concept_name = concept.get("display_name", "").lower()
        
#         # Перевірка на пряме співпадіння з запитом
#         for term in query_terms:
#             if term in concept_name:
#                 score += 0.15 * concept.get("score", 0)
        
#         # Врахування загального рівня концепту
#         level = concept.get("level", 0)
#         if level == 0:  # Найвищий рівень концептів
#             score += 0.1 * concept.get("score", 0)
#         elif level == 1:
#             score += 0.05 * concept.get("score", 0)
#         else:
#             score += 0.025 * concept.get("score", 0)
    
#     return min(score, 1.0)

# def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
#     """Оцінює якість цитування з урахуванням віку публікації."""
#     if not cited_by_count or not publication_year:
#         return 0.0
    
#     years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
#     # Обчислюємо середню кількість цитувань за рік
#     citations_per_year = cited_by_count / years_since_publication
#     print(min(0.3 * np.log1p(citations_per_year) / np.log1p(10), 0.3))
    
#     # Логарифмічна шкала для зменшення впливу статей з дуже високим цитуванням
#     return min(0.3 * np.log1p(citations_per_year) / np.log1p(10), 0.3)

# def get_keyword_match_score(keywords: List[Any], query: str) -> float:
#     """Оцінює збіг ключових слів статті із запитом."""
#     if not keywords:
#         return 0.0
    
#     query_terms = set(query.lower().split())
#     matches = 0
    
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
            
#         for term in query_terms:
#             if term in keyword_lower:
#                 matches += 1
    
#     return min(matches * 0.1, 0.3)

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

from typing import List, Dict, Any
import numpy as np
from app.models.article import Article
from app.config.settings import CURRENT_YEAR
from app.utils.text_processing import extract_abstract_text
from app.config.settings import PRESTIGIOUS_NAMES

def get_author_influence_score(authorships: List[Dict[str, Any]]) -> float:
    """Обчислює вплив авторів на основі їх affiliated_institutions та ролі."""
    if not authorships:
        return 0.0

    position_priority = {"first": 0, "middle": 1}
    sorted_authors = sorted(authorships, key=lambda a: position_priority.get(a.get("author_position", ""), 999))

    top_author_score = 0.0
    additional_authors_score = 0.0

    for i, authorship in enumerate(sorted_authors):
        institutions = authorship.get("institutions", [])
        author_position = authorship.get("author_position")

        # Базовий бал залежно від позиції
        if i == 0:
            author_score = 0.3
        elif i == 1:
            author_score = 0.25
        else:
            author_score = 0.15

        # Інституції
        if institutions:
            institution_count = len(institutions)
            author_score += min(institution_count * 0.08, 0.25)

            for inst in institutions:
                inst_name = inst.get("display_name", "").lower()
                if inst.get("type") == "education" and inst_name:
                    if any(univ in inst_name for univ in PRESTIGIOUS_NAMES):
                        author_score += 0.2
                    else:
                        author_score += 0.05
                elif inst.get("type") == "facility" and inst_name:
                    author_score += 0.08

        # Підсилення за позицію
        if author_position in ["first", "middle"]:
            author_score *= 1.6

        # Додавання до загального результату
        if i == 0:
            top_author_score = author_score
        else:
            additional_authors_score += author_score * (0.9 ** (i - 1))

    score = top_author_score + 0.5 * min(additional_authors_score, 1.0)
    return min(score, 1.0)

def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
    """Оцінює релевантність концептів статті до запиту з покращеною диференціацією."""
    if not concepts:
        return 0.0
    
    query_terms = set(query.lower().split())
    score = 0.0
    match_found = False
    
    # Сортуємо концепти за рівнем (щоб спочатку обробити найвищі рівні)
    # та за оцінкою (щоб спочатку обробити найбільш релевантні концепти)
    sorted_concepts = sorted(concepts, key=lambda c: (c.get("level", 999), -c.get("score", 0)))
    
    for concept in sorted_concepts:
        concept_name = concept.get("display_name", "").lower()
        concept_score = concept.get("score", 0)
        level = concept.get("level", 0)
        
        # Перевірка на пряме співпадіння з запитом
        term_match = False
        for term in query_terms:
            if term in concept_name:
                # Значно збільшуємо бал за пряме співпадіння з запитом
                term_match = True
                match_found = True
                direct_match_score = 0.3 * concept_score
                score += direct_match_score
        
        # Додатковий бонус за повне співпадіння
        if concept_name in query_terms:
            score += 0.2
            match_found = True
        
        # Врахування загального рівня концепту (лише якщо не було прямого співпадіння)
        # Використовуємо експоненційне затухання за рівнями
        level_factor = np.exp(-level * 0.5)  # e^0 = 1 для level=0, ~0.6 для level=1, ~0.37 для level=2
        
        # Додаємо бал на основі рівня навіть якщо немає прямого співпадіння
        score += 0.2 * concept_score * level_factor
    
    # Бонус, якщо знайдено хоча б одне співпадіння (заохочує статті з будь-яким релевантним концептом)
    if match_found:
        score *= 1.2
    
    return min(score, 1.0)

def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
    """Оцінює якість цитування з урахуванням віку публікації з покращеною диференціацією."""
    if not cited_by_count or not publication_year:
        return 0.0
    
    years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
    # Обчислюємо середню кількість цитувань за рік
    citations_per_year = cited_by_count / years_since_publication
    
    # Використовуємо більш розтягнуту логарифмічну шкалу для кращої диференціації
    # Збільшуємо максимальний бал до 1.0 замість 0.3 для ширшого діапазону
    return min(np.log1p(citations_per_year) / np.log1p(30), 1.0)

def get_keyword_match_score(keywords: List[Any], query: str) -> float:
    """Оцінює збіг ключових слів статті із запитом з покращеною диференціацією."""
    if not keywords:
        return 0.0
    
    query_terms = set(query.lower().split())
    
    # Відстеження збігів для кожного ключового слова в запиті
    term_matches = {term: 0 for term in query_terms}
    exact_matches = 0
    partial_matches = 0
    
    # Розширена оцінка для кожного ключового слова
    for keyword in keywords:
        # Обробляємо випадок, коли keyword може бути словником
        if isinstance(keyword, dict):
            # Шукаємо потрібні поля у словнику (display_name, name або value)
            keyword_text = keyword.get('display_name', keyword.get('name', keyword.get('value', '')))
            if isinstance(keyword_text, str):
                keyword_lower = keyword_text.lower()
            else:
                continue
        elif isinstance(keyword, str):
            keyword_lower = keyword.lower()
        else:
            continue
        
        # Перевірка на точний збіг (коли ключове слово повністю співпадає з терміном запиту)
        if keyword_lower in query_terms:
            exact_matches += 1
            term_matches[keyword_lower] += 2  # Даємо більшу вагу точним збігам
            continue
            
        # Перевірка на частковий збіг (коли термін запиту міститься в ключовому слові)
        for term in query_terms:
            if term in keyword_lower:
                partial_matches += 1
                term_matches[term] += 1
                break
    
    # Відсоток знайдених термінів у запиті (використовуємо для розрахунку повноти збігу)
    terms_covered = sum(1 for term, count in term_matches.items() if count > 0)
    coverage_ratio = terms_covered / len(query_terms) if query_terms else 0
    
    # Обчислюємо загальний бал на основі кількості та якості збігів
    base_score = 0.4 * min(exact_matches * 0.2, 0.6) + 0.2 * min(partial_matches * 0.1, 0.4)
    
    # Додатковий бонус за широке охоплення термінів запиту
    coverage_bonus = coverage_ratio * 0.4
    
    # Комбінуємо бали
    final_score = base_score + coverage_bonus
    
    return min(final_score, 1.0)

def get_language_relevance(language: str, query: str) -> float:
    """Оцінює релевантність мови статті відносно запиту."""
    # Визначаємо мову запиту (спрощений підхід)
    query_lang = "en"  # За замовчуванням англійська
    
    # Спрощена логіка визначення мови запиту
    cyrillic_chars = set('абвгдеєжзиіїйклмнопрстуфхцчшщьюя')
    if any(c in cyrillic_chars for c in query.lower()):
        query_lang = "uk"
    
    # Якщо мова статті співпадає з мовою запиту або стаття англійською
    if language and ((language == query_lang) or (language == "en" and query_lang != "en")):
        return 0.1
    
    return 0.0
