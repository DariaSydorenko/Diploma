from typing import List, Dict, Any
import numpy as np
from app.models.article import Article
from app.config.settings import CURRENT_YEAR
from app.utils.text_processing import extract_abstract_text
from app.config.settings import PRESTIGIOUS_NAMES

def get_author_influence_score(authorships: List[Dict[str, Any]]) -> float:
    if not authorships:
        return 0.0

    top_score = 0.0
    other_score = 0.0

    for i, author in enumerate(authorships):
        # Базовий бал залежно від позиції автора у списку
        score = 0.3 if i == 0 else 0.25 if i == 1 else 0.15

        institutions = author.get("institutions", [])
        # Бонус за кількість інституцій (до 0.25 максимум)
        score += min(len(institutions) * 0.08, 0.25)

        for inst in institutions:
            name = inst.get("display_name", "").lower()
            if not name:
                continue

            # Додаткові бали за тип і престиж інституції
            if inst.get("type") == "education":
                score += 0.2 if any(u in name for u in PRESTIGIOUS_NAMES) else 0.05
            elif inst.get("type") == "facility":
                score += 0.08

        # Підсилення балу, якщо автор на важливій позиції
        if author.get("author_position") in ["first", "middle"]:
            score *= 1.6

        # Додавання до підсумку
        if i == 0:
            top_score = score
        else:
            # Зменшення ваги для авторів далі в списку
            other_score += score * (0.9 ** (i - 1))

    # Підсумковий бал з обмеженням максимуму
    return min(top_score + 0.5 * min(other_score, 1.0), 1.0)


def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
    if not concepts:
        return 0.0
    
    query_terms = set(query.lower().split())
    score = 0.0
    match_found = False

    sorted_concepts = sorted(concepts, key=lambda c: (c.get("level", 999), -c.get("score", 0)))
    
    for concept in sorted_concepts:
        concept_name = concept.get("display_name", "").lower()
        concept_score = concept.get("score", 0)
        level = concept.get("level", 0)
        
        # Перевірка на пряме співпадіння з запитом
        for term in query_terms:
            if term in concept_name:
                match_found = True
                direct_match_score = 0.3 * concept_score
                score += direct_match_score
        
        # Додатковий бонус за повне співпадіння
        if concept_name in query_terms:
            score += 0.2
            match_found = True
        
        # Врахування загального рівня концепту
        level_factor = np.exp(-level * 0.5) # Експоненційне затухання за рівнями e^0 = 1 для level=0, ~0.6 для level=1, ~0.37 для level=2

        # Додавання балу на основі рівня навіть якщо немає прямого співпадіння
        score += 0.2 * concept_score * level_factor
    
    # Бонус, якщо знайдено хоча б одне співпадіння
    if match_found:
        score *= 1.2
    
    return min(score, 1.0)


def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
    if not cited_by_count or not publication_year:
        return 0.0
    
    years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
    # Середня кількість цитувань за рік
    citations_per_year = cited_by_count / years_since_publication

    return min(np.log1p(citations_per_year) / np.log1p(30), 1.0)


def get_keyword_match_score(keywords: List[Any], query: str) -> float:
    if not keywords:
        return 0.0
    
    query_terms = set(query.lower().split())
    
    # Відстеження збігів для кожного ключового слова в запиті
    term_matches = {term: 0 for term in query_terms}
    exact_matches = 0
    partial_matches = 0
    
    # Розширена оцінка для кожного ключового слова
    for keyword in keywords:
        if isinstance(keyword, dict):
            keyword_text = keyword.get('display_name', keyword.get('name', keyword.get('value', '')))
            if isinstance(keyword_text, str):
                keyword_lower = keyword_text.lower()
            else:
                continue
        elif isinstance(keyword, str):
            keyword_lower = keyword.lower()
        else:
            continue
        
        # Перевірка на точний збіг
        if keyword_lower in query_terms:
            exact_matches += 1
            term_matches[keyword_lower] += 2
            continue
            
        # Перевірка на частковий збіг
        for term in query_terms:
            if term in keyword_lower:
                partial_matches += 1
                term_matches[term] += 1
                break
    
    # Відсоток знайдених термінів у запиті
    terms_covered = sum(1 for term, count in term_matches.items() if count > 0)
    coverage_ratio = terms_covered / len(query_terms) if query_terms else 0
    
    # Загальний бал на основі кількості та якості збігів
    base_score = 0.4 * min(exact_matches * 0.2, 0.6) + 0.2 * min(partial_matches * 0.1, 0.4)
    
    # Додатковий бонус за широке охоплення термінів запиту
    coverage_bonus = coverage_ratio * 0.4

    final_score = base_score + coverage_bonus
    
    return min(final_score, 1.0)

def get_language_relevance(language: str, query: str) -> float:
    query_lang = "en"  # За замовчуванням англійська

    cyrillic_chars = set('абвгдеєжзиіїйклмнопрстуфхцчшщьюя')
    if any(c in cyrillic_chars for c in query.lower()):
        query_lang = "uk"
    
    # Якщо мова статті співпадає з мовою запиту або стаття англійською
    if language and ((language == query_lang) or (language == "en" and query_lang != "en")):
        return 0.1
    
    return 0.0
