from typing import List, Dict, Any
from app.models.article import Article
from app.expert_analysis.expert_score.similarity import compute_semantic_similarity, compute_query_similarity
from app.expert_analysis.expert_score.compute_expert_score import compute_expert_score
from app.config.settings import (
    MODEL_NAME,
    MIN_ARTICLES_THRESHOLD,
    SIMILARITY_YEAR_SPAN,
    CURRENT_YEAR,
    MIN_RELEVANCE_THRESHOLD,
    QUERY_LENGTH_THRESHOLD
)
from fastapi import Request
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')

def is_results_insufficient(articles: List[Any], query: str = None) -> bool:
    """
    Перевіряє, чи достатньо результатів для повноцінного аналізу.
    
    Args:
        articles: список статей
        query: пошуковий запит (опціонально)
        
    Returns:
        True, якщо результатів недостатньо або вони нерелевантні
    """

    # Перевірка 1: Мінімальна кількість статей
    if len(articles) < MIN_ARTICLES_THRESHOLD:
        print(f"Недостатньо статей: {len(articles)} < {MIN_ARTICLES_THRESHOLD}")
        return True
    
    # Якщо запит не надано, ми не можемо перевірити релевантність
    if not query:
        # Перевірка 2: Різноманіття років публікації (тільки якщо запит не надано)
        years = [a.publication_year for a in articles if getattr(a, 'publication_year', None)]
        if years and len(years) >= 2 and (max(years) - min(years) <= SIMILARITY_YEAR_SPAN):
            print(f"Недостатнє різноманіття років: {min(years)}-{max(years)}")
            return True
    else:
        # Перевірка 3: Релевантність статей до запиту
        avg_similarity = compute_query_similarity(query, articles)
        if avg_similarity < MIN_RELEVANCE_THRESHOLD:
            print(f"Низька релевантність статей до запиту: {avg_similarity:.2f} < {MIN_RELEVANCE_THRESHOLD}")
            return True
        
        # Якщо релевантність висока, не перевіряємо різноманіття років
        # (релевантність важливіша за різноманіття)
        print(f"Висока релевантність статей до запиту: {avg_similarity:.2f}")
        return False
    
    # Якщо всі перевірки пройдені успішно
    return False

def analyze_articles(
        articles: List[Any], 
        query: str, 
        top_k: int, 
        raw_data: bool = True,
        ) -> List[Dict[str, Any]]:
    """
    Аналізує список статей і повертає top_k найбільш релевантних до запиту.
    
    Args:
        articles: список статей (словники або об'єкти Article)
        query: пошуковий запит
        top_k: кількість статей для повернення
        raw_data: якщо True, повертає словники замість об'єктів Article
        
    Returns:
        List[Dict[str, Any]]: список словників з даними статей, якщо raw_data=True
        або список об'єктів Article, якщо raw_data=False
    """
    if not articles:
        return []

    # Перевіряємо тип даних і перетворюємо відповідно
    article_objects = []
    for article_data in articles:
        if isinstance(article_data, dict):
            article_objects.append(Article.from_dict(article_data))
        elif hasattr(article_data, 'openalex_id'):  # Це вже об'єкт Article
            article_objects.append(article_data)
        else:
            print(f"Невідомий тип даних статті: {type(article_data)}")
            continue
    
    # Створюємо вектор запиту
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Обчислюємо оцінки для кожної статті
    scored_articles = []
    for article in article_objects:
        try:
            semantic_similarity = compute_semantic_similarity(article, query_embedding)
            expert_score = compute_expert_score(article, semantic_similarity, query)
            scored_articles.append((article, expert_score))
        except Exception as e:
            print(f"Помилка при оцінці статті: {e}")
    
    # Сортуємо за оцінкою в спадному порядку
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    
    filtered_articles = [(a, s) for a, s in scored_articles if s >= MIN_RELEVANCE_THRESHOLD]
    top_filtered = filtered_articles[:top_k]

    # Вивід у консоль результатів
    print(f"\n=== Результати аналізу для запиту: \"{query}\" ===")
    if not top_filtered:
        print("Не знайдено жодної релевантної статті вище порогу.")
    else:
        for idx, (article, score) in enumerate(top_filtered, 1):
            print(f"{idx}. {article.display_name} — Score: {score:.4f}")

    if raw_data:
        return [article.to_dict() for article, _ in top_filtered]
    else:
        return [article for article, _ in top_filtered]


