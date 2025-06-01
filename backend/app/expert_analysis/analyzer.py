from typing import List, Dict, Any
from app.models.article import Article
from app.expert_analysis.expert_score.similarity import compute_semantic_similarity, compute_query_similarity
from app.expert_analysis.expert_score.compute_expert_score import compute_expert_score
from app.config.settings import (
    MODEL_NAME,
    MIN_ARTICLES_THRESHOLD,
    MIN_RELEVANCE_THRESHOLD
)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME)

def is_results_insufficient(articles: List[Any], query: str = None) -> bool:
    if len(articles) < MIN_ARTICLES_THRESHOLD:
        print(f"Недостатньо статей: {len(articles)} < {MIN_ARTICLES_THRESHOLD}")
        return True

    avg_similarity = compute_query_similarity(query, articles)
    if avg_similarity < MIN_RELEVANCE_THRESHOLD:
        print(f"Низька релевантність статей до запиту: {avg_similarity:.2f} < {MIN_RELEVANCE_THRESHOLD}")
        return True

    return False

def analyze_articles(
        articles: List[Any], 
        query: str, 
        top_k: int, 
        raw_data: bool = True,
        ) -> List[Dict[str, Any]]:
    if not articles:
        return []

    article_objects = []
    for article_data in articles:
        if isinstance(article_data, dict):
            article_objects.append(Article.from_dict(article_data))
        elif hasattr(article_data, 'openalex_id'):
            article_objects.append(article_data)
        else:
            print(f"Невідомий тип даних статті: {type(article_data)}")
            continue

    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Обчислення оцінки для кожної статті
    scored_articles = []
    for article in article_objects:
        try:
            semantic_similarity = compute_semantic_similarity(article, query_embedding)
            expert_score = compute_expert_score(article, semantic_similarity, query)
            scored_articles.append((article, expert_score))
        except Exception as e:
            print(f"Помилка при оцінці статті: {e}")
    
    # Сортування за оцінкою в спадному порядку
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


