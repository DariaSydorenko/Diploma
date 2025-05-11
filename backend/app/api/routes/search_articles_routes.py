from fastapi import APIRouter, Query, Depends, Request
from sqlalchemy.orm import Session
from typing import List, Optional
import aiohttp
import asyncio
from datetime import datetime, date
import time
import concurrent.futures
import torch.nn.functional as F

from sentence_transformers import util
from app.database.database import get_db
from app.models.article import Article as DbArticle
from app.schemas.article_schema import ArticleSchema
from app.crud.article_crud import create_article
# from app.utils.expert_analysis import analyze_articles, is_results_insufficient, Article
from app.expert_analysis.analyzer import analyze_articles, is_results_insufficient
from app.services.openalex_service import parallel_fetch_openalex
from app.services.embedding_service import semantic_sort, batch_encode_embeddings
from app.services.database_service import apply_filters, check_existing_ids

router = APIRouter()

@router.get("/", response_model=List[ArticleSchema])
async def search_articles(
    request: Request,
    query: str = Query(..., min_length=2),
    year: Optional[int] = None,
    min_citations: Optional[int] = None,
    db: Session = Depends(get_db),
    top_k: int = 5
):
    start_time = time.time()
    model = request.app.state.model

    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding = F.normalize(query_embedding, p=2, dim=0)

    # Пошук у локальній БД
    db_query = apply_filters(db.query(DbArticle), year, min_citations)
    local_articles = db_query.all()

    top_articles = semantic_sort(local_articles, query_embedding, top_k)

    if not is_results_insufficient(top_articles, query=query):
        print(f"Знайдено {len(top_articles)} релевантних статей у базі даних за {time.time() - start_time:.2f} сек")
        return top_articles  # Повертаємо статті, які вже є в базі даних

    print("Локальних результатів недостатньо, шукаємо в OpenAlex...")

    # Параметри запиту до OpenAlex
    base_url = "https://api.openalex.org/works"
    filters = []
    if year:
        filters.append(f"from_publication_date:{year}-01-01")
        filters.append(f"to_publication_date:{year}-12-31")
    if min_citations:
        filters.append(f"cited_by_count:>{min_citations}")
    filter_str = ",".join(filters)

    params = {
        "search": query,
        "per-page": 200,
        "select": "id,doi,display_name,publication_year,updated_date,"
                  "abstract_inverted_index,cited_by_count,concepts,keywords,"
                  "open_access,has_fulltext,is_retracted,"
                  "authorships,language,referenced_works,related_works"
    }
    if filter_str:
        params["filter"] = filter_str

    # Паралельне отримання даних з OpenAlex
    max_results = 500
    all_results = await parallel_fetch_openalex(query, base_url, params, max_results)

    if not all_results:
        print("Не знайдено нових статей в OpenAlex")
        return []

    openalex_ids = [work[1].get("id") for work in all_results if work[1].get("id")]
    existing_ids = await check_existing_ids(db, openalex_ids)
    
    # Фільтруємо результати, виключаючи ті, що вже є в БД
    filtered_results = []
    for text, work in all_results:
        openalex_id = work.get("id")
        if openalex_id and openalex_id not in existing_ids:
            filtered_results.append((text, work))
    
    all_results = filtered_results

    if not all_results:
        print("Всі знайдені статті вже є в базі даних")
        return top_articles

    # Обробка отриманих результатів оптимізованим способом
    batch_texts = [item[0] for item in all_results]
    metadata = [item[1] for item in all_results]

    # Кодування ембедінгів маленькими партіями для оптимізації пам'яті та швидкості
    embeddings = await batch_encode_embeddings(model, batch_texts)
    
    # Підготовка даних для аналізу
    candidate_articles = []
    for i, (embedding, work) in enumerate(zip(embeddings, metadata)):
        if embedding is None:
            continue
            
        try:
            updated_date_str = work.get("updated_date")
            updated_date = datetime.fromisoformat(updated_date_str).date() if updated_date_str else None

            article_data = {
                "openalex_id": work.get("id"),
                "doi": work.get("doi"),
                "display_name": work.get("display_name"),
                "publication_year": work.get("publication_year"),
                "updated_date": updated_date,
                "abstract_inverted_index": work.get("abstract_inverted_index"),
                "cited_by_count": work.get("cited_by_count"),
                "concepts": work.get("concepts"),
                "keywords": work.get("keywords"),
                "open_access": work.get("open_access"),
                "has_fulltext": work.get("has_fulltext"),
                "is_retracted": work.get("is_retracted"),
                "authorships": work.get("authorships"),
                "authors": [
                    {"name": a["author"]["display_name"]}
                    for a in work.get("authorships", [])
                    if "author" in a and "display_name" in a["author"]
                ],
                "language": work.get("language"),
                "referenced_works": work.get("referenced_works"),
                "related_works": work.get("related_works"),
                "embedding": embedding
            }

            candidate_articles.append(article_data)
        except Exception as e:
            print(f"Помилка при обробці статті: {e}")

    # Аналіз статей експертною системою перед додаванням до БД
    analyzed_candidates = analyze_articles(candidate_articles, query=query, top_k=top_k, raw_data=True)

    new_article_ids = []
    for article_data in analyzed_candidates:
        try:
            article = create_article(db, article_data)
            new_article_ids.append(article.id)
        except Exception as e:
            print(f"Помилка при створенні article: {e}")

    # Отримуємо оновлений список статей з бази даних і сортуємо для фінального результату
    db_query = apply_filters(db.query(DbArticle).filter(DbArticle.id.in_(new_article_ids)))
    filtered_articles = db_query.all()
    final_articles = semantic_sort(filtered_articles, query_embedding, top_k)

    print(f"Загальний час виконання: {time.time() - start_time:.2f} сек")
    return final_articles