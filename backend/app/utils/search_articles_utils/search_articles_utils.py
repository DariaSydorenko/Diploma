from app.models.article import Article
from sqlalchemy.orm import Session

def apply_filters(db_query, query: str, year: int = None, min_citations: int = None):
    q = db_query.filter(Article.display_name.ilike(f"%{query}%"))
    if year:
        q = q.filter(Article.publication_year == year)
    if min_citations:
        q = q.filter(Article.cited_by_count >= min_citations)
    return q

def store_new_openalex_articles(db: Session, results: list):
    from app.crud.article_crud import create_article

    for work in results:
        openalex_id = work.get("id")
        if not openalex_id:
            continue
        if db.query(Article).filter_by(openalex_id=openalex_id).first():
            continue

        create_article(db, {
            "openalex_id": openalex_id,
            "doi": work.get("doi"),
            "display_name": work.get("display_name"),
            "publication_year": work.get("publication_year"),
            "updated_date": work.get("updated_date"),
            "abstract_inverted_index": work.get("abstract_inverted_index"),
            "cited_by_count": work.get("cited_by_count"),
            "concepts": work.get("concepts"),
            "keywords": work.get("keywords"),
            "open_access": work.get("open_access"),
            "has_fulltext": work.get("has_fulltext"),
            "is_retracted": work.get("is_retracted"),
            "authorships": work.get("authorships"),
            "language": work.get("language"),
            "referenced_works": work.get("referenced_works"),
            "related_works": work.get("related_works"),
        })