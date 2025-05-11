from sqlalchemy.orm import Session
from app.models.article import Article as DbArticle

def apply_filters(q, year=None, min_citations=None):
    q = q.filter(DbArticle.embedding.isnot(None))
    if year:
        q = q.filter(DbArticle.publication_year == year)
    if min_citations:
        q = q.filter(DbArticle.cited_by_count >= min_citations)
    return q


async def check_existing_ids(db, openalex_ids):
    """Пакетна перевірка існуючих ID в базі даних"""
    if not openalex_ids:
        return set()
    
    existing = db.query(DbArticle.openalex_id).filter(
        DbArticle.openalex_id.in_(openalex_ids)
    ).all()
    
    return {record[0] for record in existing}