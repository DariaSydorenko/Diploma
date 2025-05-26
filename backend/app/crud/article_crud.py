from requests import Session
from app.models.article import Article
from app.database.database import SessionLocal

from app.models.article import Article

def create_article(db: Session, article_data: dict):
    if not isinstance(article_data, dict):
        raise TypeError("article_data повинен бути словником")

    openalex_id = article_data.get("openalex_id")
    if openalex_id:
        existing = db.query(Article).filter(Article.openalex_id == openalex_id).first()
        if existing:
            return existing

    if "embedding" in article_data and hasattr(article_data["embedding"], "cpu"):
        article_data["embedding"] = article_data["embedding"].cpu().numpy()

    article = Article(**article_data)
    db.add(article)
    db.commit()
    db.refresh(article)
    return article

