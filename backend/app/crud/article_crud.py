from requests import Session
from app.models.article import Article
from app.database.database import SessionLocal

from app.models.article import Article

# def create_article(db, article_data):
#     db_article = Article(**article_data)
#     db.add(db_article)
#     db.commit()
#     db.refresh(db_article)
#     return db_article

def create_article(db: Session, article_data: dict):
    """
    Створює статтю в базі даних.
    
    Args:
        db: сесія бази даних
        article_data: словник з даними статті
        
    Returns:
        Створений об'єкт Article
    """
    # Перевіряємо, що article_data - це словник
    if not isinstance(article_data, dict):
        raise TypeError("article_data повинен бути словником")
        
    # Перевіряємо, чи вже існує стаття з таким ID
    openalex_id = article_data.get("openalex_id")
    if openalex_id:
        existing = db.query(Article).filter(Article.openalex_id == openalex_id).first()
        if existing:
            return existing
            
    # Конвертуємо embedding в список або numpy масив, якщо це тензор
    if "embedding" in article_data and hasattr(article_data["embedding"], "cpu"):
        article_data["embedding"] = article_data["embedding"].cpu().numpy()
        
    # Створюємо новий об'єкт Article
    article = Article(**article_data)
    db.add(article)
    db.commit()
    db.refresh(article)
    return article

