from sqlalchemy import Column, Integer, String, ForeignKey
from backend.app.database.database import Base

class UserArticle(Base):
    __tablename__ = "user_articles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)  # Firebase UID
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
