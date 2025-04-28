from sqlalchemy import Column, Integer, String, Text
from backend.app.database.database import Base

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    abstract = Column(Text)
    authors = Column(String)
    link = Column(String)
