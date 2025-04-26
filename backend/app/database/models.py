from sqlalchemy import Column, Integer, String, Text, Date, ARRAY, TIMESTAMP
from .database import Base
from sqlalchemy.sql import func

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    openalex_id = Column(String, unique=True, index=True)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    authors = Column(ARRAY(String))
    publication_date = Column(Date)
    citation_count = Column(Integer)
    concepts = Column(ARRAY(String))
    doi = Column(String)
    created_at = Column(TIMESTAMP, server_default=func.now())
