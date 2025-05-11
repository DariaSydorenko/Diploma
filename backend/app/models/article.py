from sqlalchemy import Column, Integer, String, Text, JSON, Boolean, Date
from app.database.database import Base
from sqlalchemy.types import PickleType
from typing import Dict, Any
from datetime import datetime, date

# class Article(Base):
#     __tablename__ = "articles"

#     id = Column(Integer, primary_key=True, index=True)
#     openalex_id = Column(String, unique=True, index=True)
#     doi = Column(String, nullable=True, index=True)
#     display_name = Column(Text)
#     publication_year = Column(Integer)
#     updated_date = Column(Date)

#     abstract_inverted_index = Column(JSON, nullable=True)
#     cited_by_count = Column(Integer, default=0)
#     concepts = Column(JSON, nullable=True)
#     keywords = Column(JSON, nullable=True)
#     open_access = Column(JSON, nullable=True)
#     has_fulltext = Column(Boolean, default=False)
#     is_retracted = Column(Boolean, default=False)
    
#     authorships = Column(JSON, nullable=True)
#     language = Column(String, nullable=True)
#     referenced_works = Column(JSON, nullable=True)
#     related_works = Column(JSON, nullable=True)

#     embedding = Column(PickleType)

#     def __init__(self, data: Dict[str, Any]):
#         self.openalex_id = data.get("openalex_id")
#         self.doi = data.get("doi")
#         self.display_name = data.get("display_name")
#         self.publication_year = data.get("publication_year")
#         self.updated_date = data.get("updated_date")
#         self.abstract_inverted_index = data.get("abstract_inverted_index")
#         self.cited_by_count = data.get("cited_by_count")
#         self.concepts = data.get("concepts")
#         self.keywords = data.get("keywords")
#         self.open_access = data.get("open_access")
#         self.has_fulltext = data.get("has_fulltext")
#         self.is_retracted = data.get("is_retracted")
#         self.authorships = data.get("authorships")
#         self.language = data.get("language")
#         self.referenced_works = data.get("referenced_works")
#         self.related_works = data.get("related_works")
#         self.embedding = data.get("embedding")
#         # self.relevance_score = data.get("relevance_score", 0)

#     def to_dict(self) -> Dict[str, Any]:
#         """Конвертує об'єкт статті назад у словник для збереження в БД"""
#         return {
#             "openalex_id": self.openalex_id,
#             "doi": self.doi,
#             "display_name": self.display_name,
#             "publication_year": self.publication_year,
#             "updated_date": self.updated_date,
#             "abstract_inverted_index": self.abstract_inverted_index,
#             "cited_by_count": self.cited_by_count,
#             "concepts": self.concepts,
#             "keywords": self.keywords,
#             "open_access": self.open_access,
#             "has_fulltext": self.has_fulltext,
#             "is_retracted": self.is_retracted,
#             "authorships": self.authorships,
#             "language": self.language,
#             "referenced_works": self.referenced_works,
#             "related_works": self.related_works,
#             "embedding": self.embedding,
#             # "relevance_score": self.relevance_score
#         }

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    openalex_id = Column(String, unique=True, index=True)
    doi = Column(String, nullable=True, index=True)
    display_name = Column(Text)
    publication_year = Column(Integer)
    updated_date = Column(Date)

    abstract_inverted_index = Column(JSON, nullable=True)
    cited_by_count = Column(Integer, default=0)
    concepts = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    open_access = Column(JSON, nullable=True)
    has_fulltext = Column(Boolean, default=False)
    is_retracted = Column(Boolean, default=False)
    
    authorships = Column(JSON, nullable=True)  # Зберігаємо оригінальний список авторів
    authors = Column(JSON, nullable=True)
    language = Column(String, nullable=True)
    referenced_works = Column(JSON, nullable=True)
    related_works = Column(JSON, nullable=True)

    embedding = Column(PickleType)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Article":
        """Створює об'єкт Article з dict."""
        return cls(
            openalex_id=data.get("openalex_id"),
            doi=data.get("doi"),
            display_name=data.get("display_name"),
            publication_year=data.get("publication_year"),
            updated_date=data.get("updated_date") or datetime.utcnow().date(),
            abstract_inverted_index=data.get("abstract_inverted_index"),
            cited_by_count=data.get("cited_by_count", 0),
            concepts=data.get("concepts"),
            keywords=data.get("keywords"),
            open_access=data.get("open_access"),
            has_fulltext=data.get("has_fulltext", False),
            is_retracted=data.get("is_retracted", False),
            authorships=data.get("authorships"),
            language=data.get("language"),
            referenced_works=data.get("referenced_works"),
            related_works=data.get("related_works"),
            embedding=data.get("embedding")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Повертає словник зі значеннями полів статті для API."""
        # Витягуємо авторів з authorships
        authors = []
        if self.authorships:
            for entry in self.authorships:
                if isinstance(entry, dict):
                    author_info = entry.get("author", {})
                    name = author_info.get("display_name")
                    if name:
                        authors.append({"name": name})

        return {
            "id": self.id,
            "openalex_id": self.openalex_id,
            "doi": self.doi,
            "display_name": self.display_name,
            "publication_year": self.publication_year,
            "updated_date": self.updated_date,
            "abstract_inverted_index": self.abstract_inverted_index,
            "cited_by_count": self.cited_by_count,
            "concepts": self.concepts,
            "keywords": self.keywords,
            "open_access": self.open_access,
            "has_fulltext": self.has_fulltext,
            "is_retracted": self.is_retracted,
            "authorships": self.authorships,
            "authors": authors,  # ← Додаткове поле
            "language": self.language,
            "referenced_works": self.referenced_works,
            "related_works": self.related_works,
            "embedding": self.embedding
        }