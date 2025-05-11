from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import date

class ArticleSchema(BaseModel):
    id: int
    openalex_id: str
    doi: Optional[str]
    display_name: Optional[str] = 'Назва відсутня'
    publication_year: int
    updated_date: Optional[date]

    abstract_inverted_index: Optional[Dict[str, List[int]]]
    cited_by_count: Optional[int]
    concepts: Optional[List[Dict[str, Any]]]
    keywords: Optional[List[Dict[str, Any]]]
    open_access: Optional[Dict[str, Any]]
    has_fulltext: Optional[bool]
    is_retracted: Optional[bool]

    authorships: Optional[List[Dict[str, Any]]]
    authors: Optional[List[Dict[str, str]]] = None
    language: Optional[str]
    referenced_works: Optional[List[str]]
    related_works: Optional[List[str]]

    embedding: Optional[List[float]]

    model_config = {
        "from_attributes": True
    }
