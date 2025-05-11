from fastapi import APIRouter

from app.api.routes.search_articles_routes import router as search_articles_router
from .authorization_routes import router as authorization_router

router = APIRouter()

router.include_router(search_articles_router, prefix="/search_articles", tags=["Search_articles"])
router.include_router(authorization_router, prefix="/auth", tags=["Authorization"])
