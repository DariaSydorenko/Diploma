from fastapi import APIRouter

from .article_routes import router as article_router
from .authorization_routes import router as authorization_router
from .recent_request_routes import router as recent_request_router
from .user_article_routes import router as user_article_router

router = APIRouter()

router.include_router(article_router, prefix="/articles", tags=["Articles"])
router.include_router(authorization_router, prefix="/auth", tags=["Authorization"])
router.include_router(recent_request_router, prefix="/recent_request", tags=["History"])
router.include_router(user_article_router, prefix="/my_articles", tags=["UsersArticles"])
