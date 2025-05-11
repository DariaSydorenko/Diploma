from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.database.database import engine
from app.models import article
from app.database.database import SessionLocal
import logging

from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

article.Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Завантаження моделі SentenceTransformer...")
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    # app.state.model = SentenceTransformer("all-mpnet-base-v2")
    logging.info("Модель успішно завантажена.")
    
    yield

    logging.info("Завершення роботи програми.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
