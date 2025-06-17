import os
from dotenv import load_dotenv

# Завантаження змінних з .env файлу
load_dotenv()

# === Firebase ===
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")

# === PostgreSQL ===
DATABASE_URL = os.getenv("DATABASE_URL")
