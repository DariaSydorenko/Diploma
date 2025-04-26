from fastapi import APIRouter, Request, HTTPException
from firebase_admin import auth, credentials, initialize_app
import firebase_admin
import os
from dotenv import load_dotenv

# Завантаження змінних з .env
load_dotenv()

# Отримання шляху до ключа з .env
firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")

# Ініціалізація Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred_path)
    initialize_app(cred)

router = APIRouter()

@router.post("/api/auth/verify-token")
async def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="❌ Відсутній токен")

    try:
        id_token = auth_header.split(" ")[1]
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token["uid"]
        email = decoded_token.get("email")

        return {
            "uid": uid,
            "email": email,
            "name": decoded_token.get("name", ""),
            "message": "✅ Токен валідний",
        }

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"❌ Токен недійсний: {str(e)}")
