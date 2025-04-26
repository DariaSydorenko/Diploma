from firebase_admin import auth, credentials, initialize_app
import firebase_admin
import os
from dotenv import load_dotenv

# Завантаження змінних з .env
load_dotenv()

# Ініціалізація Firebase Admin
firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_cred_path)
    initialize_app(cred)

# Функція перевірки токену
def verify_token(id_token: str):
    decoded_token = auth.verify_id_token(id_token)
    return decoded_token
