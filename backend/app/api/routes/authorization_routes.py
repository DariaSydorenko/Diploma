from fastapi import APIRouter, Request, HTTPException
from app.firebase.firebase import verify_token
from app.firebase.firebase import verify_token as firebase_verify_token

router = APIRouter()

@router.get("/profile")
async def get_profile(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Missing token")

    id_token = auth_header.split(" ")[1]
    user_data = verify_token(id_token)
    if not user_data:
        raise HTTPException(status_code=403, detail="Invalid token")

    return {"email": user_data.get("email")}

@router.post("/verify-token")
async def verify_user_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="❌ Відсутній токен")

    try:
        id_token = auth_header.split(" ")[1]
        decoded_token = firebase_verify_token(id_token)
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