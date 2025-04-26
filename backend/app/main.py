from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.firebase import verify_token
from app.firebase import router as firebase_router

app = FastAPI()

app.include_router(firebase_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/profile")
async def get_profile(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Missing token")

    id_token = auth_header.split(" ")[1]
    user_data = verify_token(id_token)
    if not user_data:
        raise HTTPException(status_code=403, detail="Invalid token")

    return {"email": user_data.get("email")}
