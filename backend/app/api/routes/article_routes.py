from fastapi import APIRouter, Request, HTTPException
# from app.services import some_service
from app.firebase.firebase import verify_token
from app.firebase.firebase import verify_token as firebase_verify_token

router = APIRouter()