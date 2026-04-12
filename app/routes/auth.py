# app/routes/auth.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from datetime import datetime, timedelta
from uuid import UUID
import jwt
import os

from app.models.database import User, UserResponse
from app.services.database_service import async_session

router = APIRouter()

# ── Config ────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Pydantic Schemas ──────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


# ── Helpers ───────────────────────────────────────────────
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: UUID, email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired, please login again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.id == UUID(user_id))
        )
        user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")

    return user


# ── Endpoints ─────────────────────────────────────────────
@router.post("/auth/register", response_model=LoginResponse)
async def register(req: RegisterRequest):
    async with async_session() as session:

        # Check email already exists
        existing_email = await session.execute(
            select(User).where(User.email == req.email)
        )
        if existing_email.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already registered")

        # Check username already exists
        existing_username = await session.execute(
            select(User).where(User.username == req.username)
        )
        if existing_username.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Username already taken")

        # Validate inputs
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        if len(req.username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

        # Create user
        new_user = User(
            email=req.email,
            username=req.username,
            hashed_password=hash_password(req.password)
        )
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

    token = create_access_token(new_user.id, new_user.email)
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.from_orm(new_user)
    )


@router.post("/auth/login", response_model=LoginResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    async with async_session() as session:
        # Allow login with email or username
        result = await session.execute(
            select(User).where(
                (User.email == form.username) | (User.username == form.username)
            )
        )
        user = result.scalar_one_or_none()

    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")

    token = create_access_token(user.id, user.email)
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.from_orm(user)
    )


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)


@router.get("/auth/check-email")
async def check_email(email: str):
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.email == email)
        )
        exists = result.scalar_one_or_none() is not None
        return {"exists": exists}


@router.get("/auth/check-username")
async def check_username(username: str):
    async with async_session() as session:
        result = await session.execute(
            select(User).where(User.username == username)
        )
        exists = result.scalar_one_or_none() is not None
        return {"exists": exists}