from datetime import datetime, timedelta
import os
from typing import Optional
from passlib.context import CryptContext
import jwt
from fastapi import Request, HTTPException
from app.models.database import User

# Secret key to sign JWT token
# In production, this should be a secure random SECRET_KEY generated and stored in .env
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-super-secret-jwt-key-for-development")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days defaults

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

async def get_current_user(request: Request) -> Optional[User]:
    """Extract current user from JWT token in request headers"""
    try:
        # Get Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        
        # Extract token from "Bearer <token>" format
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        
        token = parts[1]
        
        # Decode token
        payload = decode_access_token(token)
        if not payload:
            return None
        
        # Get user ID from token
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Fetch user from database
        from app.services.database_service import async_session, select, User
        async with async_session() as session:
            result = await session.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            return user
            
    except Exception as e:
        print(f"Error getting current user: {e}")
        return None
