from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.future import select

from app.models.database import User, UserResponse
from app.services.database_service import get_db_session
from app.services.auth_service import (
    get_password_hash,
    verify_password,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

class UserSignup(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

@router.post("/register", response_model=TokenResponse)
async def register_user(user_data: UserSignup, db=Depends(get_db_session)):
    try:
        # Check if email already exists
        email_result = await db.execute(select(User).where(User.email == user_data.email))
        existing_email_user = email_result.scalar_one_or_none()
        
        if existing_email_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The email '{user_data.email}' is already registered. Please use a different email or try logging in.",
            )

        # Check if username already exists
        username_result = await db.execute(select(User).where(User.username == user_data.username))
        existing_username_user = username_result.scalar_one_or_none()
        
        if existing_username_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The username '{user_data.username}' is already taken. Please choose a different username.",
            )

        # Hash the password and create a user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Create token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(new_user.id), "email": new_user.email},
            expires_delta=access_token_expires
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse.model_validate(new_user)
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        import traceback
        print(f"Registration error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration error: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(user_data: UserLogin, db=Depends(get_db_session)):
    try:
        # Find user by email
        result = await db.execute(select(User).where(User.email == user_data.email))
        user = result.scalar_one_or_none()

        if not user or not verify_password(user_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email},
            expires_delta=access_token_expires
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserResponse.model_validate(user)
        }
    except HTTPException:
        # Re-raise HTTP exceptions (like auth failures)
        raise
    except Exception as e:
        # Log unexpected errors
        import traceback
        print(f"Login error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )

@router.get("/check-email")
async def check_email_availability(email: str, db=Depends(get_db_session)):
    """Check if email is already registered"""
    try:
        result = await db.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()
        
        return {"exists": existing_user is not None}
    except Exception as e:
        print(f"Error checking email availability: {e}")
        return {"exists": False}

@router.get("/check-username")
async def check_username_availability(username: str, db=Depends(get_db_session)):
    """Check if username is already taken"""
    try:
        result = await db.execute(select(User).where(User.username == username))
        existing_user = result.scalar_one_or_none()
        
        return {"exists": existing_user is not None}
    except Exception as e:
        print(f"Error checking username availability: {e}")
        return {"exists": False}
