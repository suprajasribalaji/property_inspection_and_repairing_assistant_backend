import os
from typing import Optional, List
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import (
    Session, Image, InspectionResult, Conversation,
    SessionResponse, ImageResponse, InspectionResultResponse, 
    ConversationResponse, SessionHistoryResponse, Base
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Convert to async URL and handle SSL parameters
if DATABASE_URL.startswith("postgresql://"):
    # Remove sslmode parameter for asyncpg compatibility
    clean_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    # Remove sslmode parameter if present
    if "sslmode=" in clean_url:
        clean_url = clean_url.split("sslmode=")[0].rstrip("&?")
    ASYNC_DATABASE_URL = clean_url
else:
    ASYNC_DATABASE_URL = DATABASE_URL

# Create async engine
engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session():
    """Get database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


# Session operations
async def get_latest_session_by_date_hour() -> Optional[SessionResponse]:
    """Get the latest session overall (simplified logic)"""
    async with async_session() as session:
        result = await session.execute(
            select(Session)
            .order_by(Session.created_at.desc())
            .limit(1)
        )
        latest_session = result.scalar_one_or_none()
        if latest_session:
            return SessionResponse.from_orm(latest_session)
        return None

async def get_session(session_id: UUID) -> Optional[SessionResponse]:
    """Get session by ID"""
    async with async_session() as session:
        result = await session.execute(select(Session).where(Session.id == session_id))
        db_session = result.scalar_one_or_none()
        if db_session:
            return SessionResponse.from_orm(db_session)
        return None


# Image operations
async def create_image(session_id: UUID, image_url: str) -> ImageResponse:
    """Create a new image record"""
    async with async_session() as session:
        db_image = Image(session_id=session_id, image_url=image_url)
        session.add(db_image)
        await session.commit()
        await session.refresh(db_image)
        return ImageResponse.from_orm(db_image)


async def get_images_by_session(session_id: UUID) -> List[ImageResponse]:
    """Get all images for a session"""
    async with async_session() as session:
        result = await session.execute(
            select(Image).where(Image.session_id == session_id).order_by(Image.uploaded_at)
        )
        images = result.scalars().all()
        return [ImageResponse.from_orm(img) for img in images]


# Inspection result operations
async def create_inspection_result(
    session_id: UUID, 
    image_id: UUID, 
    results: dict
) -> InspectionResultResponse:
    """Create a new inspection result"""
    async with async_session() as session:
        db_result = InspectionResult(
            session_id=session_id,
            image_id=image_id,
            results=results
        )
        session.add(db_result)
        await session.commit()
        await session.refresh(db_result)
        return InspectionResultResponse.from_orm(db_result)


async def get_inspection_results_by_session(session_id: UUID) -> List[InspectionResultResponse]:
    """Get all inspection results for a session"""
    async with async_session() as session:
        result = await session.execute(
            select(InspectionResult).where(InspectionResult.session_id == session_id).order_by(InspectionResult.created_at)
        )
        results = result.scalars().all()
        return [InspectionResultResponse.from_orm(res) for res in results]


# Conversation operations
async def create_conversation(
    session_id: UUID, 
    role: str, 
    message: str
) -> ConversationResponse:
    """Create a new conversation message"""
    async with async_session() as session:
        db_conv = Conversation(
            session_id=session_id,
            role=role,
            message=message
        )
        session.add(db_conv)
        await session.commit()
        await session.refresh(db_conv)
        return ConversationResponse.from_orm(db_conv)


async def get_conversations_by_session(session_id: UUID) -> List[ConversationResponse]:
    """Get all conversations for a session"""
    async with async_session() as session:
        result = await session.execute(
            select(Conversation).where(Conversation.session_id == session_id).order_by(Conversation.created_at)
        )
        conversations = result.scalars().all()
        return [ConversationResponse.from_orm(conv) for conv in conversations]


# Session history operations
async def get_session_history(session_id: UUID) -> Optional[SessionHistoryResponse]:
    """Get complete session history including images, results, and conversations"""
    async with async_session() as session:
        # Get session
        session_result = await session.execute(select(Session).where(Session.id == session_id))
        db_session = session_result.scalar_one_or_none()
        if not db_session:
            return None

        # Get related data
        images_result = await session.execute(
            select(Image).where(Image.session_id == session_id).order_by(Image.uploaded_at)
        )
        images = images_result.scalars().all()

        results_result = await session.execute(
            select(InspectionResult).where(InspectionResult.session_id == session_id).order_by(InspectionResult.created_at)
        )
        results = results_result.scalars().all()

        conversations_result = await session.execute(
            select(Conversation).where(Conversation.session_id == session_id).order_by(Conversation.created_at)
        )
        conversations = conversations_result.scalars().all()

        return SessionHistoryResponse(
            session=SessionResponse.from_orm(db_session),
            images=[ImageResponse.from_orm(img) for img in images],
            inspection_results=[InspectionResultResponse.from_orm(res) for res in results],
            conversations=[ConversationResponse.from_orm(conv) for conv in conversations]
        )

async def create_session(user_id: UUID) -> SessionResponse:
    async with async_session() as session:
        db_session = Session(user_id=user_id)               # ← pass user_id
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        return SessionResponse.from_orm(db_session)


async def save_session_to_db(session_id: UUID, user_id: UUID) -> SessionResponse:
    async with async_session() as session:
        existing = await session.execute(select(Session).where(Session.id == session_id))
        existing_session = existing.scalar_one_or_none()
        if existing_session:
            return SessionResponse.from_orm(existing_session)

        db_session = Session(id=session_id, user_id=user_id) # ← pass user_id
        session.add(db_session)
        await session.commit()
        await session.refresh(db_session)
        return SessionResponse.from_orm(db_session)


async def get_latest_session_with_results(user_id: UUID) -> Optional[SessionResponse]:
    async with async_session() as session:
        try:
            result = await session.execute(
                select(InspectionResult)
                .join(Session, InspectionResult.session_id == Session.id)
                .where(Session.user_id == user_id)          # ← filter by user
                .order_by(InspectionResult.created_at.desc())
                .limit(1)
            )
            latest_result = result.scalar_one_or_none()
            if latest_result:
                session_result = await session.execute(
                    select(Session).where(Session.id == latest_result.session_id)
                )
                latest_session = session_result.scalar_one_or_none()
                if latest_session:
                    return SessionResponse.from_orm(latest_session)
            return None
        except Exception as e:
            print(f"Error in get_latest_session_with_results: {e}")
            return None


async def get_all_sessions(user_id: UUID) -> List[SessionResponse]:
    async with async_session() as session:
        result = await session.execute(
            select(Session)
            .where(Session.user_id == user_id)              # ← filter by user
            .order_by(Session.created_at.desc())
        )
        sessions = result.scalars().all()
        return [SessionResponse.from_orm(sess) for sess in sessions]