from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr
from sqlalchemy import Column, DateTime, ForeignKey, String, Text, Boolean, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# ── Users Table ─────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    username = Column(String(255), nullable=True)
    hashed_password = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")

# ── Session Table ─────────────────────────────────────
class Session(Base):
    __tablename__ = "sessions"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(                                          # ← NEW
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,   # nullable=True for backward compat with existing sessions
        index=True
    )
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")     # ← NEW
    images = relationship("Image", back_populates="session", cascade="all, delete-orphan")
    inspection_results = relationship("InspectionResult", back_populates="session", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")

# ── Image Table ─────────────────────────────────────
class Image(Base):
    __tablename__ = "images"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    image_url = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="images")
    inspection_results = relationship("InspectionResult", back_populates="image", cascade="all, delete-orphan")

# ── Inspection Result Table ─────────────────────────────────────
class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    results = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="inspection_results")
    image = relationship("Image", back_populates="inspection_results")

# ── Conversation Table ─────────────────────────────────────
class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="conversations")

    __table_args__ = (
        CheckConstraint("role IN ('user', 'ai')", name="check_role"),
    )


# ── Pydantic Response Models ──────────────────────────────

class UserResponse(BaseModel):                                
    id: UUID
    email: str
    username: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID] = None                             
    created_at: datetime

    class Config:
        from_attributes = True


class ImageResponse(BaseModel):
    id: UUID
    session_id: UUID
    image_url: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class InspectionResultResponse(BaseModel):
    id: UUID
    session_id: UUID
    image_id: UUID
    results: dict
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    message: str
    created_at: datetime

    class Config:
        from_attributes = True


class SessionHistoryResponse(BaseModel):
    session: SessionResponse
    images: list[ImageResponse]
    inspection_results: list[InspectionResultResponse]
    conversations: list[ConversationResponse]