from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, ForeignKey, String, Text, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    # Index for email
    __table_args__ = (
        Index('idx_users_email', 'email'),
    )

    # Relationships
    sessions = relationship("Session", back_populates="user")
    images = relationship("Image", back_populates="user")
    inspection_results = relationship("InspectionResult", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    user_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    user = relationship("User", back_populates="sessions")
    images = relationship("Image", back_populates="session", cascade="all, delete-orphan")
    inspection_results = relationship("InspectionResult", back_populates="session", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_url = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    user = relationship("User", back_populates="images")
    session = relationship("Session", back_populates="images")
    inspection_results = relationship("InspectionResult", back_populates="image", cascade="all, delete-orphan")


class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    image_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    results = Column(JSONB, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    user = relationship("User", back_populates="inspection_results")
    session = relationship("Session", back_populates="inspection_results")
    image = relationship("Image", back_populates="inspection_results")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(PostgreSQLUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    user = relationship("User", back_populates="conversations")
    session = relationship("Session", back_populates="conversations")


# Pydantic models for API responses
class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    id: UUID
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
