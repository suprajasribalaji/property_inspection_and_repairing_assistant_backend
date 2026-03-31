"""
API endpoint for monitoring usage statistics
"""

from fastapi import APIRouter
from app.services.api_usage_tracker import usage_tracker

router = APIRouter()

@router.get("/api/usage/stats")
async def get_usage_stats():
    """Get current API usage statistics"""
    return usage_tracker.get_usage_stats()

@router.get("/api/usage/suggestions")
async def get_usage_suggestions():
    """Get optimization suggestions based on usage patterns"""
    suggestions = usage_tracker.suggest_optimization()
    return {"suggestions": suggestions}
