"""
API Usage Tracker for monitoring and optimizing Gemini API usage
"""

from datetime import datetime, timedelta
from typing import Dict, List
import json

class APIUsageTracker:
    def __init__(self):
        self.usage_log: List[Dict] = []
        self.daily_usage = 0
        self.last_reset = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.max_daily_requests = 1500  # Default Gemini free tier daily limit is 1500
    
    def log_api_call(self, endpoint: str, success: bool, response_time: float = 0):
        """Log an API call for tracking purposes"""
        now = datetime.now()
        
        # Reset daily counter if needed
        if now.date() > self.last_reset.date():
            self.daily_usage = 0
            self.last_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        self.daily_usage += 1
        
        log_entry = {
            "timestamp": now.isoformat(),
            "endpoint": endpoint,
            "success": success,
            "response_time": response_time,
            "daily_usage": self.daily_usage
        }
        
        self.usage_log.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.usage_log) > 100:
            self.usage_log = self.usage_log[-100:]
    
    def can_make_request(self) -> tuple[bool, str]:
        """Check if we can make a request based on usage patterns"""
        now = datetime.now()
        
        # Reset daily counter if needed
        if now.date() > self.last_reset.date():
            self.daily_usage = 0
            self.last_reset = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if self.daily_usage >= self.max_daily_requests:
            reset_time = self.last_reset + timedelta(days=1)
            wait_hours = (reset_time - now).total_seconds() / 3600
            return False, f"Daily limit reached. Resets in {wait_hours:.1f} hours."
        
        # Check if we're making too many requests in a short time
        recent_calls = [log for log in self.usage_log[-15:] 
                       if (now - datetime.fromisoformat(log["timestamp"])).total_seconds() < 60]
        
        if len(recent_calls) >= 12:  # Gemini free tier RPM is 15
            return False, "Rate limit: Too many requests in quick succession. Please wait."
        
        return True, "Request allowed"
    
    def get_usage_stats(self) -> Dict:
        """Get current usage statistics"""
        now = datetime.now()
        today_calls = [log for log in self.usage_log 
                      if datetime.fromisoformat(log["timestamp"]).date() == now.date()]
        
        successful_calls = sum(1 for log in today_calls if log["success"])
        avg_response_time = sum(log["response_time"] for log in today_calls) / len(today_calls) if today_calls else 0
        
        return {
            "daily_usage": self.daily_usage,
            "max_daily_requests": self.max_daily_requests,
            "successful_calls_today": successful_calls,
            "total_calls_today": len(today_calls),
            "success_rate": successful_calls / len(today_calls) if today_calls else 0,
            "average_response_time": avg_response_time,
            "remaining_calls": self.max_daily_requests - self.daily_usage,
            "next_reset": (self.last_reset + timedelta(days=1)).isoformat()
        }
    
    def suggest_optimization(self) -> List[str]:
        """Suggest optimizations based on usage patterns"""
        suggestions = []
        now = datetime.now()
        
        # Check for high failure rate
        recent_calls = [log for log in self.usage_log[-10:] 
                       if (now - datetime.fromisoformat(log["timestamp"])).total_seconds() < 3600]
        
        if recent_calls:
            failure_rate = sum(1 for log in recent_calls if not log["success"]) / len(recent_calls)
            if failure_rate > 0.3:
                suggestions.append("High failure rate detected. Consider implementing better error handling.")
        
        # Check for slow responses
        slow_calls = [log for log in recent_calls if log["response_time"] > 10]
        if len(slow_calls) > len(recent_calls) * 0.5:
            suggestions.append("Many slow responses detected. Consider optimizing prompts or using caching.")
        
        # Check daily usage pattern
        if self.daily_usage > self.max_daily_requests * 0.8:
            suggestions.append("Approaching daily limit. Consider upgrading to a paid plan for more requests.")
        
        return suggestions

# Global instance
usage_tracker = APIUsageTracker()
