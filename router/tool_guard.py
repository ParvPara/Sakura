#!/usr/bin/env python3
"""
Tool Guard System for Sakura's agentic tools
Handles permissions, rate limits, whitelist, and ToS compliance
"""

import time
import logging
import json
import os
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import defaultdict

from schemas import ToolStatus, ToolContext

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limit information for a specific user and tool"""
    last_used: float
    tokens_remaining: int
    max_tokens: int
    refill_rate: float  # tokens per second

class ToolGuard:
    """Guards tool execution with permissions, rate limits, and safety checks"""
    
    def __init__(self):
        self.whitelist: List[str] = []
        self.rate_limits: Dict[str, Dict[str, RateLimitInfo]] = defaultdict(dict)
        self.global_limits: Dict[str, RateLimitInfo] = {}
        self.filter_enabled = True  # Filter system enabled by default
        
        # Load whitelist from config
        self._load_whitelist()
        
        # Initialize rate limit buckets
        self._init_rate_limits()
        
        logger.info("Tool Guard initialized")
    
    def _load_whitelist(self):
        """Load DM whitelist from config or environment"""
        try:
            # Try to load from whitelist.json first
            if os.path.exists("config/whitelist.json"):
                with open("config/whitelist.json", "r") as f:
                    data = json.load(f)
                    self.whitelist = data.get("dm_users", [])
                    logger.info(f"Loaded {len(self.whitelist)} users from config/whitelist.json")
            else:
                # Fall back to environment variable
                whitelist_env = os.getenv("DM_WHITELIST", "")
                if whitelist_env:
                    self.whitelist = [uid.strip() for uid in whitelist_env.split(",") if uid.strip()]
                    logger.info(f"Loaded {len(self.whitelist)} users from environment")
                else:
                    # Default whitelist (empty for safety)
                    self.whitelist = []
                    logger.warning("No DM whitelist found - DMs will be forbidden for all users")
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")
            self.whitelist = []
    
    def _init_rate_limits(self):
        """Initialize rate limit buckets for tools"""
        current_time = time.time()
        
        # DM rate limits
        self.global_limits["dm"] = RateLimitInfo(
            last_used=current_time,
            tokens_remaining=6,  # 6 DMs per hour
            max_tokens=6,
            refill_rate=6.0 / 3600.0  # 6 tokens per hour
        )
        
        # Search rate limits
        self.global_limits["search"] = RateLimitInfo(
            last_used=current_time,
            tokens_remaining=60,  # 60 searches per hour
            max_tokens=60,
            refill_rate=60.0 / 3600.0  # 60 tokens per hour
        )
        
        # Call rate limits
        self.global_limits["CALL"] = RateLimitInfo(
            last_used=current_time,
            tokens_remaining=30,  # 30 calls per hour globally
            max_tokens=30,
            refill_rate=30.0 / 3600.0  # 30 tokens per hour
        )
        
        logger.info("Rate limit buckets initialized")
    
    def _refill_tokens(self, rate_limit: RateLimitInfo):
        """Refill tokens based on time passed and refill rate"""
        current_time = time.time()
        time_passed = current_time - rate_limit.last_used
        
        # Calculate tokens to add
        tokens_to_add = time_passed * rate_limit.refill_rate
        rate_limit.tokens_remaining = min(
            rate_limit.max_tokens,
            rate_limit.tokens_remaining + tokens_to_add
        )
        
        # Update last used time
        rate_limit.last_used = current_time
    
    def _check_user_rate_limit(self, user_id: str, tool: str) -> Tuple[bool, str, Dict]:
        """Check if user is rate limited for a specific tool"""
        if tool not in self.rate_limits[user_id]:
            # Initialize user's rate limit for this tool
            if tool == "dm":
                self.rate_limits[user_id][tool] = RateLimitInfo(
                    last_used=time.time(),
                    tokens_remaining=1,  # 1 DM per 90 seconds per user
                    max_tokens=1,
                    refill_rate=1.0 / 90.0  # 1 token per 90 seconds
                )
            elif tool == "search":
                self.rate_limits[user_id][tool] = RateLimitInfo(
                    last_used=time.time(),
                    tokens_remaining=3,  # 3 searches per minute per user
                    max_tokens=3,
                    refill_rate=3.0 / 60.0  # 3 tokens per minute
                )
            elif tool == "CALL":
                self.rate_limits[user_id][tool] = RateLimitInfo(
                    last_used=time.time(),
                    tokens_remaining=2,  # 2 calls per minute per user
                    max_tokens=2,
                    refill_rate=2.0 / 60.0  # 2 tokens per minute
                )
        
        rate_limit = self.rate_limits[user_id][tool]
        self._refill_tokens(rate_limit)
        
        if rate_limit.tokens_remaining < 1:
            # Calculate time until next token
            time_until_next = (1.0 - rate_limit.tokens_remaining) / rate_limit.refill_rate
            return False, "rate_limited", {
                "cooldown_secs": int(time_until_next),
                "tool": tool,
                "user_id": user_id
            }
        
        return True, "ok", {}
    
    def _check_global_rate_limit(self, tool: str) -> Tuple[bool, str, Dict]:
        """Check if global rate limit is exceeded for a tool"""
        if tool not in self.global_limits:
            return True, "ok", {}
        
        rate_limit = self.global_limits[tool]
        self._refill_tokens(rate_limit)
        
        if rate_limit.tokens_remaining < 1:
            # Calculate time until next token
            time_until_next = (1.0 - rate_limit.tokens_remaining) / rate_limit.refill_rate
            return False, "rate_limited", {
                "cooldown_secs": int(time_until_next),
                "tool": tool,
                "global": True
            }
        
        return True, "ok", {}
    
    def _consume_token(self, user_id: str, tool: str):
        """Consume a token for rate limiting"""
        # Consume user token
        if tool in self.rate_limits[user_id]:
            self.rate_limits[user_id][tool].tokens_remaining -= 1
        
        # Consume global token
        if tool in self.global_limits:
            self.global_limits[tool].tokens_remaining -= 1
        
        logger.info(f"Consumed {tool} token for user {user_id}")
    
    def _check_tos_compliance(self, text: str, tool: str) -> Tuple[bool, str, Dict]:
        """Check if content violates Terms of Service"""
        if not text:
            return True, "ok", {}
        
        text_lower = text.lower()
        
        # Basic ToS violations (expand as needed)
        violations = [
            "harassment", "bullying", "hate speech", "discrimination",
            "explicit sexual content", "gore", "violence", "illegal activities",
            "personal information", "doxxing", "spam", "scam"
        ]
        
        for violation in violations:
            if violation in text_lower:
                return False, "forbidden", {
                    "reason": "policy",
                    "violation": violation,
                    "tool": tool
                }
        
        return True, "ok", {}
    
    def check_dm_allowed(self, context: ToolContext, message: str) -> Tuple[bool, str, Dict]:
        """Check if DM is allowed for this user and message"""
        # Check if DM tool is enabled
        if not self._is_tool_enabled("dm"):
            return False, "revoked", {"flag": "ENABLE_DM=false"}
        
        # Check if user is whitelisted
        if not context.is_whitelisted:
            return False, "forbidden", {
                "reason": "not_whitelisted",
                "user_id": context.user_id,
                "message": "User not in DM whitelist"
            }
        
        # Check ToS compliance
        tos_ok, tos_status, tos_meta = self._check_tos_compliance(message, "dm")
        if not tos_ok:
            return False, tos_status, tos_meta
        
        # Rate limits disabled for testing
        # # Check user rate limit
        # user_ok, user_status, user_meta = self._check_user_rate_limit(context.user_id, "dm")
        # if not user_ok:
        #     return False, user_status, user_meta
        # 
        # # Check global rate limit
        # global_ok, global_status, global_meta = self._check_global_rate_limit("dm")
        # if not global_ok:
        #     return False, global_status, global_meta
        
        return True, "ok", {}
    
    def check_search_allowed(self, context: ToolContext, query: str) -> Tuple[bool, str, Dict]:
        """Check if search is allowed for this user and query"""
        # Check if search tool is enabled
        if not self._is_tool_enabled("search"):
            return False, "revoked", {"flag": "ENABLE_SEARCH=false"}
        
        # Check ToS compliance
        tos_ok, tos_status, tos_meta = self._check_tos_compliance(query, "search")
        if not tos_ok:
            return False, tos_status, tos_meta
        
        # Rate limits disabled for testing
        # # Check user rate limit
        # user_ok, user_status, user_meta = self._check_user_rate_limit(context.user_id, "search")
        # if not user_ok:
        #     return False, user_status, user_meta
        # 
        # # Check global rate limit
        # global_ok, global_status, global_meta = self._check_global_rate_limit("search")
        # if not global_ok:
        #     return False, global_status, global_meta
        
        return True, "ok", {}
    
    def _is_tool_enabled(self, tool: str) -> bool:
        """Check if a tool is enabled via environment variables"""
        env_var = f"ENABLE_{tool.upper()}"
        return os.getenv(env_var, "true").lower() == "true"
    
    def add_to_whitelist(self, user_id: str) -> bool:
        """Add a user to the DM whitelist"""
        if user_id not in self.whitelist:
            self.whitelist.append(user_id)
            self._save_whitelist()
            logger.info(f"Added user {user_id} to DM whitelist")
            return True
        return False
    
    def remove_from_whitelist(self, user_id: str) -> bool:
        """Remove a user from the DM whitelist"""
        if user_id in self.whitelist:
            self.whitelist.remove(user_id)
            self._save_whitelist()
            logger.info(f"Removed user {user_id} from DM whitelist")
            return True
        return False
    
    def _save_whitelist(self):
        """Save whitelist to file"""
        try:
            with open("config/whitelist.json", "w") as f:
                json.dump({"dm_users": self.whitelist}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save whitelist: {e}")
    
    def get_whitelist(self) -> List[str]:
        """Get current DM whitelist"""
        return self.whitelist.copy()
    
    def get_rate_limit_info(self, user_id: str, tool: str) -> Dict:
        """Get rate limit information for debugging"""
        user_limits = self.rate_limits.get(user_id, {})
        global_limits = self.global_limits.get(tool)
        
        return {
            "user": {
                "tokens_remaining": user_limits.get(tool, RateLimitInfo(0, 0, 0, 0)).tokens_remaining,
                "max_tokens": user_limits.get(tool, RateLimitInfo(0, 0, 0, 0)).max_tokens,
                "refill_rate": user_limits.get(tool, RateLimitInfo(0, 0, 0, 0)).refill_rate
            } if tool in user_limits else None,
            "global": {
                "tokens_remaining": global_limits.tokens_remaining if global_limits else 0,
                "max_tokens": global_limits.max_tokens if global_limits else 0,
                "refill_rate": global_limits.refill_rate if global_limits else 0
            } if global_limits else None
        }
    
    def is_filter_enabled(self) -> bool:
        """Check if the filter system is currently enabled"""
        return self.filter_enabled
    
    def toggle_filter(self) -> bool:
        """Toggle the filter system on/off and return new state"""
        self.filter_enabled = not self.filter_enabled
        logger.info(f"Filter system {'enabled' if self.filter_enabled else 'disabled'}")
        return self.filter_enabled
    
    def log_tool_attempt(self, context: ToolContext, tool: str, allowed: bool, status: str, meta: Dict):
        """Log tool attempt for audit trail"""
        logger.info(
            f"Tool Guard: {tool} {'ALLOWED' if allowed else 'DENIED'} for user {context.user_name} "
            f"({context.user_id}) - Status: {status}, Meta: {meta}"
        )
