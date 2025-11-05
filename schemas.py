#!/usr/bin/env python3
"""
Core schemas for Sakura's agentic tool calling system
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import time

class ToolStatus(str, Enum):
    """Possible tool execution statuses"""
    EXECUTED = "executed"      # Tool ran successfully
    REVOKED = "revoked"        # Tool disabled by config
    FORBIDDEN = "forbidden"    # Permission denied (not whitelisted, ToS violation)
    RATE_LIMITED = "rate_limited"  # Rate limit exceeded
    INVALID_ARGS = "invalid_args"  # Bad arguments provided
    ERROR = "error"            # Tool execution failed

class ToolCall(BaseModel):
    """LLM's request to invoke a tool"""
    name: str = Field(..., description="Tool name: DM, SEARCH, or NONE")
    args: Dict[str, Any] = Field(..., description="Tool-specific arguments")
    confidence: float = Field(..., ge=0.0, le=1.0, description="LLM's confidence in needing this tool")
    reasoning: str = Field(..., description="Why this tool is needed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "DM",
                "args": {
                    "user_id": "123456789012345678",
                    "message": "I'll tell you privately about that sensitive topic..."
                },
                "confidence": 0.85,
                "reasoning": "User requested private conversation about sensitive content"
            }
        }

class ToolResponse(BaseModel):
    """Result of tool execution attempt"""
    ok: bool = Field(..., description="Whether tool execution was successful")
    tool: str = Field(..., description="Tool that was attempted")
    status: ToolStatus = Field(..., description="Execution status")
    reason: str = Field(..., description="Human-readable explanation")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: float = Field(default_factory=time.time, description="When this response was generated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ok": True,
                "tool": "DM",
                "status": "executed",
                "reason": "Direct message sent successfully",
                "meta": {"message_id": "987654321098765432", "user_id": "123456789012345678"}
            }
        }

class ToolContext(BaseModel):
    """Context for tool execution including user and channel info"""
    user_id: str = Field(..., description="Discord user ID")
    user_name: str = Field(..., description="Discord username")
    channel_id: str = Field(..., description="Discord channel ID")
    channel_type: str = Field(..., description="public, private, or voice")
    guild_id: Optional[str] = Field(None, description="Discord guild ID if applicable")
    is_whitelisted: bool = Field(False, description="Whether user is in DM whitelist")
    user_cooldowns: Dict[str, float] = Field(default_factory=dict, description="User's current cooldowns")
    voice_members: Optional[List[Dict[str, str]]] = Field(None, description="List of members in voice channel if applicable")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "123456789012345678",
                "user_name": "Eric",
                "channel_id": "876543210987654321",
                "channel_type": "public",
                "guild_id": "111111111111111111",
                "is_whitelisted": True,
                "user_cooldowns": {"dm": 0.0, "search": 0.0}
            }
        }

class RouterDecision(BaseModel):
    """Router's decision about what tool to use"""
    action: str = Field(..., description="Action: DM, SEARCH, or CHAT")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    reasoning: str = Field(..., description="Why this action was chosen")
    latency_ms: int = Field(..., description="Decision latency in milliseconds")
    method: str = Field(..., description="Router method used")
    tool_call: Optional[ToolCall] = Field(None, description="Tool call if action requires tools")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "DM",
                "confidence": 0.92,
                "reasoning": "User explicitly requested private conversation",
                "latency_ms": 0,
                "method": "neuro_fast",
                "tool_call": {
                    "name": "DM",
                    "args": {"user_id": "123456789012345678", "message": "I'll message you privately"},
                    "confidence": 0.92,
                    "reasoning": "User requested private conversation"
                }
            }
        }
