#!/usr/bin/env python3
"""
Enhanced DM Tool for Sakura's agentic system
Integrates with tool guard for permissions, rate limits, and safety
"""

import logging
import discord
from typing import Dict, Any, Optional
from dataclasses import dataclass
import re

from schemas import ToolResponse, ToolStatus, ToolContext
from router.tool_guard import ToolGuard

logger = logging.getLogger(__name__)

@dataclass
class DMResult:
    """Result of DM attempt"""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    status: str = "unknown"

class DMTool:
    """Enhanced DM tool with guard integration"""
    
    def __init__(self, bot, tool_guard: ToolGuard):
        self.bot = bot
        self.tool_guard = tool_guard
        logger.info("DM Tool initialized with guard integration")
    
    async def execute_dm(self, context: ToolContext, message: str, target_person: Optional[str] = None) -> ToolResponse:
        """Execute DM with full guard checks and execution"""
        try:
            # Handle person-specific DM routing
            if target_person:
                return await self._execute_person_dm(context, message, target_person)
            
            # Check if DM is allowed through tool guard
            allowed, status, meta = self.tool_guard.check_dm_allowed(context, message)
            
            if not allowed:
                # Log the denial
                self.tool_guard.log_tool_attempt(context, "DM", False, status, meta)
                
                return ToolResponse(
                    ok=False,
                    tool="DM",
                    status=ToolStatus(status),
                    reason=self._get_denial_reason(status, meta),
                    meta=meta
                )
            
            # DM is allowed - attempt to send
            dm_result = await self._send_dm_message(context, message)
            
            if dm_result.success:
                # Consume rate limit token
                self.tool_guard._consume_token(context.user_id, "dm")
                
                # Log successful execution
                self.tool_guard.log_tool_attempt(context, "DM", True, "executed", {
                    "message_id": dm_result.message_id,
                    "user_id": context.user_id
                })
                
                return ToolResponse(
                    ok=True,
                    tool="DM",
                    status=ToolStatus.EXECUTED,
                    reason="Direct message sent successfully",
                    meta={
                        "message_id": dm_result.message_id,
                        "user_id": context.user_id,
                        "user_name": context.user_name
                    }
                )
            else:
                # DM failed to send
                self.tool_guard.log_tool_attempt(context, "DM", False, "error", {
                    "error": dm_result.error,
                    "user_id": context.user_id
                })
                
                return ToolResponse(
                    ok=False,
                    tool="DM",
                    status=ToolStatus.ERROR,
                    reason=f"Failed to send DM: {dm_result.error}",
                    meta={
                        "error": dm_result.error,
                        "user_id": context.user_id
                    }
                )
                
        except Exception as e:
            logger.error(f"DM tool execution error: {e}")
            
            return ToolResponse(
                ok=False,
                tool="DM",
                status=ToolStatus.ERROR,
                reason=f"DM tool error: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def _execute_person_dm(self, context: ToolContext, message: str, target_person: str) -> ToolResponse:
        """Execute DM to a specific person mentioned in voice"""
        try:
            # Find the target person's user ID
            target_user_id = await self._find_person_user_id(target_person)
            
            if not target_user_id:
                return ToolResponse(
                    ok=False,
                    tool="DM",
                    status=ToolStatus.ERROR,
                    reason=f"Could not find Discord user for '{target_person}'",
                    meta={"target_person": target_person, "search_attempted": True}
                )
            
            # Create context for the target person
            target_context = ToolContext(
                user_id=target_user_id,
                user_name=target_person,
                channel_id=context.channel_id,
                channel_type="dm",
                guild_id=context.guild_id,
                is_whitelisted=True,  # Allow person-specific DMs
                user_cooldowns={}
            )
            
            # Send the DM to the target person
            dm_result = await self._send_dm_message(target_context, message)
            
            if dm_result.success:
                # Log successful person DM
                self.tool_guard.log_tool_attempt(context, "DM", True, "executed", {
                    "target_person": target_person,
                    "target_user_id": target_user_id,
                    "message_id": dm_result.message_id
                })
                
                return ToolResponse(
                    ok=True,
                    tool="DM",
                    status=ToolStatus.EXECUTED,
                    reason=f"Message sent to {target_person} successfully",
                    meta={
                        "target_person": target_person,
                        "target_user_id": target_user_id,
                        "message_id": dm_result.message_id,
                        "dm_status": dm_result.status
                    }
                )
            else:
                return ToolResponse(
                    ok=False,
                    tool="DM",
                    status=ToolStatus.ERROR,
                    reason=f"Failed to send message to {target_person}: {dm_result.error}",
                    meta={
                        "target_person": target_person,
                        "target_user_id": target_user_id,
                        "error": dm_result.error,
                        "dm_status": dm_result.status
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in person DM to {target_person}: {e}")
            return ToolResponse(
                ok=False,
                tool="DM",
                status=ToolStatus.ERROR,
                reason=f"Unexpected error contacting {target_person}: {str(e)}",
                meta={"target_person": target_person, "error": str(e)}
            )
    
    async def _find_person_user_id(self, person_name: str) -> Optional[str]:
        """Find Discord user ID for a person name using people.json and Discord search"""
        try:
            # First try to resolve real name to Discord username via people.json
            discord_username = await self._resolve_real_name_to_discord_username(person_name)
            if discord_username:
                logger.info(f"Resolved real name '{person_name}' to Discord username '{discord_username}'")
                # Look for the Discord username in all guilds
                for guild in self.bot.guilds:
                    for member in guild.members:
                        if (member.name.lower() == discord_username.lower() or 
                            member.display_name.lower() == discord_username.lower()):
                            return str(member.id)
            
            # If no people.json match, search through bot's cached users directly
            for guild in self.bot.guilds:
                for member in guild.members:
                    if (member.name.lower() == person_name.lower() or 
                        member.display_name.lower() == person_name.lower()):
                        return str(member.id)
            
            logger.warning(f"Could not find Discord user for person: {person_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding user ID for {person_name}: {e}")
            return None
    
    async def _resolve_real_name_to_discord_username(self, real_name: str) -> Optional[str]:
        """Resolve a real name to Discord username using people.json"""
        try:
            import json
            import os
            
            people_file = "data/people.json"
            if not os.path.exists(people_file):
                logger.warning(f"People file {people_file} not found")
                return None
            
            with open(people_file, 'r', encoding='utf-8') as f:
                people_data = json.load(f)
            
            # Check for exact name match (case insensitive)
            for person_name, person_data in people_data.items():
                if person_name.lower() == real_name.lower():
                    discord_username = person_data.get('discord_username')
                    if discord_username:
                        logger.info(f"Found Discord username '{discord_username}' for real name '{real_name}'")
                        return discord_username
                
                # Also check aliases
                aliases = person_data.get('aliases', [])
                for alias in aliases:
                    if alias.lower() == real_name.lower():
                        discord_username = person_data.get('discord_username')
                        if discord_username:
                            logger.info(f"Found Discord username '{discord_username}' for alias '{real_name}'")
                            return discord_username
            
            logger.info(f"No Discord username found for real name '{real_name}' in people.json")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving real name to Discord username: {e}")
            return None
    
    async def _send_dm_message(self, context: ToolContext, message: str) -> DMResult:
        """Actually send the DM message via Discord"""
        try:
            # PRE-SEND GUARDRAILS: Check for filtered words before sending
            filtered_message = self._check_and_clean_dm_content(message)
            if not filtered_message:
                return DMResult(
                    success=False,
                    error="Message content contains inappropriate language and was blocked",
                    status="content_filtered"
                )
            
            # Get the user object
            user = await self.bot.fetch_user(int(context.user_id))
            if not user:
                return DMResult(
                    success=False,
                    error="User not found",
                    status="user_not_found"
                )
            
            # Check if user accepts DMs
            try:
                # Try to send the clean DM
                dm_message = await user.send(filtered_message)
                
                return DMResult(
                    success=True,
                    message_id=str(dm_message.id),
                    status="sent"
                )
                
            except discord.Forbidden:
                return DMResult(
                    success=False,
                    error="User has DMs disabled",
                    status="dms_disabled"
                )
            except discord.HTTPException as e:
                return DMResult(
                    success=False,
                    error=f"Discord API error: {e}",
                    status="api_error"
                )
                
        except Exception as e:
            logger.error(f"Error sending DM to {context.user_id}: {e}")
            return DMResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                status="unknown_error"
            )
    
    def _get_denial_reason(self, status: str, meta: Dict[str, Any]) -> str:
        """Get human-readable reason for DM denial"""
        if status == "revoked":
            return "DM tool is currently disabled"
        elif status == "forbidden":
            if meta.get("reason") == "not_whitelisted":
                return "User not in DM whitelist"
            elif meta.get("reason") == "policy":
                return f"Content violates ToS: {meta.get('violation', 'unknown')}"
            else:
                return "DM not allowed for this user"
        elif status == "rate_limited":
            cooldown = meta.get("cooldown_secs", 0)
            if meta.get("global"):
                return f"Global DM rate limit exceeded. Try again in {cooldown} seconds"
            else:
                return f"User DM rate limit exceeded. Try again in {cooldown} seconds"
        else:
            return f"DM denied: {status}"
    
    async def get_dm_status(self, context: ToolContext) -> Dict[str, Any]:
        """Get DM status and rate limit information for a user"""
        try:
            # Check if user is whitelisted
            is_whitelisted = context.user_id in self.tool_guard.get_whitelist()
            
            # Get rate limit info
            rate_limit_info = self.tool_guard.get_rate_limit_info(context.user_id, "dm")
            
            # Check if DM tool is enabled
            dm_enabled = self.tool_guard._is_tool_enabled("dm")
            
            return {
                "user_id": context.user_id,
                "user_name": context.user_name,
                "dm_enabled": dm_enabled,
                "is_whitelisted": is_whitelisted,
                "rate_limits": rate_limit_info,
                "can_dm": dm_enabled and is_whitelisted
            }
            
        except Exception as e:
            logger.error(f"Error getting DM status: {e}")
            return {
                "user_id": context.user_id,
                "user_name": context.user_name,
                "error": str(e)
            }
    
    async def add_user_to_whitelist(self, user_id: str) -> bool:
        """Add a user to the DM whitelist"""
        try:
            success = self.tool_guard.add_to_whitelist(user_id)
            if success:
                logger.info(f"Added user {user_id} to DM whitelist")
            return success
        except Exception as e:
            logger.error(f"Error adding user to whitelist: {e}")
            return False
    
    async def remove_user_from_whitelist(self, user_id: str) -> bool:
        """Remove a user from the DM whitelist"""
        try:
            success = self.tool_guard.remove_from_whitelist(user_id)
            if success:
                logger.info(f"Removed user {user_id} from DM whitelist")
            return success
        except Exception as e:
            logger.error(f"Error removing user from whitelist: {e}")
            return False
    
    def get_whitelist(self) -> list:
        """Get current DM whitelist"""
        return self.tool_guard.get_whitelist()
    
    def _check_and_clean_dm_content(self, message: str) -> Optional[str]:
        """Check DM content for filtered words and clean or reject it"""
        try:
            # Get filter handler from bot
            if not hasattr(self.bot, 'filter_handler') or not self.bot.filter_handler:
                # No filter available, allow message as-is
                return message
            
            filter_handler = self.bot.filter_handler
            
            # Check if message contains filtered words
            if filter_handler.is_filtered(message):
                logger.warning(f"DM content contains filtered words: {message[:50]}...")
                
                # Strategy 1: Try to clean the content by removing problematic words
                cleaned_message = self._attempt_content_cleaning(message, filter_handler)
                if cleaned_message and len(cleaned_message.strip()) > 10:  # Ensure meaningful content remains
                    logger.info(f"Successfully cleaned DM content: {cleaned_message[:50]}...")
                    return cleaned_message
                
                # Strategy 2: If cleaning failed, reject the message entirely
                logger.info("Could not clean DM content effectively, blocking message")
                return None
            
            # Message is clean, return as-is
            return message
            
        except Exception as e:
            logger.error(f"Error checking DM content: {e}")
            # On error, be conservative and allow the message
            return message
    
    def _attempt_content_cleaning(self, message: str, filter_handler) -> Optional[str]:
        """Attempt to intelligently clean filtered words from DM content using word boundaries"""
        try:
            # Get filtered words
            filtered_words = filter_handler.get_filtered_words()
            
            cleaned = message
            words_removed = []
            
            # Use word-boundary matching to avoid "laFiltered" issues
            for word in filtered_words:
                # Create pattern that matches whole words only (with word boundaries)
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                
                if pattern.search(cleaned):
                    # Replace whole word occurrences only
                    cleaned = pattern.sub('', cleaned)
                    words_removed.append(word)
            
            # Clean up extra spaces and punctuation
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            cleaned = re.sub(r'\s+([,.!?])', r'\1', cleaned)  # Fix "word ." → "word."
            
            # Check if remaining content is meaningful
            if len(cleaned) < 10 or not any(c.isalpha() for c in cleaned):
                logger.info(f"Cleaned content too short or meaningless: '{cleaned}'")
                return None
            
            if words_removed:
                logger.info(f"Removed filtered words from DM (word-boundary): {words_removed}")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning DM content: {e}")
            return None
