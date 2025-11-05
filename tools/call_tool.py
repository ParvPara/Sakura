"""
Voice Channel Call Tool for Sakura Bot
Handles calling users to join voice channels with custom messages
"""

import logging
import discord
from typing import Optional
from dataclasses import dataclass

from schemas import ToolContext, ToolResponse, ToolStatus
from router.tool_guard import ToolGuard

logger = logging.getLogger(__name__)

@dataclass
class CallResult:
    """Result of a voice channel call attempt"""
    success: bool
    message_id: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None
    status: str = "unknown"

class CallTool:
    """Tool for calling users to voice channels with custom messages"""
    
    def __init__(self, bot, tool_guard: ToolGuard):
        self.bot = bot
        self.tool_guard = tool_guard
        logger.info("Invite Tool initialized with guard integration")
    
    async def execute_invite(self, context: ToolContext, message: str, target_user: Optional[str] = None) -> ToolResponse:
        """Execute voice channel invite with custom message"""
        try:
            # Check if bot is in a voice channel
            if not self.bot.voice_clients:
                return ToolResponse(
                    ok=False,
                    tool="CALL",
                    status=ToolStatus.ERROR,
                    reason="I'm not currently in a voice channel to invite you to",
                    meta={"error": "no_voice_channel"}
                )
            
            voice_client = self.bot.voice_clients[0]
            voice_channel = voice_client.channel
            
            if not voice_channel:
                return ToolResponse(
                    ok=False,
                    tool="CALL",
                    status=ToolStatus.ERROR,
                    reason="I can't determine which voice channel I'm in",
                    meta={"error": "no_channel_info"}
                )
            
            # Find the target user
            if target_user:
                # Look for specific user
                target_user_obj = await self._find_user_by_name(target_user, voice_channel.guild)
            else:
                # Use the context user (person who requested the invite)
                target_user_obj = self.bot.get_user(int(context.user_id))
                if not target_user_obj:
                    # Try to find in guild
                    for guild in self.bot.guilds:
                        target_user_obj = guild.get_member(int(context.user_id))
                        if target_user_obj:
                            break
            
            if not target_user_obj:
                user_name = target_user if target_user else context.user_name
                return ToolResponse(
                    ok=False,
                    tool="CALL",
                    status=ToolStatus.ERROR,
                    reason=f"I couldn't find user '{user_name}' to invite",
                    meta={"target_user": user_name, "error": "user_not_found"}
                )
            
            # Check if user is already in the voice channel
            if target_user_obj in voice_channel.members:
                return ToolResponse(
                    ok=False,
                    tool="CALL",
                    status=ToolStatus.ERROR,
                    reason=f"{target_user_obj.display_name} is already in the voice channel!",
                    meta={"target_user": target_user_obj.display_name, "error": "already_in_channel"}
                )
            
            # Send the invite message via DM
            invite_result = await self._send_invite_message(target_user_obj, voice_channel, message)
            
            if invite_result.success:
                self.tool_guard.log_tool_attempt(context, "CALL", True, "executed", {
                    "target_user": target_user_obj.display_name,
                    "target_user_id": str(target_user_obj.id),
                    "voice_channel": voice_channel.name,
                    "message_id": invite_result.message_id
                })
                
                return ToolResponse(
                    ok=True,
                    tool="CALL",
                    status=ToolStatus.EXECUTED,
                    reason=f"Voice channel invite sent to {target_user_obj.display_name}",
                    meta={
                        "target_user": target_user_obj.display_name,
                        "target_user_id": str(target_user_obj.id),
                        "voice_channel": voice_channel.name,
                        "message_id": invite_result.message_id,
                        "invite_status": invite_result.status
                    }
                )
            else:
                return ToolResponse(
                    ok=False,
                    tool="CALL",
                    status=ToolStatus.ERROR,
                    reason=f"Failed to send invite to {target_user_obj.display_name}: {invite_result.error}",
                    meta={
                        "target_user": target_user_obj.display_name,
                        "target_user_id": str(target_user_obj.id),
                        "error": invite_result.error,
                        "invite_status": invite_result.status
                    }
                )
                
        except Exception as e:
            logger.error(f"Error in voice invite: {e}")
            return ToolResponse(
                ok=False,
                tool="INVITE",
                status=ToolStatus.ERROR,
                reason=f"Voice invite error: {str(e)}",
                meta={"error": str(e)}
            )
    
    async def _find_user_by_name(self, user_name: str, guild: discord.Guild) -> Optional[discord.Member]:
        """Find a user by name in the guild, including real name lookup from people.json"""
        try:
            # First try to resolve real name to Discord username via people.json
            discord_username = await self._resolve_real_name_to_discord_username(user_name)
            if discord_username:
                logger.info(f"Resolved real name '{user_name}' to Discord username '{discord_username}'")
                # Look for the Discord username
                for member in guild.members:
                    if member.name.lower() == discord_username.lower():
                        return member
                    if member.display_name.lower() == discord_username.lower():
                        return member
            
            # If no people.json match, try direct Discord name matches
            for member in guild.members:
                if member.name.lower() == user_name.lower():
                    return member
            
            for member in guild.members:
                if member.display_name.lower() == user_name.lower():
                    return member
            
            for member in guild.members:
                if user_name.lower() in member.name.lower() or user_name.lower() in member.display_name.lower():
                    return member
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding user {user_name}: {e}")
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
    
    async def _send_invite_message(self, user: discord.Member, voice_channel: discord.VoiceChannel, message: str) -> CallResult:
        """Send the actual call message to the user"""
        try:
            # Create a rich embed call
            embed = discord.Embed(
                title="📞 Voice Channel Call",
                description=message,
                color=0x7289DA  # Discord blurple
            )
            
            embed.add_field(
                name="Voice Channel",
                value=f"#{voice_channel.name}",
                inline=True
            )
            
            embed.add_field(
                name="Server",
                value=voice_channel.guild.name,
                inline=True
            )
            
            embed.add_field(
                name="Current Members",
                value=f"{len(voice_channel.members)} people in channel",
                inline=True
            )
            
            # Try to create a voice channel invite link
            try:
                invite = await voice_channel.create_invite(
                    max_age=300,  # 5 minutes
                    max_uses=1,
                    reason=f"Voice channel invite for {user.display_name}"
                )
                embed.add_field(
                    name="Quick Join",
                    value=f"[Click here to join]({invite.url})",
                    inline=False
                )
            except Exception as e:
                logger.warning(f"Could not create voice channel invite: {e}")
                # Fallback without invite link
                embed.add_field(
                    name="How to Join",
                    value=f"Look for the voice channel **#{voice_channel.name}** in **{voice_channel.guild.name}**",
                    inline=False
                )
            
            embed.set_footer(text="This invitation will expire in 5 minutes")
            
            # Send DM to user
            logger.info(f"Attempting to send voice call DM to {user.display_name} ({user.name})")
            dm_message = await user.send(embed=embed)
            logger.info(f"Successfully sent voice call DM to {user.display_name}, message ID: {dm_message.id}")
            
            return CallResult(
                success=True,
                message_id=str(dm_message.id),
                user_id=str(user.id),
                status="sent"
            )
            
        except discord.Forbidden:
            return CallResult(
                success=False,
                user_id=str(user.id),
                error="User has DMs disabled or blocked the bot",
                status="forbidden"
            )
        except Exception as e:
            logger.error(f"Error sending invite to {user.display_name}: {e}")
            return CallResult(
                success=False,
                user_id=str(user.id) if user else None,
                error=str(e),
                status="error"
            )