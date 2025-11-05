import discord
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class DMConversation:
    """Represents a DM conversation with a user"""
    user_id: int
    user_name: str
    messages: List[Dict[str, Any]]  # List of message objects
    last_activity: datetime
    is_active: bool = True
    auto_reply_enabled: bool = True
    priority_level: int = 1  # 1=normal, 2=important, 3=urgent
    context_notes: str = ""
    
class DiscordDMHandler:
    def __init__(self, bot, chat_handler, memory_system, llm_wrapper):
        self.bot = bot
        self.chat_handler = chat_handler
        self.memory_system = memory_system
        self.llm_wrapper = llm_wrapper
        
        # DM conversation tracking
        self.conversations: Dict[int, DMConversation] = {}
        self.pending_messages: List[Dict[str, Any]] = []
        self.auto_reply_cooldown: Dict[int, float] = {}  # User ID -> last reply time
        self.cooldown_duration = 30  # Seconds between auto-replies to same user
        
        # DM settings
        self.max_conversations = 50
        self.max_messages_per_conversation = 100
        self.auto_reply_enabled = True
        self.dm_monitoring_enabled = True
        
        # User preferences and blacklist
        self.user_preferences: Dict[int, Dict[str, Any]] = {}
        self.blacklisted_users: Set[int] = set()
        
        # Notification settings
        self.notify_on_new_dm = True
        self.notify_on_urgent = True
        
        # Background tasks will be started when bot is ready
        self._background_tasks_started = False
    
    def _start_background_tasks(self):
        """Start background tasks for DM management"""
        if not self._background_tasks_started:
            asyncio.create_task(self._cleanup_old_conversations())
            asyncio.create_task(self._process_pending_messages())
            asyncio.create_task(self._monitor_dm_activity())
            self._background_tasks_started = True
            logger.info("DM background tasks started")
    
    async def start_background_tasks(self):
        """Start background tasks when bot is ready"""
        if not self._background_tasks_started:
            self._start_background_tasks()
    
    async def _cleanup_old_conversations(self):
        """Periodically cleanup old conversations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                cutoff_time = datetime.now() - timedelta(days=7)
                
                to_remove = []
                for user_id, conv in self.conversations.items():
                    if conv.last_activity < cutoff_time and not conv.is_active:
                        to_remove.append(user_id)
                
                for user_id in to_remove:
                    del self.conversations[user_id]
                    logger.info(f"Cleaned up old DM conversation with user {user_id}")
                    
            except Exception as e:
                logger.error(f"Error in conversation cleanup: {e}")
    
    async def _process_pending_messages(self):
        """Process pending DM messages"""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if not self.pending_messages:
                    continue
                
                # Process messages in order
                while self.pending_messages:
                    message_data = self.pending_messages.pop(0)
                    await self._send_dm_message(message_data)
                    
            except Exception as e:
                logger.error(f"Error processing pending messages: {e}")
    
    async def _send_dm_message(self, message_data: Dict[str, Any]):
        """Send a DM message from pending queue"""
        try:
            user_id = message_data.get("user_id")
            content = message_data.get("content")
            
            if user_id and content:
                await self.send_dm_to_user(user_id, content)
                
        except Exception as e:
            logger.error(f"Error sending pending DM message: {e}")
    
    async def _monitor_dm_activity(self):
        """Monitor DM activity and trigger notifications"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.notify_on_new_dm:
                    continue
                
                # Check for new conversations that need attention
                for user_id, conv in self.conversations.items():
                    if conv.is_active and conv.priority_level >= 2:
                        # High priority conversation - could trigger notifications
                        pass
                        
            except Exception as e:
                logger.error(f"Error in DM monitoring: {e}")
    
    async def handle_incoming_dm(self, message: discord.Message) -> bool:
        """Handle incoming DM message and decide whether to reply"""
        try:
            user_id = message.author.id
            user_name = message.author.name
            content = message.content
            
            logger.info(f"Received DM from {user_name} ({user_id}): {content[:50]}...")
            
            # Check if user is blacklisted
            if user_id in self.blacklisted_users:
                logger.info(f"Ignoring DM from blacklisted user {user_name}")
                return False
            
            # Get or create conversation
            conversation = self._get_or_create_conversation(user_id, user_name)
            
            # Add message to conversation history
            conversation.messages.append({
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "message_id": message.id
            })
            
            # Update conversation activity
            conversation.last_activity = datetime.now()
            conversation.is_active = True
            
            # Limit conversation history
            if len(conversation.messages) > self.max_messages_per_conversation:
                conversation.messages = conversation.messages[-self.max_messages_per_conversation:]
            
            # Notify voice channel about new DM (if enabled and bot is in voice)
            if self.notify_on_new_dm:
                await self._notify_voice_channel_about_dm(user_name, content)
            
            # Decide whether to auto-reply
            should_reply = await self._decide_auto_reply(message, conversation)
            
            if should_reply:
                await self._generate_and_send_reply(message, conversation)
                return True
            else:
                logger.info(f"Decided not to auto-reply to {user_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling incoming DM: {e}")
            return False
    
    def _get_or_create_conversation(self, user_id: int, user_name: str) -> DMConversation:
        """Get existing conversation or create new one"""
        if user_id in self.conversations:
            return self.conversations[user_id]
        
        # Create new conversation
        conversation = DMConversation(
            user_id=user_id,
            user_name=user_name,
            messages=[],
            last_activity=datetime.now(),
            auto_reply_enabled=self.auto_reply_enabled
        )
        
        self.conversations[user_id] = conversation
        
        # Limit total conversations
        if len(self.conversations) > self.max_conversations:
            # Remove oldest inactive conversation
            oldest_user_id = min(
                self.conversations.keys(),
                key=lambda uid: self.conversations[uid].last_activity
            )
            del self.conversations[oldest_user_id]
        
        logger.info(f"Created new DM conversation with {user_name}")
        return conversation
    
    async def _decide_auto_reply(self, message: discord.Message, conversation: DMConversation) -> bool:
        """Use LLM to decide whether to auto-reply to a DM"""
        try:
            # Check cooldown
            user_id = message.author.id
            current_time = time.time()
            if user_id in self.auto_reply_cooldown:
                if current_time - self.auto_reply_cooldown[user_id] < self.cooldown_duration:
                    logger.info(f"Auto-reply on cooldown for user {message.author.name}")
                    return False
            
            # Check if auto-reply is disabled for this conversation
            if not conversation.auto_reply_enabled:
                return False
            
            # Get user context from memory system
            user_context = self._get_user_context(message.author)
            
            # Build decision prompt
            decision_prompt = self._build_auto_reply_decision_prompt(message, conversation, user_context)
            
            # Get LLM decision
            response = self.llm_wrapper.get_response_sync(
                messages=[{"role": "user", "content": decision_prompt}]
            )
            
            # Parse decision
            should_reply = self._parse_auto_reply_decision(response)
            
            logger.info(f"Auto-reply decision for {message.author.name}: {should_reply}")
            return should_reply
            
        except Exception as e:
            logger.error(f"Error deciding auto-reply: {e}")
            return False
    
    def _build_auto_reply_decision_prompt(self, message: discord.Message, conversation: DMConversation, user_context: Dict) -> str:
        """Build prompt for LLM to decide whether to auto-reply"""
        recent_messages = conversation.messages[-5:]  # Last 5 messages
        
        prompt = f"""You are Sakura, an AI assistant. You need to decide whether to automatically reply to a direct message.

USER CONTEXT:
- Name: {message.author.name}
- User Context: {json.dumps(user_context, indent=2)}
- Conversation Priority: {conversation.priority_level}
- Auto-reply enabled: {conversation.auto_reply_enabled}

RECENT CONVERSATION HISTORY:
"""
        
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"- {role.upper()}: {content}\n"
        
        prompt += f"""
CURRENT MESSAGE:
{message.content}

DECISION CRITERIA:
- Reply if the message requires a response (questions, requests, greetings)
- Reply if the user seems to expect a response
- Don't reply if the message is just a statement that doesn't need a response
- Don't reply if the user is just venting or thinking out loud
- Don't reply if the message is spam or inappropriate

RESPOND WITH ONLY: YES or NO

Decision:"""
        
        return prompt
    
    def _parse_auto_reply_decision(self, response: str) -> bool:
        """Parse LLM response to determine if should auto-reply"""
        response_lower = response.strip().lower()
        
        # Look for clear yes/no indicators
        if any(word in response_lower for word in ['yes', 'true', 'reply', 'respond']):
            return True
        elif any(word in response_lower for word in ['no', 'false', 'don\'t', 'shouldn\'t']):
            return False
        
        # Default to no if unclear
        return False
    
    async def _generate_and_send_reply(self, message: discord.Message, conversation: DMConversation):
        """Generate and send a reply to a DM"""
        try:
            # Update cooldown
            self.auto_reply_cooldown[message.author.id] = time.time()
            
            # Get user context
            user_context = self._get_user_context(message.author)
            
            # Build conversation context
            conversation_context = self._build_conversation_context(conversation)
            
            # Generate reply using chat handler with proper context
            guild_id = 0  # DM channel
            channel_id = message.channel.id
            
            # Resolve Discord username to real name for chat handler
            real_name = self._resolve_username_to_real_name(message.author.name)
            user_name = real_name if real_name != message.author.name else message.author.name
            
            # Build context-aware message that includes conversation history
            if len(conversation.messages) > 1:
                # Include recent conversation context
                context_messages = conversation.messages[-5:]  # Last 5 messages
                context_summary = "Recent DM conversation:\n"
                for msg in context_messages:
                    role = "You" if msg["role"] == "assistant" else user_name
                    context_summary += f"{role}: {msg['content']}\n"
                
                enhanced_message = f"[DM Context]\n{context_summary}\n[Current Message]\n{user_name}: {message.content}"
            else:
                enhanced_message = f"[DM from {user_name}]\n{message.content}"
            
            # Generate DM reply using specialized prompt that prevents tool usage
            dm_reply_prompt = f"""DM REPLY MODE: You're responding to a private DM from {user_name}.

{enhanced_message}

TASK: Generate a natural, friendly DM reply as Sakura.

IMPORTANT RULES:
- DO NOT use any tools or JSON blocks
- DO NOT generate tool_call objects
- This is a private DM response, not a tool action
- Be conversational and in character as Sakura
- Keep it engaging but not too long
- Respond directly to what they said

Generate your DM reply:"""

            # Use direct LLM call to avoid tool system
            reply = self.llm_wrapper.get_response_sync(
                messages=[{"role": "user", "content": dm_reply_prompt}]
            )
            
            # Clean the reply to remove any JSON artifacts
            if reply:
                reply = self._clean_announcement_response(reply)
            
            # Fallback if reply is empty after cleaning
            if not reply or len(reply.strip()) < 5:
                reply = f"hey {user_name}! got your message, thanks for reaching out!"
            
            # Send reply
            await message.channel.send(reply)
            
            # Add reply to conversation history
            conversation.messages.append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now().isoformat(),
                "message_id": None  # We don't have the message ID yet
            })
            
            logger.info(f"Sent auto-reply to {message.author.name}: {reply[:50]}...")
            
        except Exception as e:
            logger.error(f"Error generating and sending reply: {e}")
    
    def _get_user_context(self, user: discord.User) -> Dict[str, Any]:
        """Get user context from memory system"""
        try:
            # Try to find person by Discord username first
            person = self._find_person_by_discord_username(user.name)
            if person:
                return {
                    "name": person.name,
                    "discord_username": getattr(person, 'discord_username', user.name),
                    "traits": getattr(person, 'traits', []),
                    "interests": getattr(person, 'interests', []),
                    "relationship": getattr(person, 'relationship', 'unknown'),
                    "interaction_count": getattr(person, 'interaction_count', 0),
                    "last_seen": getattr(person, 'last_seen', 'unknown')
                }
            return {"name": user.name, "discord_username": user.name, "new_user": True}
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {"name": user.name, "discord_username": user.name, "error": str(e)}
    
    def _find_person_by_discord_username(self, discord_username: str):
        """Find person by Discord username in memory system"""
        try:
            # First try direct lookup
            person = self.memory_system.get_person(discord_username)
            if person:
                return person
            
            # Search through all people for Discord username match
            all_people = self.memory_system.people
            for person in all_people.values():
                if hasattr(person, 'discord_username') and person.discord_username == discord_username:
                    return person
            
            return None
        except Exception as e:
            logger.error(f"Error finding person by Discord username {discord_username}: {e}")
            return None
    
    def _build_conversation_context(self, conversation: DMConversation) -> str:
        """Build conversation context for LLM"""
        if not conversation.messages:
            return "No previous conversation history."
        
        context = "Recent conversation:\n"
        for msg in conversation.messages[-10:]:  # Last 10 messages
            role = msg["role"]
            content = msg["content"]
            context += f"{role}: {content}\n"
        
        return context
    
    async def send_dm_to_user(self, user_id: int, message_content: str, priority: int = 1) -> bool:
        """Send a DM to a specific user"""
        try:
            # Get user
            user = self.bot.get_user(user_id)
            if not user:
                logger.error(f"User {user_id} not found")
                return False
            
            # Check if user is blacklisted
            if user_id in self.blacklisted_users:
                logger.warning(f"Attempted to send DM to blacklisted user {user.name}")
                return False
            
            # Check if content should be filtered for DM
            if hasattr(self.bot, 'filter_handler') and self.bot.filter_handler.is_filtered(message_content):
                logger.info(f"DM: Content filtered, sending 'Filtered.' instead of: '{message_content[:50]}...'")
                await user.send("Filtered.")
            else:
                await user.send(message_content)
            
            # Update conversation
            conversation = self._get_or_create_conversation(user_id, user.name)
            conversation.messages.append({
                "role": "assistant",
                "content": message_content,
                "timestamp": datetime.now().isoformat(),
                "message_id": None
            })
            conversation.last_activity = datetime.now()
            conversation.is_active = True
            
            logger.info(f"Sent DM to {user.name}: {message_content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error sending DM to user {user_id}: {e}")
            return False
    
    async def initiate_conversation(self, user_id: int, reason: str = "") -> bool:
        """Initiate a conversation with a user"""
        try:
            # Get user
            user = self.bot.get_user(user_id)
            if not user:
                logger.error(f"User {user_id} not found")
                return False
            
            # Generate initiation message using LLM
            initiation_prompt = f"""You are Sakura. You want to initiate a conversation with {user.name}. 

Reason for reaching out: {reason}

Generate a natural message in character as Sakura to start the conversation. Keep it casual and conversational.

Message:"""
            
            initiation_message = await self.llm_wrapper.get_response_sync(
                messages=[{"role": "user", "content": initiation_prompt}]
            )
            
            # Send the message
            success = await self.send_dm_to_user(user_id, initiation_message)
            
            if success:
                logger.info(f"Initiated conversation with {user.name}: {initiation_message[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initiating conversation with user {user_id}: {e}")
            return False
    
    def get_conversation_summary(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get summary of conversation with a user"""
        if user_id not in self.conversations:
            return None
        
        conv = self.conversations[user_id]
        
        return {
            "user_name": conv.user_name,
            "message_count": len(conv.messages),
            "last_activity": conv.last_activity.isoformat(),
            "is_active": conv.is_active,
            "auto_reply_enabled": conv.auto_reply_enabled,
            "priority_level": conv.priority_level,
            "context_notes": conv.context_notes
        }
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Get summary of all conversations"""
        summaries = []
        for user_id, conv in self.conversations.items():
            summaries.append({
                "user_id": user_id,
                "user_name": conv.user_name,
                "message_count": len(conv.messages),
                "last_activity": conv.last_activity.isoformat(),
                "is_active": conv.is_active,
                "priority_level": conv.priority_level
            })
        
        # Sort by last activity (most recent first)
        summaries.sort(key=lambda x: x["last_activity"], reverse=True)
        return summaries
    
    def set_conversation_priority(self, user_id: int, priority: int):
        """Set priority level for a conversation"""
        if user_id in self.conversations:
            self.conversations[user_id].priority_level = max(1, min(3, priority))
            logger.info(f"Set conversation priority for user {user_id} to {priority}")
    
    def toggle_auto_reply(self, user_id: int, enabled: bool = None):
        """Toggle auto-reply for a specific conversation"""
        if user_id in self.conversations:
            if enabled is None:
                # Toggle current state
                enabled = not self.conversations[user_id].auto_reply_enabled
            
            self.conversations[user_id].auto_reply_enabled = enabled
            logger.info(f"Auto-reply {'enabled' if enabled else 'disabled'} for user {user_id}")
    
    def blacklist_user(self, user_id: int):
        """Add user to blacklist"""
        self.blacklisted_users.add(user_id)
        logger.info(f"User {user_id} added to DM blacklist")
    
    def unblacklist_user(self, user_id: int):
        """Remove user from blacklist"""
        self.blacklisted_users.discard(user_id)
        logger.info(f"User {user_id} removed from DM blacklist")
    
    def is_user_blacklisted(self, user_id: int) -> bool:
        """Check if user is blacklisted"""
        return user_id in self.blacklisted_users
    
    def get_dm_stats(self) -> Dict[str, Any]:
        """Get DM statistics"""
        total_conversations = len(self.conversations)
        active_conversations = sum(1 for conv in self.conversations.values() if conv.is_active)
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())
        blacklisted_count = len(self.blacklisted_users)
        
        return {
            "total_conversations": total_conversations,
            "active_conversations": active_conversations,
            "total_messages": total_messages,
            "blacklisted_users": blacklisted_count,
            "auto_reply_enabled": self.auto_reply_enabled,
            "dm_monitoring_enabled": self.dm_monitoring_enabled
        }
    
    async def _notify_voice_channel_about_dm(self, user_name: str, content: str):
        """Notify voice channel about incoming DM by letting Sakura mention it"""
        try:
            # Only notify if bot is in a voice channel
            if not self.bot.voice_clients:
                return
            
            voice_client = self.bot.voice_clients[0]
            if not voice_client or not voice_client.channel:
                return
            
            # Resolve Discord username to real name
            real_name = self._resolve_username_to_real_name(user_name)
            display_name = real_name if real_name != user_name else user_name
            
            # Let the LLM generate a natural response about receiving the DM
            dm_summary = content[:100] + "..." if len(content) > 100 else content
            
            # Create specialized prompt that prevents tool usage and stays on topic
            announcement_prompt = f"""VOICE ANNOUNCEMENT: {display_name} just sent you a DM saying: "{dm_summary}"

Generate a SHORT, natural reaction (1 sentence) to announce this to your voice channel friends.

RULES:
- Be Sakura - chaotic, sassy, dramatic
- NO tools, NO JSON, NO analysis
- React to what they actually said
- Keep it under 15 words

Examples:
- "OMG {display_name} just asked if I'm lonely lmaooo"
- "Hold up, {display_name} just DMed me something spicy!"
- "{display_name} just slid into my DMs asking about [topic]"

Your reaction:"""
            
            # Generate response using direct LLM call to avoid tool system
            try:
                llm_response = self.llm_wrapper.get_response_sync(
                    messages=[{"role": "user", "content": announcement_prompt}]
                )
                
                if llm_response and llm_response.strip():
                    # Clean the response to remove any JSON artifacts
                    notification_text = self._clean_announcement_response(llm_response.strip())
                    
                    # If cleaning removed too much content, use fallback
                    if len(notification_text) < 10:
                        notification_text = f"OMG {display_name} just DMed me! They said: '{dm_summary}' - lemme reply real quick!"
                else:
                    # Fallback if LLM fails
                    notification_text = f"Hold on, I just got a DM from {display_name}! They said: '{dm_summary}' - lemme reply to them real quick"
                    
            except Exception as e:
                logger.error(f"Error generating LLM DM announcement: {e}")
                # Fallback notification
                notification_text = f"Hold on, I just got a DM from {display_name}! They said: '{dm_summary}' - lemme reply to them real quick"
            
            # Use TTS to announce the DM in voice channel
            if hasattr(self.bot, 'tts_handler'):
                logger.info(f"Announcing DM from {display_name} ({user_name}) in voice channel")
                await self.bot.tts_handler.generate_and_play_speech(notification_text, self.bot)
                    
        except Exception as e:
            logger.error(f"Error notifying voice channel about DM: {e}")
    
    def _resolve_username_to_real_name(self, discord_username: str) -> str:
        """Resolve Discord username to real name from memory system"""
        try:
            logger.info(f"[DM RESOLUTION] Attempting to resolve Discord username: {discord_username}")
            
            # Use the chat handler's memory system to find person by Discord username
            person = self.memory_system.get_person(discord_username)
            if person and hasattr(person, 'name'):
                logger.info(f"[DM RESOLUTION] SUCCESS: Resolved {discord_username} to real name: {person.name}")
                return person.name
            else:
                logger.warning(f"[DM RESOLUTION] get_person({discord_username}) returned: {person}")
            
            # Try to find by Discord username in all people
            all_people = self.memory_system.people
            logger.info(f"[DM RESOLUTION] Searching {len(all_people)} people for Discord username: {discord_username}")
            
            # Debug: Show all people and their Discord usernames
            for name, person in all_people.items():
                discord_user = getattr(person, 'discord_username', 'None')
                logger.info(f"[DM RESOLUTION] Person '{name}' has Discord username: '{discord_user}'")
            
            for person in all_people.values():
                if hasattr(person, 'discord_username') and person.discord_username == discord_username:
                    logger.info(f"[DM RESOLUTION] FOUND: {person.name} (Discord: {person.discord_username})")
                    return person.name
            
            # If no mapping found, return the Discord username
            logger.warning(f"[DM RESOLUTION] FAILED: No real name mapping found for {discord_username}, using username")
            return discord_username
            
        except Exception as e:
            logger.error(f"[DM RESOLUTION] ERROR resolving username {discord_username}: {e}")
            return discord_username
    
    def _clean_announcement_response(self, response: str) -> str:
        """Clean LLM response to remove JSON tool calls for voice announcements"""
        try:
            import re
            
            # Remove JSON tool call blocks (various formats)
            cleaned = re.sub(r'```json\s*\{.*?\}\s*```', '', response, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'json\s*\{.*?\}', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'\{.*?"tool_call".*?\}', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any standalone JSON objects
            cleaned = re.sub(r'\{[^}]*"name"\s*:\s*"[^"]*"[^}]*\}', '', cleaned, flags=re.DOTALL)
            
            # Remove AI-like analysis phrases
            cleaned = re.sub(r'json\s+response\s+analysis\s+suggests?\s+that\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'analysis\s+suggests?\s+that\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'based\s+on\s+the\s+analysis\s*', '', cleaned, flags=re.IGNORECASE)
            
            # Clean up extra whitespace and line breaks
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # Remove leading/trailing quotes if present
            cleaned = cleaned.strip('"\'')
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning announcement response: {e}")
            return response
