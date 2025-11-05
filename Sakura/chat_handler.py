import time
import re
import asyncio
import datetime
from typing import Dict, List, Optional, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

from Sakura.llm_wrapper import LLMWrapper
from Sakura.filter_handler import FilterHandler
from Sakura.memory_system import AdvancedMemorySystem  # assume available

# -------- Constants / Config --------
CONV_TTL_SECONDS = 3600          # GC conversations idle > 1h
MAX_TURNS_PER_CONV = 24          # cap history to last N turns (user+assistant entries)
STATUS_IDLE = "idle"
STATUS_PROCESSING = "processing"
STATUS_STREAMING = "streaming"
STATUS_COMPLETED = "completed"
STATUS_CANCELLED = "cancelled"
STATUS_ERROR = "error"
STATUS_QUEUED = "queued"

# Matches one or more [MEMORY] ... [/MEMORY] blocks (non-greedy, multiline)
MEMORY_BLOCK_RE = re.compile(r"\[MEMORY\](.+?)\[/MEMORY\]", re.DOTALL | re.IGNORECASE)

# A lightweight “tool-ish” tag scrubber to prevent accidental prompt injection via user text
TOOLISH_TAG_RE = re.compile(
    r"(\[SYSTEM\]|\[/SYSTEM\]|\[INTERNAL\]|\[/INTERNAL\]|\[TOOL\]|\[/TOOL\])",
    re.IGNORECASE
)

def strip_toolish_tags(text: str) -> str:
    """Remove accidental tool-ish tags from user input without touching normal content."""
    if not text:
        return text
    return TOOLISH_TAG_RE.sub("", text)

def extract_memories(text: str) -> Tuple[str, List[str]]:
    """
    Extract all [MEMORY]...[/MEMORY] blocks, return (text_without_blocks, memories_list).
    Non-greedy to allow multiple blocks in one response.
    """
    if not text:
        return text, []
    memories = MEMORY_BLOCK_RE.findall(text) or []
    cleaned = MEMORY_BLOCK_RE.sub("", text).strip()
    return cleaned, [m.strip() for m in memories if m.strip()]

class ChatHandler:
    def __init__(self):
        self.filter_handler = FilterHandler()
        self.llm_wrapper = LLMWrapper(self.filter_handler)
        self.conversations: Dict[str, Dict[str, any]] = {}
        self.memory_system = AdvancedMemorySystem()

        # Status for UI
        self.current_message = {"status": STATUS_IDLE, "content": ""}
        self.next_message = {"status": STATUS_QUEUED, "content": ""}
        
        # Tool call tracking for UI
        self.tool_calls_log = []  # List of tool call attempts with timestamps

        # Per-conversation async locks to avoid races
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_conversation_key(self, guild_id: int, channel_id: int) -> str:
        return f"{guild_id}:{channel_id}"

    def _get_lock(self, conv_key: str) -> asyncio.Lock:
        if conv_key not in self._locks:
            self._locks[conv_key] = asyncio.Lock()
        return self._locks[conv_key]

    def _cleanup_old_conversations(self):
        now = time.time()
        stale = [k for k, v in self.conversations.items() if now - v.get("last_updated", 0) > CONV_TTL_SECONDS]
        for k in stale:
            self.conversations.pop(k, None)
            self._locks.pop(k, None)

    def _initialize_conversation(self, conv_key: str):
        self.conversations[conv_key] = {
            "messages": [],          # list of {"role": "user"|"assistant", "content": str}
            "last_updated": time.time(),
        }

    def _cap_conversation(self, conv_key: str):
        """Trim very long histories. Optionally summarize the tail (stub)."""
        msgs = self.conversations[conv_key]["messages"]
        if len(msgs) > MAX_TURNS_PER_CONV:
            # Keep last N turns; optionally summarize the older ones (no-op stub here)
            # summary = self._summarize_history(msgs[:-MAX_TURNS_PER_CONV])
            # if summary: msgs = [{"role":"system","content": f"Summary so far: {summary}"}] + msgs[-MAX_TURNS_PER_CONV:]
            msgs = msgs[-MAX_TURNS_PER_CONV:]
            self.conversations[conv_key]["messages"] = msgs

    def _clean_response_for_memory(self, response: str) -> str:
        """Clean response before storing in memory to prevent bad examples"""
        
        # First, strip prompt leakage that appears at the end (e.g., "Prompt: User: ... You: ...")
        prompt_leak_markers = ["Prompt:", " You:", " User:"]
        for marker in prompt_leak_markers:
            if marker in response:
                idx = response.find(marker)
                if idx > 10:  # Only strip if there's actual content before it
                    print(f"[DEBUG] Stripping prompt leakage from response starting at: {marker}")
                    response = response[:idx].strip()
                    break
        
        # Don't store responses with system prompt leakage or wrong personality
        bad_markers = [
            "CORE IDENTITY:", "EVIL DIRECTION:", "COMMUNICATION STYLE:", 
            "RESPONSE LENGTH:", "TTS OPTIMIZATION:", "BEHAVIORAL DIRECTION:",
            "GUARDRAILS:", "CRITICAL:", "Your personality:", "## TOOLS AVAILABLE:",
            "### Available Tools:", "### When to use tools:", "CONTENT SAFETY:",
            "Respond naturally like a real person", "Be direct and authentic",
            "Never use brackets, tags, or formatting", "No meta-commentary or explanations",
            "Unclear. What does this prompt have", "What does this prompt",
            "VLMSUMARYJSON", "Yes/No/Unclear + Reason:", "REDACTED",
            "within the bounding box provided", "visible within",
            # Wrong personality phrases that indicate helpful AI behavior
            "it's all in good fun", "exercise in pure chaos and creativity",
            "let's work through this mess together", "I am so flattered by your suggestion",
            "Always ready to spice things up"
        ]
        
        response_lower = response.lower()
        for marker in bad_markers:
            if marker.lower() in response_lower:
                print(f"[DEBUG] Not storing response in memory due to bad content: {marker}")
                return "[Response contained system prompt leakage or wrong personality - not stored]"
        
        return response

    def _resolve_username_to_real_name(self, discord_username: str) -> str:
        try:
            person = self.memory_system.get_person(discord_username)
            if person:
                print(f"[DEBUG] Resolved {discord_username} → {person.name}")
                return person.name
            print(f"[DEBUG] No mapping for {discord_username}, using username as name")
            return discord_username
        except Exception as e:
            print(f"[DEBUG] Error resolving username {discord_username}: {e}")
            return discord_username

    # Optional: stub if you want to summarize trimmed content
    def _summarize_history(self, old_messages: List[Dict[str, str]]) -> Optional[str]:
        # You could call a very small local model or heuristic summary here.
        # Keeping as a stub to avoid extra latency.
        return None

    @asynccontextmanager
    async def _locked_conversation(self, conv_key: str):
        lock = self._get_lock(conv_key)
        async with lock:
            yield

    async def get_response(
        self,
        guild_id: int,
        channel_id: int,
        user_name: str,
        message: str
    ) -> str:
        conv_key = self._get_conversation_key(guild_id, channel_id)
        try:
            async with self._locked_conversation(conv_key):
                self.current_message = {"status": STATUS_PROCESSING, "content": message}
                self._cleanup_old_conversations()

                if conv_key not in self.conversations:
                    self._initialize_conversation(conv_key)

                # Reset cancellation flag for this cycle
                self.llm_wrapper.llm_state.next_cancelled = False

                self.conversations[conv_key]["last_updated"] = time.time()

                # Person + resolved name
                # First try to get existing person by Discord username
                existing_person = self.memory_system.get_person(user_name)
                if existing_person:
                    # Person exists, update their interaction data and use their real name
                    existing_person.last_seen = datetime.datetime.now().isoformat()
                    existing_person.interaction_count += 1
                    person = existing_person
                    real_name = person.name
                else:
                    # New person, create with Discord username as both name and username initially
                    # This can be updated later when we learn their real name
                    person = self.memory_system.add_person(user_name, discord_username=user_name)
                    real_name = self._resolve_username_to_real_name(user_name)

                # Sanitize user input for tool-ish tags (avoid accidental prompt injection)
                safe_message = strip_toolish_tags(message)

                # Append user message
                self.conversations[conv_key]["messages"].append({
                    "role": "user",
                    "content": safe_message
                })
                self._cap_conversation(conv_key)

                # Memories
                try:
                    memories = self.memory_system.get_relevant_memories(safe_message, [person.name])
                except Exception as me:
                    print(f"[DEBUG] Memory fetch error: {me}")
                    memories = {}

                # LLM
                bot_response = await self.llm_wrapper.get_response(
                    self.conversations[conv_key]["messages"],
                    memories,
                    real_name
                )

                # Extract memory blocks (all), then save clean response to history
                cleaned_response, memory_items = extract_memories(bot_response)

                # Append assistant message (cleaned)
                self.conversations[conv_key]["messages"].append({
                    "role": "assistant",
                    "content": cleaned_response
                })
                self.conversations[conv_key]["last_updated"] = time.time()
                self._cap_conversation(conv_key)

                # Short-term interaction (use resolved name)
                memory_safe_response = self._clean_response_for_memory(cleaned_response)
                interaction = f"{real_name} said: {safe_message}\nSakura replied: {memory_safe_response}"
                try:
                    self.memory_system.add_short_term_memory(interaction, [person.name])
                except Exception as me:
                    print(f"[DEBUG] add_short_term_memory error: {me}")

                # Long-term memory blocks
                for m in memory_items:
                    try:
                        self.memory_system.add_long_term_memory(m, "fact", [person.name])
                    except Exception as me:
                        print(f"[DEBUG] add_long_term_memory error: {me}")

                self.current_message = {"status": STATUS_COMPLETED, "content": cleaned_response}
                return cleaned_response

        except Exception as e:
            print(f"Error in chat response: {e}")
            self.current_message = {"status": STATUS_ERROR, "content": str(e)}
            return "Oops~ Something went wrong with my brain circuits!"

    def get_response_sync(
        self,
        guild_id: int,
        channel_id: int,
        user_name: str,
        message: str
    ) -> str:
        """Synchronous version of get_response for agent system"""
        conv_key = self._get_conversation_key(guild_id, channel_id)
        try:
            self.current_message = {"status": STATUS_PROCESSING, "content": message}
            self._cleanup_old_conversations()

            if conv_key not in self.conversations:
                self._initialize_conversation(conv_key)

            # Reset cancellation flag for this cycle
            self.llm_wrapper.llm_state.next_cancelled = False

            self.conversations[conv_key]["last_updated"] = time.time()

            # Person + resolved name
            # First try to get existing person by Discord username
            existing_person = self.memory_system.get_person(user_name)
            if existing_person:
                # Person exists, update their interaction data and use their real name
                existing_person.last_seen = datetime.datetime.now().isoformat()
                existing_person.interaction_count += 1
                person = existing_person
                real_name = person.name
            else:
                # New person, create with Discord username as both name and username initially
                person = self.memory_system.add_person(user_name, discord_username=user_name)
                real_name = self._resolve_username_to_real_name(user_name)

            # Sanitize user input for tool-ish tags (avoid accidental prompt injection)
            safe_message = strip_toolish_tags(message)

            # Append user message
            self.conversations[conv_key]["messages"].append({
                "role": "user",
                "content": safe_message
            })
            self._cap_conversation(conv_key)

            # Memories
            try:
                memories = self.memory_system.get_relevant_memories(safe_message, [person.name])
            except Exception as me:
                print(f"[DEBUG] Memory fetch error: {me}")
                memories = {}

            # LLM (synchronous)
            bot_response = self.llm_wrapper.get_response_sync(
                self.conversations[conv_key]["messages"],
                memories,
                real_name
            )

            # Extract memory blocks (all), then save clean response to history
            cleaned_response, memory_items = extract_memories(bot_response)

            # Append assistant message (cleaned)
            self.conversations[conv_key]["messages"].append({
                "role": "assistant",
                "content": cleaned_response
            })
            self.conversations[conv_key]["last_updated"] = time.time()
            self._cap_conversation(conv_key)

            # Short-term interaction (use resolved name)
            interaction = f"{real_name} said: {safe_message}\nSakura replied: {cleaned_response}"
            try:
                self.memory_system.add_short_term_memory(interaction, [person.name])
            except Exception as me:
                print(f"[DEBUG] add_short_term_memory error: {me}")

            # Long-term memory blocks
            for m in memory_items:
                try:
                    self.memory_system.add_long_term_memory(m, "fact", [person.name])
                except Exception as me:
                    print(f"[DEBUG] add_long_term_memory error: {me}")

            self.current_message = {"status": STATUS_COMPLETED, "content": cleaned_response}
            return cleaned_response

        except Exception as e:
            print(f"Error in chat response (sync): {e}")
            self.current_message = {"status": STATUS_ERROR, "content": str(e)}
            return "Oops~ Something went wrong with my brain circuits!"

    async def get_response_streaming(
        self,
        guild_id: int,
        channel_id: int,
        user_name: str,
        message: str
    ) -> AsyncGenerator[str, None]:
        """Stream response for ultra-low latency, with dedup + safe memory handling."""
        conv_key = self._get_conversation_key(guild_id, channel_id)
        try:
            async with self._locked_conversation(conv_key):
                self.current_message = {"status": STATUS_PROCESSING, "content": message}
                self._cleanup_old_conversations()

                if conv_key not in self.conversations:
                    self._initialize_conversation(conv_key)

                # Reset cancellation flag for this cycle
                self.llm_wrapper.llm_state.next_cancelled = False

                self.conversations[conv_key]["last_updated"] = time.time()

                # Person and name
                # First try to get existing person by Discord username
                existing_person = self.memory_system.get_person(user_name)
                if existing_person:
                    # Person exists, update their interaction data and use their real name
                    existing_person.last_seen = datetime.datetime.now().isoformat()
                    existing_person.interaction_count += 1
                    person = existing_person
                    real_name = person.name
                else:
                    # New person, create with Discord username as both name and username initially
                    person = self.memory_system.add_person(user_name, discord_username=user_name)
                    real_name = self._resolve_username_to_real_name(user_name)

                # Sanitize user input
                safe_message = strip_toolish_tags(message)

                # Add user message
                self.conversations[conv_key]["messages"].append({
                    "role": "user",
                    "content": safe_message
                })
                self._cap_conversation(conv_key)

                # Memories
                try:
                    memories = self.memory_system.get_relevant_memories(safe_message, [person.name])
                except Exception as me:
                    print(f"[DEBUG] Memory fetch error: {me}")
                    memories = {}

                # Stream from LLM
                response_chunks: List[str] = []
                accumulated_response = ""
                recent_norm = set()  # live de-dup for streamed sentences
                
                try:
                    async for chunk in self.llm_wrapper.get_response_streaming(
                        self.conversations[conv_key]["messages"],
                        memories,
                        real_name
                    ):
                        # Handle cancel request
                        if self.llm_wrapper.llm_state.next_cancelled:
                            print("[DEBUG] Message generation cancelled during streaming")
                            self.current_message = {"status": STATUS_CANCELLED, "content": accumulated_response}
                            return

                        # Live de-dup at sentence granularity (wrapper already does some; this is extra safe)
                        norm = re.sub(r'\W+', '', chunk).casefold()
                        if norm in recent_norm:
                            continue
                        recent_norm.add(norm)
                        if len(recent_norm) > 30:
                            # bound memory
                            recent_norm = set(list(recent_norm)[-20:])

                        response_chunks.append(chunk)
                        accumulated_response += (chunk if accumulated_response == "" else " " + chunk)

                        self.current_message = {"status": STATUS_STREAMING, "content": accumulated_response}
                        print(f"[DEBUG] Streaming update: {len(accumulated_response)} chars - '{accumulated_response[-60:]}'")

                        # Yield to caller/UI
                        yield chunk
                        
                        # Small delay to prevent overwhelming the UI
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    print(f"[DEBUG] Streaming error: {e}")
                    self.current_message = {"status": "error", "content": f"Streaming error: {str(e)}"}
                    return

                # Combine final response
                full_response = " ".join(response_chunks).strip()
                cleaned_response, memory_items = extract_memories(full_response)

                # Save to history as a single assistant turn
                if cleaned_response:
                    self.conversations[conv_key]["messages"].append({
                        "role": "assistant",
                        "content": cleaned_response
                    })

                self.conversations[conv_key]["last_updated"] = time.time()
                self._cap_conversation(conv_key)

                # Memories
                memory_safe_response = self._clean_response_for_memory(cleaned_response)
                interaction = f"{real_name} said: {safe_message}\nSakura replied: {memory_safe_response}"
                try:
                    self.memory_system.add_short_term_memory(interaction, [person.name])
                except Exception as me:
                    print(f"[DEBUG] add_short_term_memory error: {me}")

                for m in memory_items:
                    try:
                        self.memory_system.add_long_term_memory(m, "fact", [person.name])
                    except Exception as me:
                        print(f"[DEBUG] add_long_term_memory error: {me}")

                self.current_message = {"status": STATUS_COMPLETED, "content": cleaned_response}

        except Exception as e:
            print(f"Error in streaming chat response: {e}")
            self.current_message = {"status": STATUS_ERROR, "content": str(e)}
            yield "Someone tell Eric there is a problem with my AI"

    async def get_response_chunked(
        self,
        guild_id: int,
        channel_id: int,
        user_name: str,
        message: str,
        chunk_size: int = 50
    ) -> List[str]:
        """Chunked response for TTS. Uses streaming path internally, but returns a list."""
        conv_key = self._get_conversation_key(guild_id, channel_id)
        try:
            # Reuse the streaming generator to collect
            collected: List[str] = []
            async for piece in self.get_response_streaming(guild_id, channel_id, user_name, message):
                collected.append(piece)

            # `get_response_streaming` already stored assistant turn and memories,
            # so we only need to chunk client-side now.
            full = " ".join(collected).strip()
            if not full:
                self.current_message = {"status": STATUS_COMPLETED, "content": ""}
                return []

            # Simple sentence-based chunking to ~chunk_size characters
            sentences = re.split(r'(?<=[.!?])\s+', full)
            tts_chunks: List[str] = []
            cur = ""
            for s in sentences:
                if not s:
                    continue
                if cur and len(cur) + 1 + len(s) > chunk_size:
                    tts_chunks.append(cur)
                    cur = s
                else:
                    cur = s if not cur else f"{cur} {s}"
            if cur:
                tts_chunks.append(cur)

            self.current_message = {"status": STATUS_COMPLETED, "content": full}
            return tts_chunks

        except Exception as e:
            print(f"Error in chunked chat response: {e}")
            self.current_message = {"status": STATUS_ERROR, "content": str(e)}
            return ["Oops~ Something went wrong with my brain circuits!"]

    def reset_conversation(self, guild_id: int, channel_id: int):
        conv_key = self._get_conversation_key(guild_id, channel_id)
        self._initialize_conversation(conv_key)

    def clear_all_conversations(self):
        """Clear all conversation history to start fresh."""
        self.conversations.clear()
        self._locks.clear()
        self.current_message = {"status": STATUS_IDLE, "content": ""}
        self.next_message = {"status": STATUS_QUEUED, "content": ""}
        print("[CHAT] Cleared all conversation history")

    def force_reset(self):
        """Force reset everything to start completely fresh."""
        self.clear_all_conversations()
        print("[CHAT] Force reset completed - starting fresh")

    def stop_current_message(self):
        """Stop current message generation (affects next streaming tick)."""
        print("[DEBUG] Stop current message requested")
        self.llm_wrapper.llm_state.next_cancelled = True
        if self.current_message["status"] in (STATUS_PROCESSING, STATUS_STREAMING):
            self.current_message = {"status": STATUS_CANCELLED, "content": self.current_message["content"]}
            print("[DEBUG] Message status updated to cancelled")

    def log_tool_call(self, tool_name: str, args: dict, success: bool, reason: str = ""):
        """Log a tool call attempt for UI tracking"""
        from datetime import datetime
        
        # Keep only last 50 tool calls
        if len(self.tool_calls_log) >= 50:
            self.tool_calls_log = self.tool_calls_log[-49:]
        
        self.tool_calls_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "tool": tool_name,
            "args": args,
            "success": success,
            "reason": reason
        })

    def get_message_status(self) -> Dict[str, Dict[str, str]]:
        """Get current message status for web UI."""
        # Clean tool calls from content before showing in control panel
        current_content = self.current_message.get("content", "")
        next_content = self.next_message.get("content", "")
        
        # Remove JSON tool calls for clean display
        current_content_clean = self._clean_content_for_display(current_content)
        next_content_clean = self._clean_content_for_display(next_content)
        
        return {
            "current_message": {
                "status": self.current_message.get("status", "idle"),
                "content": current_content_clean
            },
            "next_message": {
                "status": self.next_message.get("status", "queued"), 
                "content": next_content_clean
            },
            "tool_calls": self.tool_calls_log[-10:]  # Last 10 tool calls
        }
    
    def _clean_content_for_display(self, content: str) -> str:
        """Clean content for control panel display by removing tool calls and other internal markup"""
        if not content:
            return content
            
        # Remove JSON tool call blocks
        import re
        
        # More robust removal of JSON tool call blocks
        # This handles nested objects and various formatting
        content = re.sub(r'```json\s*\{.*?"tool_call".*?\}\s*```', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining ```json ``` blocks (even empty ones)
        content = re.sub(r'```json\s*.*?\s*```', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any standalone JSON objects that look like tool calls
        content = re.sub(r'\{\s*"tool_call"\s*:.*?\}', '', content, flags=re.DOTALL)
        
        # Remove any remaining { } fragments that might be left over
        content = re.sub(r'\{\s*\}\s*', '', content)
        
        # Clean up extra whitespace and line breaks
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content

    def clear_message_queue(self):
        """No-op queue clearer placeholder (kept for API compatibility)."""
        print("[DEBUG] Clear message queue requested (no-op)")
        return
