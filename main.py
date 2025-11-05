import discord
from discord.ext import commands
from discord.ext import voice_recv
import asyncio
import logging
import config
from datetime import datetime
from Sakura.voice_handler import VoiceHandler
from Sakura.whisper_handler import WhisperHandler
from Sakura.chat_handler import ChatHandler
from Sakura.tts_handler import TTSHandler
from Sakura.filter_handler import FilterHandler
from Sakura.memory_system import AdvancedMemorySystem
from Sakura.discord_DM import DiscordDMHandler
from Sakura.websearch import BraveWebSearch	
from router.intent_router import IntentRouter
from router.tool_guard import ToolGuard
from tools.tool_executor import ToolExecutor
from schemas import ToolContext, ToolCall, ToolResponse, RouterDecision

# LLM Controller API runs separately on port 4000

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not discord.opus.is_loaded():
    import os
    import sys
    import ctypes.util
    
    opus_loaded = False
    
    # Common Windows Opus DLL names to try
    windows_opus_names = [
        'opus.dll',
        'libopus.dll', 
        'libopus-0.dll',
        'opus-0.dll'
    ]
    
    # Try loading from current directory first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for opus_name in windows_opus_names:
        opus_path = os.path.join(current_dir, opus_name)
        if os.path.exists(opus_path):
            try:
                discord.opus.load_opus(opus_path)
                print(f"[DEBUG] Opus library loaded successfully from: {opus_path}")
                opus_loaded = True
                break
            except Exception as e:
                print(f"[DEBUG] Failed to load {opus_path}: {e}")
    
    # If not found in current directory, try PATH
    if not opus_loaded:
        for opus_name in windows_opus_names:
            try:
                discord.opus.load_opus(opus_name)
                print(f"[DEBUG] Opus library loaded successfully from PATH: {opus_name}")
                opus_loaded = True
                break
            except Exception as e:
                print(f"[DEBUG] Failed to load {opus_name} from PATH: {e}")
    
    # Try system search as fallback
    if not opus_loaded:
        try:
            opus_path = ctypes.util.find_library('opus')
            if opus_path:
                discord.opus.load_opus(opus_path)
                print(f"[DEBUG] Loaded system Opus from: {opus_path}")
                opus_loaded = True
            else:
                print("[DEBUG] System Opus not found via ctypes.util")
        except Exception as e:
            print(f"[DEBUG] Failed to load system Opus: {e}")
    
    # Final attempt with explicit Windows search
    if not opus_loaded and sys.platform == "win32":
        try:
            import winreg
            # Check common installation paths
            common_paths = [
                r"C:\Windows\System32",
                r"C:\Windows\SysWOW64",
                current_dir,
                os.path.join(current_dir, "lib"),
                os.path.join(current_dir, "bin"),
            ]
            
            # Add Python installation directory
            python_dir = os.path.dirname(sys.executable)
            common_paths.extend([
                python_dir,
                os.path.join(python_dir, "Library", "bin"),
                os.path.join(python_dir, "Scripts"),
            ])
            
            for search_path in common_paths:
                if os.path.exists(search_path):
                    for opus_name in windows_opus_names:
                        full_path = os.path.join(search_path, opus_name)
                        if os.path.exists(full_path):
                            try:
                                discord.opus.load_opus(full_path)
                                print(f"[DEBUG] Opus library loaded successfully from: {full_path}")
                                opus_loaded = True
                                break
                            except Exception as e:
                                print(f"[DEBUG] Failed to load {full_path}: {e}")
                if opus_loaded:
                    break
                    
        except Exception as e:
            print(f"[DEBUG] Windows-specific Opus search failed: {e}")
    
    if not opus_loaded:
        print("[DEBUG] WARNING: Could not load Opus library. Voice features may not work properly.")
        print("[DEBUG] Please ensure opus.dll or libopus.dll is in your project directory or PATH.")
        print(f"[DEBUG] Current directory: {current_dir}")
        print(f"[DEBUG] Python executable: {sys.executable}")
        
        # List what DLL files are actually in the current directory
        dll_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.dll')]
        if dll_files:
            print(f"[DEBUG] DLL files found in current directory: {dll_files}")
        else:
            print("[DEBUG] No DLL files found in current directory")
else:
    print("[DEBUG] Opus library already loaded")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.members = True
intents.presences = True

class VoiceAssistantBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix=config.COMMAND_PREFIX,
            intents=intents,
            application_id=config.APPLICATION_ID
        )
        self.voice_handler = None
        self.whisper_handler = None
        self.chat_handler = None
        self.tts_handler = None
        self.singing_handler = None
        self.filter_handler = None
        self.processing_voice = {}
        self.is_ready_flag = asyncio.Event()
        self.patience_level = 20
        
        # AI Control states
        self.llm_enabled = True
        self.tts_enabled = True
        self.stt_enabled = True
        self.movement_enabled = True
        self.voice_enabled = True
        
        # Agentic system components
        self.memory_system = None
        self.discord_dm_handler = None
        self.websearch = None
        self.intent_router = None
        self.tool_guard = None
        self.tool_executor = None

    async def setup_hook(self):
        # Initialize core handlers
        self.voice_handler = VoiceHandler(self)
        self.whisper_handler = WhisperHandler()
        self.chat_handler = ChatHandler()
        self.tts_handler = TTSHandler()
        
        # Initialize singing handler
        from Sakura.singing_handler import SingingHandler
        self.singing_handler = SingingHandler(self, self.voice_handler)
        
        # Initialize vision tool
        from Sakura.vlm_handler import VLMHandler
        from Sakura.vision_tool import VisionTool
        from Sakura.live_vision_controller import LiveVisionController
        self.vlm_handler = VLMHandler()
        
        # Check VLM availability
        try:
            import requests
            vlm_check = requests.get("http://localhost:11434/api/tags", timeout=5)
            if vlm_check.status_code == 200:
                logger.info("VLM service is available")
            else:
                logger.warning(f"VLM service not available: {vlm_check.status_code}")
        except Exception as e:
            logger.warning(f"VLM service not reachable: {e}")
        
        self.vision_tool = VisionTool(self.vlm_handler)
        self.live_vision_controller = LiveVisionController(self.vlm_handler, self.vision_tool)
        self.filter_handler = FilterHandler()
        
        # Initialize agentic system
        logger.info("Initializing agentic system...")
        
        # Memory system
        self.memory_system = AdvancedMemorySystem()
        
        # Web search
        self.websearch = BraveWebSearch()
        
        # Discord DM handler (needs chat_handler, memory_system, llm_wrapper)
        self.discord_dm_handler = DiscordDMHandler(self, self.chat_handler, self.memory_system, self.chat_handler.llm_wrapper)
        
        # Tool guard and router
        self.tool_guard = ToolGuard()
        self.intent_router = IntentRouter()
        
        # Tool executor
        self.tool_executor = ToolExecutor(self, self.tool_guard, self.websearch)
        
        logger.info("Agentic system initialized successfully")
        
        # Register slash commands
        await self.tree.sync()
        
        # Note: Guilds will be available after the gateway connection is established
        # We'll connect to the LLM controller in the on_guild_available event instead
        logger.info("Bot initialization complete. Waiting for Discord gateway connection and guild data...")

    def is_fully_ready(self):
        return self.is_ready_flag.is_set() and self.is_ready()
    
    async def connect_to_llm_controller(self):
        """Connect to the LLM controller API"""
        try:
            import requests
            
            # Try to connect to the LLM controller
            response = requests.get("http://localhost:4000/status", timeout=5)
            if response.status_code == 200:
                # Send connection signal with bot info
                guild_data = [{"id": str(guild.id), "name": guild.name} for guild in self.guilds]
                logger.info(f"Sending guild data to controller: {len(self.guilds)} guilds")
                for guild in self.guilds:
                    logger.info(f"  - {guild.name} (id: {guild.id})")
                
                connection_data = {
                    "connected": True,
                    "guilds": guild_data,
                    "guild_count": len(self.guilds),
                    "bot_name": str(self.user) if self.user else "Sakura",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send connection signal to controller
                set_response = requests.post("http://localhost:4000/bot_connected", 
                                           json=connection_data, timeout=5)
                
                if set_response.status_code == 200:
                    logger.info("✅ Connected to LLM Controller API")
                    
                    # Now set the actual bot components
                    components_data = {
                        "llm_wrapper": self.chat_handler.llm_wrapper is not None,
                        "voice_handler": self.voice_handler is not None,
                        "tool_guard": self.tool_guard is not None
                    }
                    
                    # Send components status
                    components_response = requests.post("http://localhost:4000/set_components", 
                                                      json=components_data, timeout=5)
                    
                    if components_response.status_code == 200:
                        logger.info("✅ Bot components registered with LLM Controller")
                    
                    return True
                else:
                    logger.warning(f"Failed to connect to controller: {set_response.status_code}")
            else:
                logger.warning(f"LLM Controller not responding: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not connect to LLM Controller: {e}")
        
        return False

    def stop_current_message(self):
        if self.chat_handler:
            self.chat_handler.stop_current_message()
        if self.tts_handler:
            self.tts_handler.stop_current_speech()

    def clear_message_queue(self):
        if self.chat_handler:
            self.chat_handler.clear_message_queue()

    # AI Control methods
    def set_llm_enabled(self, enabled: bool):
        """Enable or disable LLM processing"""
        self.llm_enabled = enabled
        if self.chat_handler and hasattr(self.chat_handler, 'llm_wrapper') and self.chat_handler.llm_wrapper:
            self.chat_handler.llm_wrapper.llm_state.enabled = enabled
        print(f"[CONTROL] LLM {'enabled' if enabled else 'disabled'}")

    def set_tts_enabled(self, enabled: bool):
        """Enable or disable TTS processing"""
        self.tts_enabled = enabled
        print(f"[CONTROL] TTS {'enabled' if enabled else 'disabled'}")

    def set_stt_enabled(self, enabled: bool):
        """Enable or disable STT processing"""
        self.stt_enabled = enabled
        print(f"[CONTROL] STT {'enabled' if enabled else 'disabled'}")

    def set_movement_enabled(self, enabled: bool):
        """Enable or disable movement processing"""
        self.movement_enabled = enabled
        print(f"[CONTROL] Movement {'enabled' if enabled else 'disabled'}")

    def set_voice_enabled(self, enabled: bool):
        """Enable or disable voice processing"""
        self.voice_enabled = enabled
        print(f"[CONTROL] Voice {'enabled' if enabled else 'disabled'}")

    def get_ai_status(self):
        """Get current AI component status"""
        return {
            'llm_enabled': self.llm_enabled,
            'tts_enabled': self.tts_enabled,
            'stt_enabled': self.stt_enabled,
            'movement_enabled': self.movement_enabled,
            'voice_enabled': self.voice_enabled
        }

    def set_patience_level(self, level: int):
        """Set patience level for voice processing"""
        self.patience_level = max(0, min(100, level))
        print(f"[CONTROL] Patience level set to {self.patience_level}")

    # =============================================================================
    # AGENTIC SYSTEM METHODS
    # =============================================================================
    
    async def _create_tool_context(self, message) -> ToolContext:
        """Create tool context for a message"""
        # Get user info from memory system
        user_id = str(message.author.id)
        user_name = message.author.name  # Use actual Discord username, not display name
        display_name = message.author.display_name  # Keep display name separate
        
        is_whitelisted = False
        if self.tool_guard:
            whitelist = self.tool_guard.get_whitelist()
            # Check user ID and actual username for whitelist (not display name)
            is_whitelisted = (user_id in whitelist or user_name in whitelist)
            
            logger.info(f"Whitelist check for {display_name} ({user_id}): is_whitelisted={is_whitelisted}")
            logger.info(f"Whitelist contains: {whitelist}")
            logger.info(f"Checking: user_id={user_id}, username={user_name}, display_name={display_name}")
        
        return ToolContext(
            user_id=user_id,
            user_name=user_name,  # This is now the actual username
            channel_id=str(message.channel.id),
            channel_type="voice" if hasattr(message, 'voice') else "text",
            is_whitelisted=is_whitelisted,
        )
    
    def _route_message_agentic(self, text: str, context: ToolContext) -> RouterDecision:
        """Route a message through the intent router"""
        if not self.intent_router:
            return RouterDecision(
                action="CHAT",
                tool_call=None,
                confidence=1.0,
                reasoning="Router not available, defaulting to chat"
            )
        
        return self.intent_router.route_message(text, context)
    
    async def process_message_agentic(self, message: discord.Message):
        try:
            context = await self._create_tool_context(message)
            
            router_decision = self._route_message_agentic(message.content, context)
            
            logger.info(f"Router decision: {router_decision.action} (confidence: {router_decision.confidence:.2f})")
            
            if router_decision.action == "DM" and router_decision.tool_call:
                await self._handle_dm_request_agentic(message, context, router_decision.tool_call)
                
            elif router_decision.action == "CALL" and router_decision.tool_call:
                await self._handle_call_request_agentic(message, context, router_decision.tool_call)
                
            elif router_decision.action == "SEARCH" and router_decision.tool_call:
                await self._handle_search_request_agentic(message, context, router_decision.tool_call)
                
            else:
                await self._handle_chat_with_tool_awareness(message, context)
                
        except Exception as e:
            logger.error(f"Error in agentic message processing: {e}")
            await self._handle_normal_chat(message)
    
    async def _handle_dm_request_agentic(self, message, context: ToolContext, tool_call: ToolCall):
        """Handle DM request through agentic system"""
        try:
            logger.info(f"User {message.author.name} wants DM - letting LLM generate natural DM content")
            
            context = await self._create_tool_context(message)
            llm_response = await self._get_agentic_response(context, message.content)
            
            if llm_response:
                llm_tool_call = self.intent_router.parse_tool_call_from_llm(llm_response)
                
                if llm_tool_call and llm_tool_call.name == "DM":
                    logger.info(f"LLM generated DM tool call with natural content")
                    tool_response = await self.tool_executor.execute_tool_call(llm_tool_call, context)
                    
                    clean_response = self.intent_router.remove_tool_call_from_response(llm_response)
                    if clean_response.strip():
                        await message.channel.send(clean_response)
                    
                    self.tool_executor.log_tool_execution(llm_tool_call, tool_response, context)
                    
                else:
                    logger.info(f"LLM didn't generate DM tool call, using router fallback")
                    tool_response = await self.tool_executor.execute_tool_call(tool_call, context)
                    
                    if tool_response.ok:
                        await message.channel.send(f" I've sent you a private message about that!")
                    else:
                        await message.channel.send(f" I couldn't send you a DM: {tool_response.reason}")
                        
                    self.tool_executor.log_tool_execution(tool_call, tool_response, context)
            else:
                logger.warning(f"LLM returned None for DM request from {message.author.name}")
                await message.channel.send("I'm having trouble generating a DM right now.")
                
        except Exception as e:
            logger.error(f"Error handling DM request: {e}")
            await message.channel.send("Sorry, there was an error processing your DM request.")
    
    async def _handle_call_request_agentic(self, message, context: ToolContext, tool_call: ToolCall):
        """Handle voice channel call request through agentic system"""
        try:
            logger.info(f"User {message.author.name} wants to call someone - letting LLM generate natural call content")
            
            # Let the LLM generate the call content naturally using agentic system
            context = await self._create_tool_context(message)
            llm_response = await self._get_agentic_response(context, message.content)
            
            if llm_response:
                llm_tool_call = self.intent_router.parse_tool_call_from_llm(llm_response)
                
                if llm_tool_call and llm_tool_call.name == "CALL":
                    logger.info(f"LLM generated CALL tool call with natural content")
                    tool_response = await self.tool_executor.execute_tool_call(llm_tool_call, context)
                    
                    # Send the chat response (without the tool call JSON)
                    clean_response = self.intent_router.remove_tool_call_from_response(llm_response)
                    if clean_response.strip():
                        await message.channel.send(clean_response)
                    
                    if tool_response.ok:
                        logger.info(f"LLM-generated call succeeded for {message.author.name}")
                    else:
                        logger.info(f"LLM-generated call failed for {message.author.name}: {tool_response.reason}")
                else:
                    logger.info(f"LLM didn't generate CALL tool call, using router fallback")
                    tool_response = await self.tool_executor.execute_tool_call(tool_call, context)
                    
                    if tool_response.ok:
                        await message.channel.send(f"📞 I've sent a call invitation!")
                    else:
                        await message.channel.send(f"❌ I couldn't send the call invitation: {tool_response.reason}")
                        
        except Exception as e:
            logger.error(f"Error handling call request: {e}")
            await message.channel.send("❌ Sorry, there was an error processing your call request.")
    
    async def _handle_search_request_agentic(self, message, context: ToolContext, tool_call: ToolCall):
        """Handle search request through agentic system"""
        try:
            logger.info(f"User {message.author.name} wants to search - executing search tool")
            
            # Execute the search tool
            tool_response = await self.tool_executor.execute_tool_call(tool_call, context)
            
            if tool_response.ok:
                # Send the search results
                response = tool_response.content
                
                # Check if content should be filtered for Discord
                filtered_response = self.filter_handler.filter_message_completely(response)
                if filtered_response != response:
                    logger.info(f"Discord: Content filtered, sending 'Filtered.' instead of: '{response[:50]}...'")
                    await message.channel.send(filtered_response)
                else:
                    await message.channel.send(response)
                    logger.info("Search response sent to Discord successfully")
            else:
                logger.warning(f"No search results found for: {tool_call.args.get('query', 'unknown')}")
                await message.channel.send(f"Sorry, I couldn't find any results for that search.")
                
        except Exception as e:
            logger.error(f"Search error: {e}")
            await message.channel.send("Sorry, there was an error with the search.")
    
    async def _handle_chat_with_tool_awareness(self, message, context: ToolContext):
        """Handle normal chat but let LLM decide if it wants to use tools"""
        try:
            # Get normal chat response using agentic system
            response = await self._get_agentic_response(context, message.content)
            
            # Check if LLM generated any tool calls in the response
            if response and self.intent_router:
                tool_call = self.intent_router.parse_tool_call_from_llm(response)
                
                if tool_call:
                    logger.info(f"LLM spontaneously generated {tool_call.name} tool call")
                    
                    # Execute the tool call
                    tool_response = await self.tool_executor.execute_tool_call(tool_call, context)
                    
                    # Send the clean response (without tool call JSON)
                    clean_response = self.intent_router.remove_tool_call_from_response(response)
                    if clean_response.strip():
                        # Check if content should be filtered
                        filtered_response = self.filter_handler.filter_message_completely(clean_response)
                        if filtered_response != clean_response:
                            logger.info(f"Discord: Content filtered, sending 'Filtered.' instead of: '{clean_response[:50]}...'")
                            await message.channel.send(filtered_response)
                        else:
                            await message.channel.send(clean_response)
                    
                    return
            
            # Normal chat response without tools
            if response:
                filtered_response = self.filter_handler.filter_message_completely(response)
                if filtered_response != response:
                    logger.info(f"Discord: Content filtered, sending 'Filtered.' instead of: '{response[:50]}...'")
                    await message.channel.send(filtered_response)
                else:
                    await message.channel.send(response)
            
        except Exception as e:
            logger.error(f"Chat handling error: {e}")
            await message.channel.send("Sorry, I'm having trouble responding right now.")
    
    async def _handle_normal_chat(self, message):
        """Fallback to normal chat handling"""
        try:
            context = await self._create_tool_context(message)
            response = await self._get_agentic_response(context, message.content)
            
            if response:
                # Check if content should be filtered
                filtered_response = self.filter_handler.filter_message_completely(response)
                if filtered_response != response:
                    logger.info(f"Discord: Content filtered, sending 'Filtered.' instead of: '{response[:50]}...'")
                    await message.channel.send(filtered_response)
                else:
                    await message.channel.send(response)
                    
        except Exception as e:
            logger.error(f"Normal chat error: {e}")
            await message.channel.send("Sorry, I'm having trouble responding right now.")

    # Voice message processing with agentic system
    async def _get_agentic_response(self, context, text, suggested_tool=None):
        """Generate response using the agentic system with proper context"""
        try:
            # Update control panel status
            self.chat_handler.current_message = {"status": "processing", "content": text}
            
            # Build messages for the LLM with proper agentic context
            user_message = text
            
            # If we have a suggested tool from the router, guide the LLM with emotional context
            if suggested_tool:
                if suggested_tool == "DM":
                    user_message += "\n\n<SYSTEM_HINT>User wants you to send them a DM. Be direct and natural like texting a friend. Don't explain why you're messaging them, just say what's on your mind. Use DM tool with authentic message content.</SYSTEM_HINT>"
                elif suggested_tool == "CALL":
                    user_message += "\n\n<SYSTEM_HINT>User wants to call someone to voice chat. Be casual and direct about inviting them. Use CALL tool with the person's name and a natural invitation message.</SYSTEM_HINT>"
                elif suggested_tool == "SEARCH":
                    user_message += "\n\n<SYSTEM_HINT>User needs information. Search for what they're asking about and share what you find in a natural way. Use SEARCH tool with relevant query.</SYSTEM_HINT>"
            
            # Vision capture is now handled before LLM response in voice processing
            
            # Build messages with vision context
            messages = []
            
            # Add vision context if available (prioritize live vision over static vision)
            vision_context = None
            is_stale = False
            
            # Check live vision first
            if hasattr(self, 'live_vision_controller') and self.live_vision_controller.live_enabled:
                vision_context = self.live_vision_controller.get_vision_context()
                if vision_context:
                    is_stale = not self.live_vision_controller.is_fresh
            
            # Fallback to static vision if live vision not available
            if not vision_context and hasattr(self, 'vision_tool') and self.vision_tool.enabled:
                logger.info(f"Vision tool enabled: {self.vision_tool.enabled}")
                vision_context, is_stale = self.vision_tool.get_vision_context_with_staleness()
                logger.info(f"Vision context from tool: {vision_context is not None}, is_stale: {is_stale}")
            elif not vision_context:
                logger.info(f"Vision tool not enabled: {hasattr(self, 'vision_tool') and self.vision_tool.enabled if hasattr(self, 'vision_tool') else 'No vision tool'}")
            
            if vision_context:
                # Add vision system message
                if hasattr(self, 'live_vision_controller') and self.live_vision_controller.live_enabled:
                    vision_system_msg = "You are Sakura: concise, honest, and grounded. You can see what's on the user's screen through VLM_SUMMARY_JSON. Respond to what you see in the image. Always mention the capture timestamp from VLM_SUMMARY_JSON. If age_secs > fresh_secs, say the context may be stale. Never infer details absent from VLM_SUMMARY_JSON. If sensitive_indicators are present, warn briefly and avoid repeating sensitive strings."
                else:
                    vision_system_msg = "You are Sakura: witty, concise, and honest. You can see what's on the user's screen through VLM_SUMMARY_JSON. Respond to what you see in the image. When referencing what you see, always mention the capture timestamp from VLM_SUMMARY_JSON. Never invent details not in VLM_SUMMARY_JSON. If details are missing, say so and propose a new capture or a zoomed region. If VLM_SUMMARY_JSON indicates sensitive_indicators, warn briefly and avoid repeating sensitive details. Keep responses crisp and actionable."
                
                if is_stale:
                    vision_system_msg += f"\n\nWARNING: The vision context is stale. Mention this when referencing the image."
                
                logger.info(f"Adding vision context to LLM: {vision_context[:200]}...")
                messages.append({"role": "system", "content": vision_system_msg})
                messages.append({"role": "system", "content": vision_context})
            else:
                logger.info("No vision context available for LLM")
            
            # Add vision context to user message if available
            if vision_context:
                user_message_with_vision = f"{user_message}\n\n[VISION CONTEXT AVAILABLE: You can see what's on the user's screen. Respond to what you see in the image.]"
                messages.append({"role": "user", "content": user_message_with_vision})
            else:
                messages.append({"role": "user", "content": user_message})
            
            # Get memories for this user
            memories = None
            if hasattr(self, 'memory_system') and self.memory_system:
                try:
                    person = self.memory_system.get_person(context.user_name)
                    if person:
                        memories = {
                            "short_term": [memory.content for memory in self.memory_system.short_term_memories],
                            "long_term": [memory.content for memory in self.memory_system.long_term_memories]
                        }
                except Exception as e:
                    logger.error(f"Error getting memories: {e}")
            
            # Generate response using LLM wrapper directly (not old chat handler)
            response = await self.chat_handler.llm_wrapper.get_response(messages, memories)
            
            # Save memories like the old chat handler did
            if response and hasattr(self, 'memory_system') and self.memory_system:
                try:
                    # Import memory extraction function
                    from Sakura.chat_handler import extract_memories
                    
                    # Extract memory blocks and get clean response
                    cleaned_response, memory_items = extract_memories(response)
                    
                    # Get the person for memory saving
                    person = self.memory_system.get_person(context.user_name)
                    if person:
                        # Save short-term interaction memory
                        memory_safe_response = self.chat_handler._clean_response_for_memory(cleaned_response)
                        interaction = f"{person.name} said: {text}\nSakura replied: {memory_safe_response}"
                        try:
                            self.memory_system.add_short_term_memory(interaction, [person.name])
                            logger.info(f"Saved short-term memory for {person.name}")
                        except Exception as me:
                            logger.error(f"Error saving short-term memory: {me}")
                        
                        # Save long-term memory blocks if any were extracted
                        for m in memory_items:
                            try:
                                self.memory_system.add_long_term_memory(m, "fact", [person.name])
                                logger.info(f"Saved long-term memory: {m[:50]}...")
                            except Exception as me:
                                logger.error(f"Error saving long-term memory: {me}")
                except Exception as e:
                    logger.error(f"Error in memory saving process: {e}")
            
            # Update control panel with completed response
            if response:
                self.chat_handler.current_message = {"status": "completed", "content": response}
            
            return response
            
        except Exception as e:
            logger.error(f"Error in agentic response generation: {e}")
            self.chat_handler.current_message = {"status": "error", "content": str(e)}
            return "Sorry, I encountered an error generating a response."

    async def handle_voice_message(self, user, text: str):
        """Handle voice messages through agentic system"""
        try:
            # Get the voice channel
            voice_channel = None
            for guild in self.guilds:
                for vc in guild.voice_channels:
                    if user in vc.members:
                        voice_channel = vc
                        break
                if voice_channel:
                    break
            
            if not voice_channel:
                logger.warning(f"Could not find voice channel for user {user.display_name}")
                return
            
            # Create a mock message object for voice processing
            class VoiceMessage:
                def __init__(self, author, content, guild, channel):
                    self.author = author
                    self.content = content
                    self.guild = guild
                    self.channel = channel
            
            voice_message = VoiceMessage(user, text, voice_channel.guild, voice_channel)
            
            # Create tool context for voice message
            context = await self._create_tool_context(voice_message)
            context.channel_type = "voice"  # Override to voice
            
            # Route the voice message
            router_decision = self._route_message_agentic(text, context)
            
            logger.info(f"Voice agentic router decision: {router_decision.action} (confidence: {router_decision.confidence:.2f})")
            logger.info(f"Router decision details: action={router_decision.action}, tool_call={router_decision.tool_call}, confidence={router_decision.confidence}, reasoning={router_decision.reasoning}")
            
            # Handle based on router decision
            if router_decision.action == "DM" and router_decision.tool_call:
                logger.info(f"Voice user {user.display_name} wants DM - letting LLM generate natural DM content")
                
                # Let the LLM generate the DM content naturally using agentic system
                try:
                    llm_response = await self._get_agentic_response(context, text, "DM")
                    
                    if llm_response:
                        # Check if LLM generated a tool call with natural DM content
                        llm_tool_call = self.intent_router.parse_tool_call_from_llm(llm_response)
                        
                        if llm_tool_call and llm_tool_call.name == "DM":
                            # Use the LLM's tool call (which should have natural message content)
                            logger.info(f"LLM generated DM tool call with natural content")
                            
                            # Log tool call attempt
                            self.chat_handler.log_tool_call(llm_tool_call.name, llm_tool_call.args, False, "Executing...")
                            
                            tool_response = await self.tool_executor.execute_tool_call(llm_tool_call, context)
                            
                            # Get the clean chat response (without JSON tool call) for TTS
                            logger.info(f"[DEBUG] Original LLM response: {repr(llm_response)}")
                            clean_response = self.intent_router.remove_tool_call_from_response(llm_response)
                            logger.info(f"[DEBUG] Clean response for TTS: {repr(clean_response)}")
                            if clean_response.strip():
                                await self.tts_handler.generate_and_play_speech(clean_response, self)
                            else:
                                logger.warning("Clean response is empty after tool call removal")
                            
                            if tool_response.ok:
                                logger.info(f"LLM-generated DM succeeded for {user.display_name}")
                                # Update tool call log with success
                                self.chat_handler.log_tool_call(llm_tool_call.name, llm_tool_call.args, True, "Success")
                            else:
                                logger.info(f"LLM-generated DM failed for {user.display_name}: {tool_response.reason}")
                                # Update tool call log with failure
                                self.chat_handler.log_tool_call(llm_tool_call.name, llm_tool_call.args, False, tool_response.reason)
                        else:
                            # LLM didn't generate a tool call - clean and give natural response
                            logger.info(f"LLM didn't generate DM tool call, giving natural response instead")
                            clean_response = self.intent_router.remove_tool_call_from_response(llm_response)
                            if clean_response.strip():
                                await self.tts_handler.generate_and_play_speech(clean_response, self)
                                
                except Exception as voice_dm_error:
                    logger.error(f"Error in voice DM handling: {voice_dm_error}")
                    response = "Sorry, there was an error processing your DM request."
                    await self.tts_handler.generate_and_play_speech(response, self)
                    
            elif router_decision.action == "CALL" and router_decision.tool_call:
                logger.info(f"Voice user {user.display_name} wants to call someone - letting LLM generate natural call content")
                
                # Let the LLM generate the call content naturally using agentic system
                try:
                    llm_response = await self._get_agentic_response(context, text, "CALL")
                    
                    if llm_response:
                        # Check if LLM generated a tool call with natural call content
                        llm_tool_call = self.intent_router.parse_tool_call_from_llm(llm_response)
                        
                        if llm_tool_call and llm_tool_call.name == "CALL":
                            # Use the LLM's tool call (which should have natural message content)
                            logger.info(f"LLM generated CALL tool call with natural content")
                            tool_response = await self.tool_executor.execute_tool_call(llm_tool_call, context)
                            
                            # Get the clean chat response (without JSON tool call) for TTS
                            clean_response = self.intent_router.remove_tool_call_from_response(llm_response)
                            if clean_response.strip():
                                await self.tts_handler.generate_and_play_speech(clean_response, self)
                            
                            if tool_response.ok:
                                logger.info(f"LLM-generated call succeeded for {user.display_name}")
                            else:
                                logger.info(f"LLM-generated call failed for {user.display_name}: {tool_response.reason}")
                        else:
                            # Fallback to router tool call if LLM didn't generate one
                            logger.info(f"LLM didn't generate CALL tool call, using router fallback")
                            tool_response = await self.tool_executor.execute_tool_call(router_decision.tool_call, context)
                            
                            if tool_response.ok:
                                response = "I've sent a call invitation!"
                                await self.tts_handler.generate_and_play_speech(response, self)
                            else:
                                response = f"Sorry, I couldn't send the call invitation: {tool_response.reason}"
                                await self.tts_handler.generate_and_play_speech(response, self)
                                
                except Exception as voice_call_error:
                    logger.error(f"Error in voice call handling: {voice_call_error}")
                    response = "Sorry, there was an error processing your call request."
                    await self.tts_handler.generate_and_play_speech(response, self)
                    
            elif router_decision.action == "SEARCH" and router_decision.tool_call:
                logger.info(f"Executing search tool")
                
                # Execute search tool
                tool_response = await self.tool_executor.execute_tool_call(router_decision.tool_call, context)
                
                if tool_response.ok:
                    # Use the search result directly for TTS
                    response = tool_response.content
                    
                    # Filter for TTS but preserve original for display
                    if self.filter_handler.is_filtered(response):
                        tts_response = "Filtered."
                    else:
                        tts_response = response
                    
                    await self.tts_handler.generate_and_play_speech(tts_response, self)
                else:
                    # Search failed
                    response = f"Sorry, I couldn't search for that: {tool_response.reason}"
                    await self.tts_handler.generate_and_play_speech(response, self)
                    
            else:
                # Normal chat for voice - generate response and play TTS
                logger.info(f"Generating normal chat response for voice user {user.display_name} (action: {router_decision.action})")
                
                # Capture vision context before LLM response if vision tool is enabled
                if hasattr(self, 'vision_tool') and self.vision_tool.enabled:
                    try:
                        vision_result = await self.vision_tool.process_vision_request()
                        if vision_result['success']:
                            logger.info("Vision capture successful before LLM response")
                        else:
                            logger.warning(f"Vision capture failed: {vision_result['error']}")
                    except Exception as e:
                        logger.error(f"Vision processing error: {e}")
                
                # Get normal chat response using agentic system
                response = await self._get_agentic_response(context, text)
                
                if response:
                    # Check if LLM generated any tool calls in the response
                    if self.intent_router:
                        tool_call = self.intent_router.parse_tool_call_from_llm(response)
                        
                        if tool_call:
                            logger.info(f"Voice LLM spontaneously generated {tool_call.name} tool call")
                            
                            # Execute the tool call
                            tool_response = await self.tool_executor.execute_tool_call(tool_call, context)
                            
                            # Get clean response for TTS (without tool call JSON)
                            clean_response = self.intent_router.remove_tool_call_from_response(response)
                            if clean_response.strip():
                                # Filter for TTS but preserve original for display  
                                if self.filter_handler.is_filtered(clean_response):
                                    tts_response = "Filtered."
                                else:
                                    tts_response = clean_response
                                
                                await self.tts_handler.generate_and_play_speech(tts_response, self)
                            return
                    
                    # Normal response - filter for TTS
                    if self.filter_handler.is_filtered(response):
                        tts_response = "Filtered."
                    else:
                        tts_response = response
                    
                    await self.tts_handler.generate_and_play_speech(tts_response, self)
                    
        except Exception as e:
            logger.error(f"Error in voice agentic processing: {e}")
            # Fallback to normal voice processing using agentic system
            try:
                fallback_context = await self._create_tool_context(voice_message)
                response = await self._get_agentic_response(fallback_context, text)
                
                if response:
                    # Filter for TTS
                    if self.filter_handler.is_filtered(response):
                        tts_response = "Filtered."
                    else:
                        tts_response = response
                    
                    await self.tts_handler.generate_and_play_speech(tts_response, self)
                    
            except Exception as fallback_error:
                logger.error(f"Fallback voice processing also failed: {fallback_error}")

bot = VoiceAssistantBot()

# Cache voice_recv module on bot for API integration access
bot._voice_recv_module = voice_recv

# Track if we've already connected to LLM controller
_llm_controller_connected = False

@bot.event
async def on_ready():
    print(f'{bot.user} is connected to Discord!')
    print(f'Connected to {len(bot.guilds)} guilds:')
    for guild in bot.guilds:
        print(f'- {guild.name} (id: {guild.id})')
    
    print("\n" + "="*60)
    print("🔊 VOICE PROCESSING INFORMATION:")
    print("• OpusError messages from voice_recv.router are NORMAL")
    print("• These errors occur when Discord sends corrupted packets")
    print("• They do NOT affect bot functionality or audio processing")
    print("• The bot will continue to work despite these errors")
    print("="*60 + "\n")
    
    # Connect to LLM controller now that we have guild data
    global _llm_controller_connected
    if not _llm_controller_connected and len(bot.guilds) > 0:
        logger.info("Guilds are available, connecting to LLM controller...")
        await bot.connect_to_llm_controller()
        _llm_controller_connected = True
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="your voice"
        )
    )
    
    bot.is_ready_flag.set()

@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages through the agentic system"""
    # Ignore messages from bots (including ourselves)
    if message.author.bot:
        return
    
    # Process commands first (this is required for Discord.py commands to work)
    await bot.process_commands(message)
    
    # Skip processing if the message is a command
    if message.content.startswith(bot.command_prefix):
        return
    
    # Handle DMs separately
    if not message.guild:
        # This is a DM - handle through DM system
        logger.info(f"DM received from {message.author.name} ({message.author.id}): {message.content}")
        if hasattr(bot, 'discord_dm_handler') and bot.discord_dm_handler:
            await bot.discord_dm_handler.handle_incoming_dm(message)
        return
    
    # Only process guild messages if bot has permission
    # Check if bot has permission to send messages in this channel
    permissions = message.channel.permissions_for(message.guild.me)
    if not permissions.send_messages:
        return
    
    # Process message through agentic system
    logger.info(f"Text message from {message.author.name} ({message.author.id}): {message.content}")
    await bot.process_message_agentic(message)

@bot.command(name='join', help='Bot joins your voice channel')
async def join(ctx):
    if not ctx.author.voice:
        await ctx.send("You need to be in a voice channel first!")
        return

    channel = ctx.author.voice.channel
    try:
        if ctx.voice_client is not None:
            await ctx.voice_client.move_to(channel)
        else:
            import logging
            voice_recv_logger = logging.getLogger('discord.ext.voice_recv.router')
            voice_recv_logger.setLevel(logging.CRITICAL)
            
            await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=20.0, reconnect=True)
        
        guild_id = ctx.guild.id
        bot.processing_voice[guild_id] = True
        
        await ctx.send(f"Joined {channel.name}! I'm listening and ready to chat!")
        print(f"[DEBUG] Opus errors from voice_recv.router are expected and can be ignored - they don't affect functionality")
        
        try:
            await bot.voice_handler.process_voice(ctx)
        except Exception as e:
            print(f"Error in voice processing: {e}")
            bot.chat_handler.current_message = {"status": "error", "content": str(e)}
            await ctx.send("An error occurred while processing voice. Please try rejoining.")
            
    except Exception as e:
        print(f"Error joining voice channel: {e}")
        await ctx.send("Failed to join the voice channel. Please try again.")

@bot.command(name='leave', help='Bot leaves the voice channel')
async def leave(ctx):
    if ctx.voice_client:
        guild_id = ctx.guild.id
        if guild_id in bot.processing_voice:
            bot.processing_voice[guild_id] = False
        
        if hasattr(ctx.voice_client, 'stop_listening'):
            ctx.voice_client.stop_listening()
            
        await bot.voice_handler.cleanup(guild_id)
        await ctx.voice_client.disconnect()
        await ctx.send("Left the voice channel!")
        
        bot.chat_handler.current_message = {"status": "idle", "content": ""}
        bot.chat_handler.next_message = {"status": "queued", "content": ""}
    else:
        await ctx.send("I'm not in a voice channel!")

@bot.event
async def on_voice_state_update(member, before, after):
    if member == bot.user and after.channel is None:
        guild_id = before.channel.guild.id
        if guild_id in bot.processing_voice:
            bot.processing_voice[guild_id] = False
        await bot.voice_handler.cleanup(guild_id)
        
        bot.chat_handler.current_message = {"status": "idle", "content": ""}
        bot.chat_handler.next_message = {"status": "queued", "content": ""}

@bot.command(name='reset', help='Reset the conversation history')
async def reset(ctx):
    bot.chat_handler.reset_conversation(ctx.guild.id, ctx.channel.id)
    await ctx.send("Conversation history has been reset!")

@bot.command(name='forcereset', help='Force reset everything and start fresh')
async def force_reset(ctx):
    bot.chat_handler.force_reset()
    await ctx.send("Everything has been force reset! Starting completely fresh.")

# Singing Commands
@bot.command(name='sing', help='Start singing a song')
async def sing(ctx, song_name: str = None):
    """Start singing a specific song"""
    if not hasattr(bot, 'singing_handler'):
        await ctx.send("❌ Singing feature not available")
        return
    
    if not ctx.voice_client:
        await ctx.send("❌ I need to be in a voice channel to sing!")
        return
    
    if not song_name:
        # Show available songs
        songs = bot.singing_handler.get_available_songs()
        if songs:
            song_list = "\n".join([f"• {song['name']}" for song in songs])
            await ctx.send(f"Available songs:\n{song_list}\n\nUse `!sing <song_name>` to start singing")
        else:
            await ctx.send("❌ No songs available")
        return
    
    # Find the song
    songs = bot.singing_handler.get_available_songs()
    song_file = None
    for song in songs:
        if song['name'].lower() == song_name.lower() or song['filename'].lower() == song_name.lower():
            song_file = song['filename']
            break
    
    if not song_file:
        await ctx.send(f"❌ Song '{song_name}' not found")
        return
    
    # Start singing
    success = await bot.singing_handler.start_singing(song_file)
    if success:
        await ctx.send(f"🎵 Started singing: {song_name}")
    else:
        await ctx.send(f"❌ Failed to start singing: {song_name}")

@bot.command(name='stopsing', help='Stop singing')
async def stop_sing(ctx):
    """Stop current singing"""
    if not hasattr(bot, 'singing_handler'):
        await ctx.send("❌ Singing feature not available")
        return
    
    success = await bot.singing_handler.stop_singing()
    if success:
        await ctx.send("🛑 Stopped singing")
    else:
        await ctx.send("❌ Failed to stop singing")

# Vision Tool Commands
@bot.command(name='vlm', help='Toggle vision tool on/off')
async def vlm_toggle(ctx, action: str = None):
    """Toggle vision tool or show status"""
    if not hasattr(bot, 'vision_tool'):
        await ctx.send("❌ Vision tool not available")
        return
    
    if action == "on":
        bot.vision_tool.set_enabled(True)
        await ctx.send("👁️ Vision Tool enabled")
    elif action == "off":
        bot.vision_tool.set_enabled(False)
        await ctx.send("👁️ Vision Tool disabled")
    elif action == "status":
        status = bot.vision_tool.get_status()
        vlm_status = status['vlm_status']
        
        status_msg = f"**Vision Tool Status:**\n"
        status_msg += f"Enabled: {status['enabled']}\n"
        status_msg += f"Stale threshold: {status['stale_after_secs']}s\n"
        status_msg += f"Capture preference: {status['capture_preference']}\n"
        status_msg += f"Has summary: {vlm_status['has_summary']}\n"
        if vlm_status['has_summary']:
            status_msg += f"Last capture: {vlm_status['age_seconds']}s ago\n"
            status_msg += f"Source: {vlm_status['capture_source']}\n"
            status_msg += f"Confidence: {vlm_status['last_confidence']:.2f}\n"
            status_msg += f"Stale: {vlm_status['is_stale']}"
        
        await ctx.send(status_msg)
    else:
        await ctx.send("Usage: `/vlm on|off|status`")

@bot.command(name='see', help='Capture and analyze image')
async def see_capture(ctx, source: str = None):
    """Capture image and analyze with VLM"""
    if not hasattr(bot, 'vision_tool'):
        await ctx.send("❌ Vision tool not available")
        return
    
    if not bot.vision_tool.enabled:
        await ctx.send("❌ Vision tool is disabled. Use `/vlm on` to enable.")
        return
    
    # Show typing indicator
    async with ctx.typing():
        try:
            result = await bot.vision_tool.process_vision_request(source)
            
            if result['success']:
                summary = result['summary']
                timestamp = summary.get('timestamp_iso', 'Unknown')
                confidence = summary.get('confidence', 0.0)
                
                status_msg = f"✅ {result['message']}\n"
                status_msg += f"📊 Confidence: {confidence:.2f}\n"
                status_msg += f"🕒 Timestamp: {timestamp}\n"
                status_msg += f"📝 Summary: {summary.get('high_level_summary', 'No summary')}"
                
                await ctx.send(status_msg)
            else:
                await ctx.send(f"❌ Capture failed: {result['error']}")
                
        except Exception as e:
            await ctx.send(f"❌ Vision capture error: {str(e)}")

# Live Vision Commands
@bot.command(name='live', help='Control live vision mode')
async def live_vision(ctx, action: str = None):
    """Control live vision mode"""
    if not hasattr(bot, 'live_vision_controller'):
        await ctx.send("❌ Live vision not available")
        return
    
    if action == "on":
        success = await bot.live_vision_controller.start_live_vision()
        if success:
            await ctx.send("👁️ Live vision enabled - continuous capture started")
        else:
            await ctx.send("❌ Failed to start live vision")
    elif action == "off":
        success = await bot.live_vision_controller.stop_live_vision()
        if success:
            await ctx.send("👁️ Live vision disabled")
        else:
            await ctx.send("❌ Failed to stop live vision")
    elif action == "status":
        status = bot.live_vision_controller.get_status()
        status_msg = f"**Live Vision Status:**\n"
        status_msg += f"Enabled: {status['live_enabled']}\n"
        status_msg += f"Running: {status['capture_running']}\n"
        status_msg += f"Fresh threshold: {status['fresh_secs']}s\n"
        status_msg += f"Rate limit: {status['max_rate_per_sec']}/s\n"
        status_msg += f"Redaction: {status['redaction_enabled']}\n"
        status_msg += f"Source: {status['source']}\n"
        status_msg += f"Summaries: {status['summary_count']}\n"
        if status['has_summary']:
            status_msg += f"Last capture: {status['age_secs']}s ago\n"
            status_msg += f"Fresh: {status['is_fresh']}\n"
            status_msg += f"Confidence: {status['last_confidence']:.2f}"
        else:
            status_msg += "No captures yet"
        
        await ctx.send(status_msg)
    elif action == "pause":
        success = await bot.live_vision_controller.privacy_pause()
        if success:
            await ctx.send("⏸️ Privacy pause activated - capture stopped and buffers cleared")
        else:
            await ctx.send("❌ Failed to pause live vision")
    else:
        await ctx.send("Usage: `/live on|off|status|pause`")

@bot.command(name='redact', help='Toggle redaction for live vision')
async def toggle_redaction(ctx, action: str = None):
    """Toggle redaction for live vision"""
    if not hasattr(bot, 'live_vision_controller'):
        await ctx.send("❌ Live vision not available")
        return
    
    if action == "on":
        bot.live_vision_controller.toggle_redaction(True)
        await ctx.send("🔒 Redaction enabled")
    elif action == "off":
        bot.live_vision_controller.toggle_redaction(False)
        await ctx.send("🔓 Redaction disabled")
    else:
        await ctx.send("Usage: `/redact on|off`")

@bot.tree.command(name="ping", description="Check the bot's latency")
async def ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(f"Pong! 🏓\nLatency: {latency}ms")



if __name__ == "__main__":
    # Run the Discord bot
    # LLM Controller API runs separately via llm_controller.py
    
    # Try to connect to LLM controller if it's running
    def try_connect_controller():
        try:
            import requests
            response = requests.get("http://localhost:4000/status", timeout=2)
            if response.status_code == 200:
                print("✅ LLM Controller detected - bot will connect automatically")
            else:
                print("⚠️ LLM Controller not responding - run llm_controller.py separately")
        except:
            print("⚠️ LLM Controller not running - run llm_controller.py separately")
    
    # Start the original API server for direct bot integration
    def start_bot_api():
        try:
            import sys
            import os
            import time
            
            # Wait for bot to initialize
            time.sleep(3)
            
            # Add sakura-frontend to path
            frontend_path = os.path.join(os.path.dirname(__file__), 'sakura-frontend')
            if frontend_path not in sys.path:
                sys.path.insert(0, frontend_path)
            
            # Check if port 5000 is already in use
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', 5000)) == 0:
                    print("⚠️ Port 5000 already in use - skipping API server")
                    return
            
            from api_integration import start_api_server
            print("🌐 Starting Discord Bot API server on port 5000...")
            start_api_server(bot, port=5000, host='0.0.0.0')  # Listen on all interfaces
        except ImportError as e:
            print(f"⚠️ API integration not found: {e} - remote control may use fallback")
        except Exception as e:
            print(f"⚠️ Failed to start bot API server: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        # Start API server in background thread
        import threading
        api_thread = threading.Thread(target=start_bot_api, daemon=True)
        api_thread.start()
        
        try_connect_controller()
        bot.run(config.DISCORD_TOKEN)
    except KeyboardInterrupt:
        print("Bot stopped by user.")
    except Exception as e:
        print(f"Bot error: {e}")
        raise