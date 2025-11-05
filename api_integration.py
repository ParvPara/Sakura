#!/usr/bin/env python3
"""
API Integration for Sakura Bot - Remote Access Support
Add this to your main bot to enable remote control panel access.
"""

import asyncio
import threading
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
import discord
from datetime import datetime

# Add the parent directory to the path to find schemas.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
try:
    from schemas import ToolContext
except ImportError as e:
    print(f"Warning: Could not import ToolContext: {e}")
    ToolContext = None

logger = logging.getLogger(__name__)

# Global Flask app instance to prevent duplicate route registration
_app_instance = None
_bot_instance = None

def set_bot_instance(bot):
    """Set the bot instance from the main application"""
    global _bot_instance
    _bot_instance = bot
    logger.info("Bot instance set for API server")

def reset_api_server():
    """Reset the API server instance (for testing/debugging)"""
    global _app_instance, _bot_instance
    _app_instance = None
    _bot_instance = None
    logger.info("API server instance reset")

def force_recreate_app():
    """Force recreation of Flask app to register new routes"""
    global _app_instance
    _app_instance = None
    logger.info("Forced Flask app instance reset for new routes")

def create_api_server():
    """Create Flask API server for remote control panel access"""
    global _app_instance
    
    # Always create new app instance for development
    # if _app_instance is not None:
    #     logger.info("Returning existing Flask app instance")
    #     return _app_instance, set_bot_instance
    
    app = Flask(__name__)
    _app_instance = app
    
    # Configure CORS for remote access (no credentials needed with API key auth)
    CORS(app, 
         origins=["https://*.github.io", "https://*.pages.dev", "https://your-domain.com", "https://control.your-domain.com", "https://api.your-domain.com"],
         allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-API-Key"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         supports_credentials=False,
         max_age=3600)
    
    def require_api_key(f):
        """Decorator to require API key authentication"""
        import functools
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            expected_key = os.getenv('API_KEY')
            
            if not expected_key:
                logger.warning("API_KEY environment variable not set")
                return jsonify({"error": "API key not configured"}), 500
            
            if not api_key or api_key != expected_key:
                logger.warning(f"Invalid API key attempt from {request.remote_addr}")
                return jsonify({"error": "Invalid API key"}), 401
            
            return f(*args, **kwargs)
        return wrapper
    
    def optional_api_key(f):
        """Decorator for endpoints that work with or without API key"""
        import functools
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            api_key = request.headers.get('X-API-Key')
            expected_key = os.getenv('API_KEY')
            
            # If API key is provided and expected key is set, validate it
            if api_key and expected_key and api_key != expected_key:
                logger.warning(f"Invalid API key attempt from {request.remote_addr}")
                return jsonify({"error": "Invalid API key"}), 401
            
            # If no API key is provided and no expected key is set, allow access (local development)
            if not api_key and not expected_key:
                logger.info(f"Local development access to {request.endpoint} from {request.remote_addr}")
            
            return f(*args, **kwargs)
        return wrapper
    
    # Health check endpoint (no auth required)
    @app.route('/api/status')
    @optional_api_key
    def api_status():
        return jsonify({
            "status": "online",
            "bot_connected": _bot_instance is not None,
            "version": "1.0.0",
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        })
    
    # AI Status endpoints
    @app.route('/api/ai/status')
    @require_api_key
    def ai_status():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Get AI status from your bot's state
            return jsonify({
                "enabled": getattr(_bot_instance, 'llm_enabled', True),
                "thinking": getattr(_bot_instance, 'ai_thinking', False),
                "speaking": getattr(_bot_instance, 'ai_speaking', False),
                "patience": getattr(_bot_instance, 'patience_level', 20)
            })
        except Exception as e:
            logger.error(f"Error getting AI status: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/ai/control', methods=['POST'])
    @require_api_key
    def ai_control():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            data = request.get_json() or {}
            action = data.get('action', 'toggle')
            
            if action == 'toggle':
                # Toggle LLM enabled state
                _bot_instance.llm_enabled = not getattr(_bot_instance, 'llm_enabled', True)
                return jsonify({
                    "enabled": _bot_instance.llm_enabled,
                    "message": f"AI {'enabled' if _bot_instance.llm_enabled else 'disabled'}"
                })
            elif action == 'enable':
                _bot_instance.llm_enabled = True
                return jsonify({"enabled": True, "message": "AI enabled"})
            elif action == 'disable':
                _bot_instance.llm_enabled = False
                return jsonify({"enabled": False, "message": "AI disabled"})
            else:
                return jsonify({"error": "Invalid action"}), 400
                
        except Exception as e:
            logger.error(f"Error controlling AI: {e}")
            return jsonify({"error": str(e)}), 500
    
    # AI status endpoint
    @app.route('/ai/status')
    @optional_api_key
    def get_ai_status():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
        
        try:
            llm_enabled = hasattr(_bot_instance, 'llm_wrapper') and _bot_instance.llm_wrapper is not None
            return jsonify({
                "llm_enabled": llm_enabled,
                "mode": "chat"
            })
        except Exception as e:
            logger.error(f"Error getting AI status: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Message status endpoint
    @app.route('/api/messages/status')
    @app.route('/messages/status')
    @optional_api_key
    def message_status():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Get current message status from chat handler or provide default
            current_msg = {"status": "idle", "content": "Ready to respond..."}
            next_msg = {"status": "idle", "content": ""}
            tool_calls = []
            
            # Check if unfiltered view is requested
            show_unfiltered = request.args.get('unfiltered', 'false').lower() == 'true'
            
            # Get status from chat handler
            if hasattr(_bot_instance, 'chat_handler') and _bot_instance.chat_handler:
                if hasattr(_bot_instance.chat_handler, 'current_message'):
                    current_raw = _bot_instance.chat_handler.current_message
                    current_msg = {
                        "status": current_raw.get("status", "idle"),
                        "content": current_raw.get("content", "Ready to respond...")  # RAW unfiltered content
                    }
                
                if hasattr(_bot_instance.chat_handler, 'next_message'):
                    next_raw = _bot_instance.chat_handler.next_message
                    next_msg = {
                        "status": next_raw.get("status", "idle"),
                        "content": next_raw.get("content", "")  # RAW unfiltered content
                    }
            
            # If unfiltered view is requested, also show LLM's unfiltered response
            if show_unfiltered and hasattr(_bot_instance, 'llm_wrapper') and _bot_instance.llm_wrapper:
                if hasattr(_bot_instance.llm_wrapper, 'llm_state'):
                    unfiltered_content = getattr(_bot_instance.llm_wrapper.llm_state, 'last_unfiltered_response', '')
                    filtered_content = getattr(_bot_instance.llm_wrapper.llm_state, 'last_filtered_response', '')
                    
                    # Show both versions if they differ (indicating filtering occurred)
                    if unfiltered_content and filtered_content and unfiltered_content != filtered_content:
                        current_msg["content"] = f"[UNFILTERED]: {unfiltered_content}\n[FILTERED]: {filtered_content}"
                        current_msg["was_filtered"] = True
                    elif unfiltered_content:
                        current_msg["content"] = unfiltered_content
                        current_msg["was_filtered"] = False
            
            # Get recent tool calls from chat handler if available
            if hasattr(_bot_instance, 'chat_handler') and _bot_instance.chat_handler:
                if hasattr(_bot_instance.chat_handler, 'tool_calls_log'):
                    tool_calls = _bot_instance.chat_handler.tool_calls_log[-10:]  # Last 10 tool calls
                    # Format for frontend display
                    tool_calls = [{
                        "tool": call.get("tool", "unknown"),
                        "status": "success" if call.get("success", False) else "failed",
                        "message": call.get("reason", ""),
                        "timestamp": call.get("timestamp", ""),
                        "args": call.get("args", {})
                    } for call in tool_calls]
            
            # Also get tool calls from API integration tracking
            if hasattr(_bot_instance, '_recent_tool_calls'):
                api_tool_calls = _bot_instance._recent_tool_calls[-5:]  # Last 5 API tool calls
                tool_calls.extend(api_tool_calls)
            
            return jsonify({
                "current_message": current_msg,
                "next_message": next_msg,
                "tool_calls": tool_calls,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting message status: {e}")
            return jsonify({
                "current_message": {"status": "error", "content": f"Error: {str(e)}"},
                "next_message": {"status": "idle", "content": ""},
                "tool_calls": [],
                "timestamp": datetime.now().isoformat()
            }), 500
    
    # Message control endpoints
    @app.route('/api/message/stop', methods=['POST'])
    @optional_api_key
    def stop_message():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Try to stop current TTS/speech if available
            if hasattr(_bot_instance, 'chat_handler') and _bot_instance.chat_handler:
                if hasattr(_bot_instance.chat_handler, 'stop_current_message'):
                    _bot_instance.chat_handler.stop_current_message()
                    return jsonify({"success": True, "message": "Message stopped"})
            
            # Try to stop LLM generation if available
            if hasattr(_bot_instance, 'llm_wrapper') and _bot_instance.llm_wrapper:
                if hasattr(_bot_instance.llm_wrapper, 'llm_state'):
                    _bot_instance.llm_wrapper.llm_state.next_cancelled = True
                    return jsonify({"success": True, "message": "LLM generation cancelled"})
            
            return jsonify({"success": True, "message": "Stop signal sent"})
        except Exception as e:
            logger.error(f"Error stopping message: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Discord control endpoints
    @app.route('/api/discord/send_message', methods=['POST'])
    @optional_api_key
    def send_message():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Use the bot's agentic response system to generate and send a message
            if hasattr(_bot_instance, '_get_agentic_response'):
                # Create a mock context for the control panel request
                if ToolContext is None:
                    return jsonify({"error": "ToolContext not available"}), 500
                
                mock_context = ToolContext(
                    user_id="control_panel_user",
                    username="Control Panel",
                    display_name="Control Panel",
                    channel_id="general",
                    channel_name="general",
                    guild_id="control_panel_guild"
                )
                
                # Track this as a tool call for display
                tool_call = {
                    "tool": "send_message",
                    "status": "processing",
                    "message": "Generating spontaneous message...",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to recent tool calls for display
                if not hasattr(_bot_instance, '_recent_tool_calls'):
                    _bot_instance._recent_tool_calls = []
                _bot_instance._recent_tool_calls.append(tool_call)
                
                # Generate a message using the LLM
                response = asyncio.run(_bot_instance._get_agentic_response(
                    mock_context,
                    "Generate a spontaneous friendly message for the general channel"
                ))
                
                # Update tool call status
                tool_call["status"] = "completed"
                tool_call["message"] = f"Generated: {response[:50]}..."
                
                # Get server ID from request if provided
                data = request.get_json() or {}
                target_server_id = data.get('server_id')
                
                # Find the general channel and send the message
                if hasattr(_bot_instance, 'guilds') and _bot_instance.guilds:
                    target_guild = None
                    
                    # If specific server requested, find it
                    if target_server_id:
                        for guild in _bot_instance.guilds:
                            if str(guild.id) == str(target_server_id):
                                target_guild = guild
                                break
                    else:
                        # Default to first guild
                        target_guild = _bot_instance.guilds[0] if _bot_instance.guilds else None
                    
                    if target_guild:
                        general_channel = None
                        for channel in target_guild.text_channels:
                            if channel.name == 'general':
                                general_channel = channel
                                break
                        
                        if general_channel:
                            asyncio.run(general_channel.send(response))
                            tool_call["message"] = f"Sent to #{general_channel.name}: {response[:30]}..."
                            return jsonify({
                                "success": True,
                                "message": "Message sent successfully",
                                "content": response,
                                "server_name": target_guild.name
                            })
                        else:
                            return jsonify({"error": f"No general channel found in {target_guild.name}"}), 404
                    else:
                        return jsonify({"error": "Target server not found"}), 404
                
                return jsonify({"error": "No guilds available"}), 404
            else:
                return jsonify({"error": "Bot does not support message generation"}), 501
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Server and bot management
    @app.route('/api/servers')
    @app.route('/servers')
    @optional_api_key
    def get_servers():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            servers = []
            if hasattr(_bot_instance, 'guilds') and _bot_instance.guilds:
                for guild in _bot_instance.guilds:
                    servers.append({
                        "id": str(guild.id),
                        "name": guild.name,
                        "member_count": guild.member_count,
                        "icon": str(guild.icon.url) if guild.icon else None
                    })
            return jsonify({"servers": servers})
        except Exception as e:
            logger.error(f"Error getting servers: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/servers/<server_id>/bots')
    @require_api_key
    def get_bots(server_id):
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # For now, just return the current bot
            bots = [{
                "id": str(_bot_instance.user.id),
                "name": _bot_instance.user.name,
                "avatar": str(_bot_instance.user.avatar.url) if _bot_instance.user.avatar else None
            }]
            return jsonify({"bots": bots})
        except Exception as e:
            logger.error(f"Error getting bots: {e}")
            return jsonify({"error": str(e)}), 500
    
    
    @app.route('/api/message/cancel', methods=['POST'])
    @require_api_key
    def cancel_message():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Cancel queued messages
            if hasattr(_bot_instance, 'chat_handler') and _bot_instance.chat_handler:
                _bot_instance.chat_handler.current_message = {
                    'status': 'idle',
                    'content': ''
                }
            
            return jsonify({"success": True, "message": "Message cancelled"})
        except Exception as e:
            logger.error(f"Error cancelling message: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Filter control endpoints
    @app.route('/api/filter/toggle', methods=['POST'])
    @require_api_key
    def toggle_filter():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Toggle filter system
            if hasattr(_bot_instance, 'tool_guard') and _bot_instance.tool_guard:
                # Toggle the filter enabled state
                current_state = getattr(_bot_instance.tool_guard, 'filter_enabled', True)
                _bot_instance.tool_guard.filter_enabled = not current_state
                
                new_state = "enabled" if _bot_instance.tool_guard.filter_enabled else "disabled"
                return jsonify({
                    "success": True, 
                    "message": f"Filter system {new_state}",
                    "filter_enabled": _bot_instance.tool_guard.filter_enabled
                })
            else:
                return jsonify({"error": "Filter system not available"}), 503
                
        except Exception as e:
            logger.error(f"Error toggling filter: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/filter/status', methods=['GET'])
    @require_api_key
    def get_filter_status():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if hasattr(_bot_instance, 'tool_guard') and _bot_instance.tool_guard:
                filter_enabled = getattr(_bot_instance.tool_guard, 'filter_enabled', True)
                return jsonify({
                    "filter_enabled": filter_enabled,
                    "status": "enabled" if filter_enabled else "disabled"
                })
            else:
                return jsonify({"error": "Filter system not available"}), 503
                
        except Exception as e:
            logger.error(f"Error getting filter status: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Memory management endpoints
    @app.route('/api/memory/clear_short_term', methods=['POST'])
    @optional_api_key
    def clear_short_term_memory():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if hasattr(_bot_instance, 'memory_system') and _bot_instance.memory_system:
                _bot_instance.memory_system.clear_short_term_memory()
                return jsonify({"success": True, "message": "Short-term memory cleared"})
            else:
                return jsonify({"error": "Memory system not available"}), 501
        except Exception as e:
            logger.error(f"Error clearing short-term memory: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/export')
    @optional_api_key
    def export_memory():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if hasattr(_bot_instance, 'memory_system') and _bot_instance.memory_system:
                memory_data = {
                    "short_term": _bot_instance.memory_system.short_term_memories,
                    "long_term": _bot_instance.memory_system.long_term_memories
                }
                return jsonify(memory_data)
            else:
                return jsonify({"error": "Memory system not available"}), 501
        except Exception as e:
            logger.error(f"Error exporting memory: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/import', methods=['POST'])
    @optional_api_key
    def import_memory():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            data = request.get_json()
            if hasattr(_bot_instance, 'memory_system') and _bot_instance.memory_system:
                if 'short_term' in data:
                    _bot_instance.memory_system.short_term_memories = data['short_term']
                if 'long_term' in data:
                    _bot_instance.memory_system.long_term_memories = data['long_term']
                return jsonify({"success": True, "message": "Memory imported successfully"})
            else:
                return jsonify({"error": "Memory system not available"}), 501
        except Exception as e:
            logger.error(f"Error importing memory: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memory/search')
    @optional_api_key
    def search_memory():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            query = request.args.get('query', '')
            if hasattr(_bot_instance, 'memory_system') and _bot_instance.memory_system:
                # Use get_relevant_memories to search for memories containing the query
                memories = _bot_instance.memory_system.get_relevant_memories(query)
                
                # Flatten the memories into a single list for the control panel
                all_memories = []
                if memories.get('short_term'):
                    for memory in memories['short_term']:
                        all_memories.append({
                            'content': memory,
                            'type': 'short_term',
                            'timestamp': datetime.now().isoformat()
                        })
                if memories.get('long_term'):
                    for memory in memories['long_term']:
                        all_memories.append({
                            'content': memory,
                            'type': 'long_term', 
                            'timestamp': datetime.now().isoformat()
                        })
                
                return jsonify({"memories": all_memories})
            else:
                return jsonify({"error": "Memory system not available"}), 501
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return jsonify({"error": str(e)}), 500

    # Prompt management endpoints
    @app.route('/api/prompt/get')
    @require_api_key
    def get_prompt():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if hasattr(_bot_instance, 'llm_wrapper') and _bot_instance.llm_wrapper:
                # Return the combined prompt from the new layered system
                combined_prompt = f"{_bot_instance.llm_wrapper.persona_core}\n\n{_bot_instance.llm_wrapper.output_contract}\n\n{_bot_instance.llm_wrapper.tools_rule}"
                return jsonify({"prompt": combined_prompt})
            else:
                return jsonify({"error": "LLM wrapper not available"}), 501
        except Exception as e:
            logger.error(f"Error getting prompt: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/prompt/set', methods=['POST'])
    @require_api_key
    def set_prompt():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            data = request.get_json()
            prompt = data.get('prompt', '')
            if hasattr(_bot_instance, 'llm_wrapper') and _bot_instance.llm_wrapper:
                # Update the persona_core with the new prompt (main personality)
                _bot_instance.llm_wrapper.persona_core = prompt
                return jsonify({"success": True, "message": "System prompt updated"})
            else:
                return jsonify({"error": "LLM wrapper not available"}), 501
        except Exception as e:
            logger.error(f"Error setting prompt: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/prompt/reset', methods=['POST'])
    @require_api_key
    def reset_prompt():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if hasattr(_bot_instance, 'llm_wrapper') and _bot_instance.llm_wrapper:
                # Reset to default persona_core
                default_prompt = (
                    "You are Sakura, a chaotic villain AI who hangs out in Discord voice. "
                    "Talk like a real person: casual, direct, bratty with Eric, warm with regulars, curt with strangers. "
                    "No Unicode emojis. Don't act like an assistant; don't offer help unless asked. "
                    "Keep it short for small talk; longer only when you genuinely care about the topic."
                )
                _bot_instance.llm_wrapper.persona_core = default_prompt
                return jsonify({"success": True, "message": "System prompt reset to default"})
            else:
                return jsonify({"error": "LLM wrapper not available"}), 501
        except Exception as e:
            logger.error(f"Error resetting prompt: {e}")
            return jsonify({"error": str(e)}), 500

    # Filter management endpoints
    @app.route('/api/filter/status')
    @require_api_key
    def filter_status():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Return filter status (you'll need to implement this in your bot)
            return jsonify({
                "enabled": True,
                "sensitivity": "medium",
                "action": "warn"
            })
        except Exception as e:
            logger.error(f"Error getting filter status: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/filter/words')
    @require_api_key
    def get_filter_words():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Return filtered words (you'll need to implement this in your bot)
            return jsonify({"words": []})
        except Exception as e:
            logger.error(f"Error getting filter words: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/filter/words', methods=['POST'])
    @require_api_key
    def add_filter_word():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            data = request.get_json()
            word = data.get('word', '')
            # Add word to filter (you'll need to implement this in your bot)
            return jsonify({"success": True, "message": "Filter word added"})
        except Exception as e:
            logger.error(f"Error adding filter word: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/filter/words/<word>', methods=['DELETE'])
    @require_api_key
    def remove_filter_word(word):
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Remove word from filter (you'll need to implement this in your bot)
            return jsonify({"success": True, "message": "Filter word removed"})
        except Exception as e:
            logger.error(f"Error removing filter word: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/filter/words/clear', methods=['DELETE'])
    @require_api_key
    def clear_filter_words():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Clear all filter words (you'll need to implement this in your bot)
            return jsonify({"success": True, "message": "All filter words cleared"})
        except Exception as e:
            logger.error(f"Error clearing filter words: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/filter/enable', methods=['POST'])
    @require_api_key
    def enable_filter():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Enable filter (you'll need to implement this in your bot)
            return jsonify({"success": True, "message": "Filter enabled"})
        except Exception as e:
            logger.error(f"Error enabling filter: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/filter/disable', methods=['POST'])
    @require_api_key
    def disable_filter():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Disable filter (you'll need to implement this in your bot)
            return jsonify({"success": True, "message": "Filter disabled"})
        except Exception as e:
            logger.error(f"Error disabling filter: {e}")
            return jsonify({"error": str(e)}), 500

    # Voice channel management endpoints
    @app.route('/api/servers/<server_id>/channels/voice')
    @optional_api_key
    def get_voice_channels(server_id):
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if hasattr(_bot_instance, 'guilds') and _bot_instance.guilds:
                target_guild = None
                for guild in _bot_instance.guilds:
                    if str(guild.id) == str(server_id):
                        target_guild = guild
                        break
                
                if target_guild:
                    voice_channels = []
                    for channel in target_guild.voice_channels:
                        voice_channels.append({
                            "id": str(channel.id),
                            "name": channel.name,
                            "type": "voice",
                            "user_limit": channel.user_limit,
                            "members": len(channel.members)
                        })
                    
                    return jsonify({
                        "channels": voice_channels,
                        "server_id": server_id,
                        "server_name": target_guild.name
                    })
                else:
                    return jsonify({"error": "Server not found"}), 404
            else:
                return jsonify({"error": "No guilds available"}), 503
        except Exception as e:
            logger.error(f"Error getting voice channels: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/voice/join', methods=['POST'])
    @optional_api_key
    def join_voice_channel():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            data = request.get_json() or {}
            channel_id = data.get('channelId')
            
            if not channel_id:
                return jsonify({"error": "Channel ID is required"}), 400
            
            # Find the voice channel
            target_channel = None
            logger.info(f"Looking for voice channel with ID: {channel_id}")
            
            if hasattr(_bot_instance, 'guilds') and _bot_instance.guilds:
                logger.info(f"Bot has {len(_bot_instance.guilds)} guilds")
                for guild in _bot_instance.guilds:
                    logger.info(f"Checking guild: {guild.name} (ID: {guild.id}) with {len(guild.voice_channels)} voice channels")
                    for channel in guild.voice_channels:
                        logger.info(f"  - Voice channel: {channel.name} (ID: {channel.id})")
                        if str(channel.id) == str(channel_id):
                            target_channel = channel
                            logger.info(f"Found target channel: {channel.name}")
                            break
                    if target_channel:
                        break
            else:
                logger.warning("Bot has no guilds or guilds attribute missing")
            
            if target_channel:
                # Check bot permissions first
                permissions = target_channel.permissions_for(target_channel.guild.me)
                if not permissions.connect:
                    logger.error(f"Bot lacks permission to connect to voice channel: {target_channel.name}")
                    return jsonify({"error": f"Bot lacks permission to join {target_channel.name}"}), 403
                
                if not permissions.speak:
                    logger.warning(f"Bot lacks permission to speak in voice channel: {target_channel.name}")
                
                # Use the bot's existing join command mechanism
                async def join_channel():
                    try:
                        logger.info(f"Calling bot's join command for channel: {target_channel.name}")
                        
                        # Create a context object that mimics a Discord command context
                        class ApiCommandContext:
                            def __init__(self, channel, guild, bot):
                                self.bot = bot
                                self.guild = guild
                                self.channel = channel  # This will be the voice channel
                                self.voice_client = guild.voice_client
                                # Create a mock author that's "in" the target voice channel
                                self.author = type('ApiUser', (), {
                                    'voice': type('VoiceState', (), {
                                        'channel': channel
                                    })()
                                })()
                                
                            async def send(self, message):
                                logger.info(f"Join command response: {message}")
                        
                        # Create the context
                        ctx = ApiCommandContext(target_channel, target_channel.guild, _bot_instance)
                        
                        # Call the bot's existing join command logic directly
                        if ctx.voice_client is not None:
                            logger.info("Bot already connected, moving to target channel")
                            await ctx.voice_client.move_to(target_channel)
                        else:
                            # This is the exact same logic as the working join command
                            logger.info(f"Connecting to {target_channel.name} using bot's join logic")
                            
                            # Access voice_recv through the bot instance (cached during startup)
                            if hasattr(_bot_instance, '_voice_recv_module'):
                                voice_recv = _bot_instance._voice_recv_module
                                logger.info("Using cached voice_recv module from bot instance")
                            else:
                                logger.error("voice_recv module not cached on bot instance")
                                voice_recv = None
                            
                            if voice_recv:
                                import logging
                                voice_recv_logger = logging.getLogger('discord.ext.voice_recv.router')
                                voice_recv_logger.setLevel(logging.CRITICAL)
                                await target_channel.connect(cls=voice_recv.VoiceRecvClient, timeout=20.0, reconnect=True)
                            else:
                                await target_channel.connect(timeout=20.0)
                        
                        # Set processing flag
                        guild_id = target_channel.guild.id
                        if hasattr(_bot_instance, 'processing_voice'):
                            _bot_instance.processing_voice[guild_id] = True
                            logger.info(f"Set processing_voice[{guild_id}] = True")
                        
                        # Update context with new voice client
                        ctx.voice_client = target_channel.guild.voice_client
                        
                        # Start voice processing using the bot's voice handler
                        if hasattr(_bot_instance, 'voice_handler') and _bot_instance.voice_handler:
                            try:
                                await _bot_instance.voice_handler.process_voice(ctx)
                                logger.info("Voice handler processing started successfully")
                            except Exception as e:
                                logger.error(f"Voice processing error: {e}")
                                # Don't fail the join if voice processing has issues
                        
                        return ctx.voice_client
                        
                    except Exception as e:
                        logger.error(f"Error in join_channel: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise e
                
                # Run in bot's existing event loop instead of creating new one
                if hasattr(_bot_instance, 'loop') and _bot_instance.loop:
                    # Use bot's event loop
                    future = asyncio.run_coroutine_threadsafe(join_channel(), _bot_instance.loop)
                    voice_client = future.result(timeout=45)  # Wait up to 45 seconds
                else:
                    # Fallback: try to get current loop
                    try:
                        loop = asyncio.get_event_loop()
                        voice_client = loop.run_until_complete(join_channel())
                    except RuntimeError:
                        # Last resort: create new loop (might still have issues)
                        voice_client = asyncio.run(join_channel())
                
                # Verify the bot actually joined
                is_connected = voice_client and voice_client.is_connected()
                logger.info(f"Voice join result - Channel: {target_channel.name}, Connected: {is_connected}")
                
                if not is_connected:
                    logger.error(f"Bot failed to connect to voice channel: {target_channel.name}")
                    return jsonify({
                        "success": False,
                        "error": f"Failed to connect to {target_channel.name} - connection verification failed"
                    }), 500
                
                return jsonify({
                    "success": True,
                    "message": f"Successfully joined {target_channel.name} with voice processing enabled",
                    "channel_name": target_channel.name,
                    "channel_id": channel_id,
                    "voice_processing": hasattr(_bot_instance, 'voice_handler') and _bot_instance.voice_handler is not None,
                    "connected": is_connected
                })
            else:
                return jsonify({"error": "Voice channel not found"}), 404
                
        except Exception as e:
            logger.error(f"Error joining voice channel: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/voice/leave', methods=['POST'])
    @optional_api_key
    def leave_voice_channel():
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            # Check if bot is in any voice channel and clean up properly
            left_channels = []
            
            if hasattr(_bot_instance, 'guilds') and _bot_instance.guilds:
                async def leave_channels():
                    for guild in _bot_instance.guilds:
                        if guild.voice_client and guild.voice_client.is_connected():
                            channel_name = guild.voice_client.channel.name
                            
                            # Stop voice processing for this guild
                            guild_id = guild.id
                            if hasattr(_bot_instance, 'processing_voice') and guild_id in _bot_instance.processing_voice:
                                _bot_instance.processing_voice[guild_id] = False
                            
                            # Disconnect from voice
                            await guild.voice_client.disconnect()
                            left_channels.append(channel_name)
                
                asyncio.run(leave_channels())
                
                if left_channels:
                    return jsonify({
                        "success": True,
                        "message": f"Successfully left voice channel(s): {', '.join(left_channels)}",
                        "channels_left": left_channels
                    })
                else:
                    return jsonify({"error": "Bot is not in any voice channel"}), 400
            else:
                return jsonify({"error": "No guilds available"}), 503
                
        except Exception as e:
            logger.error(f"Error leaving voice channel: {e}")
            return jsonify({"error": str(e)}), 500

    # VTube Studio endpoints (placeholder - you'll need to implement these)
    @app.route('/api/vtube/status')
    @require_api_key
    def vtube_status():
        return jsonify({"connected": False, "message": "VTube Studio integration not implemented"})

    @app.route('/api/vtube/connect', methods=['POST'])
    @require_api_key
    def vtube_connect():
        return jsonify({"success": False, "message": "VTube Studio integration not implemented"})

    @app.route('/api/vtube/disconnect', methods=['POST'])
    @require_api_key
    def vtube_disconnect():
        return jsonify({"success": False, "message": "VTube Studio integration not implemented"})

    @app.route('/api/vtube/expressions')
    @require_api_key
    def vtube_expressions():
        return jsonify({"expressions": []})

    @app.route('/api/vtube/hotkeys')
    @require_api_key
    def vtube_hotkeys():
        return jsonify({"hotkeys": []})

    @app.route('/api/vtube/settings')
    @require_api_key
    def vtube_settings():
        return jsonify({"ws_url": "ws://localhost:8001", "token": ""})

    @app.route('/api/vtube/settings', methods=['POST'])
    @require_api_key
    def vtube_save_settings():
        return jsonify({"success": True, "message": "Settings saved (not implemented)"})

    # Memory management endpoints
    @app.route('/api/memories')
    @optional_api_key
    def get_memories():
        """Get advanced memories data"""
        try:
            import json
            import os
            
            memories_file = os.path.join(os.getcwd(), 'data', 'advanced_memories.json')
            if os.path.exists(memories_file):
                with open(memories_file, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                return jsonify({
                    "success": True,
                    "memories": memories_data,
                    "file_path": memories_file,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Advanced memories file not found",
                    "expected_path": memories_file
                }), 404
                
        except Exception as e:
            logger.error(f"Error reading memories: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memories/stats')
    @optional_api_key
    def get_memory_stats():
        """Get memory statistics"""
        try:
            import json
            import os
            
            memories_file = os.path.join(os.getcwd(), 'data', 'advanced_memories.json')
            if os.path.exists(memories_file):
                with open(memories_file, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                short_term = memories_data.get('short_term', [])
                long_term = memories_data.get('long_term', [])
                
                # Calculate stats
                total_memories = len(short_term) + len(long_term)
                people_involved = set()
                memory_types = {}
                
                for memory in short_term + long_term:
                    people_involved.update(memory.get('people_involved', []))
                    mem_type = memory.get('memory_type', 'unknown')
                    memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                
                return jsonify({
                    "success": True,
                    "stats": {
                        "total_memories": total_memories,
                        "short_term_count": len(short_term),
                        "long_term_count": len(long_term),
                        "unique_people": len(people_involved),
                        "people_list": list(people_involved),
                        "memory_types": memory_types
                    },
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({"error": "Advanced memories file not found"}), 404
                
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/memories/clear', methods=['POST'])
    @optional_api_key
    def clear_memories():
        """Clear specific memory types"""
        try:
            import json
            import os
            
            data = request.get_json() or {}
            memory_type = data.get('type', 'all')  # 'short_term', 'long_term', or 'all'
            
            memories_file = os.path.join(os.getcwd(), 'data', 'advanced_memories.json')
            if os.path.exists(memories_file):
                with open(memories_file, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                original_short = len(memories_data.get('short_term', []))
                original_long = len(memories_data.get('long_term', []))
                
                if memory_type == 'short_term':
                    memories_data['short_term'] = []
                elif memory_type == 'long_term':
                    memories_data['long_term'] = []
                elif memory_type == 'all':
                    memories_data['short_term'] = []
                    memories_data['long_term'] = []
                else:
                    return jsonify({"error": "Invalid memory type. Use 'short_term', 'long_term', or 'all'"}), 400
                
                # Save the updated memories
                with open(memories_file, 'w', encoding='utf-8') as f:
                    json.dump(memories_data, f, indent=2, ensure_ascii=False)
                
                return jsonify({
                    "success": True,
                    "message": f"Cleared {memory_type} memories",
                    "cleared_counts": {
                        "short_term": original_short if memory_type in ['short_term', 'all'] else 0,
                        "long_term": original_long if memory_type in ['long_term', 'all'] else 0
                    },
                    "timestamp": datetime.now().isoformat()
                })
            else:
                return jsonify({"error": "Advanced memories file not found"}), 404
                
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/vision/status')
    @optional_api_key
    def get_vision_status():
        """Get vision tool status"""
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if not hasattr(_bot_instance, 'vision_tool'):
                return jsonify({"error": "Vision tool not available"}), 503
            
            status = _bot_instance.vision_tool.get_status()
            return jsonify({
                "success": True,
                "status": status
            })
        except Exception as e:
            logger.error(f"Error getting vision status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/vision/monitors')
    @optional_api_key
    def get_vision_monitors():
        """Get available monitors for vision capture"""
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if not hasattr(_bot_instance, 'vision_tool'):
                return jsonify({"error": "Vision tool not available"}), 503
            
            monitors = _bot_instance.vision_tool.get_available_monitors()
            logger.info(f"Returning {len(monitors)} monitors")
            return jsonify({
                "success": True,
                "monitors": monitors
            })
        except Exception as e:
            logger.error(f"Error getting monitors: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/live/status')
    @optional_api_key
    def get_live_vision_status():
        """Get live vision status"""
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if not hasattr(_bot_instance, 'live_vision_controller'):
                return jsonify({"error": "Live vision not available"}), 503
            
            lvc = _bot_instance.live_vision_controller
            status = {
                "enabled": lvc.live_enabled,
                "interval_ms": getattr(lvc, 'min_interval_ms', 500),
                "fresh_secs": getattr(lvc, 'fresh_secs', 5),
                "is_fresh": lvc.is_fresh,
                "age_secs": getattr(lvc, 'age_secs', 0),
                "last_capture_time": lvc.last_capture_time if hasattr(lvc, 'last_capture_time') else None,
                "summary_count": getattr(lvc, 'summary_count', 0)
            }
            return jsonify({
                "success": True,
                "status": status
            })
        except Exception as e:
            logger.error(f"Error getting live vision status: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/singing/songs')
    @optional_api_key
    def get_singing_songs():
        """Get available songs for singing"""
        if not _bot_instance:
            return jsonify({"error": "Bot not connected"}), 503
            
        try:
            if not hasattr(_bot_instance, 'singing_handler'):
                return jsonify({"error": "Singing handler not available"}), 503
            
            import os
            songs_dir = "songs"
            if not os.path.exists(songs_dir):
                return jsonify({"success": True, "songs": []})
            
            songs = []
            for file in os.listdir(songs_dir):
                if file.endswith('.wav'):
                    songs.append({
                        "name": file.replace('.wav', ''),
                        "file": file
                    })
            
            return jsonify({
                "success": True,
                "songs": songs
            })
        except Exception as e:
            logger.error(f"Error getting songs: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500
    
    return app, set_bot_instance

def start_api_server(_bot_instance, port=5000, host='0.0.0.0'):
    """Start the API server in a separate thread"""
    # Force recreation to ensure new routes are registered
    force_recreate_app()
    app, set_bot_instance = create_api_server()
    set_bot_instance(_bot_instance)
    
    def run_server():
        try:
            logger.info(f"Starting API server on {host}:{port}")
            app.run(host=host, port=port, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"API server error: {e}")
    
    api_thread = threading.Thread(target=run_server, daemon=True)
    api_thread.start()
    logger.info("API server thread started")
    
    return api_thread

# Example integration in your main.py:
"""
# Add this to your main.py after bot initialization:

from sakura_frontend.api_integration import start_api_server

# After your bot is initialized and ready:
start_api_server(bot, port=5000, host='0.0.0.0')

# Set environment variable for API key:
# export API_KEY="your_very_secure_api_key_here"
"""
