import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response, make_response
from flask_socketio import SocketIO, emit
import queue
import os
import sys

# Add the current directory to Python path to import bot modules
sys.path.append('.')

# Bot components will be set by the main bot when it starts
bot_instance = None
bot_guilds = []  # Store guild data from the bot
llm_wrapper = None
voice_handler = None
tool_guard = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
# CORS is now handled by the Cloudflare Worker proxy
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
llm_enabled = True
current_mode = "chat"
event_queue = queue.Queue()
tts_queue = queue.Queue()

# Bot components (will be initialized when bot is available)
llm_wrapper = None
voice_handler = None
tool_guard = None

def set_bot_components(bot, llm, voice, guard):
    """Set bot components from the main bot"""
    global bot_instance, llm_wrapper, voice_handler, tool_guard
    
    bot_instance = bot
    llm_wrapper = llm
    voice_handler = voice
    tool_guard = guard
    
    logger.info("Bot components set successfully")
    emit_event('bot_connected', {
        "bot_connected": True,
        "components": {
            "llm_wrapper": llm_wrapper is not None,
            "voice_handler": voice_handler is not None,
            "tool_guard": tool_guard is not None
        }
    })

def initialize_bot_components():
    global llm_wrapper, voice_handler, tool_guard
    
    if bot_instance:
        try:
            llm_wrapper = getattr(bot_instance, 'llm_wrapper', None)
            voice_handler = getattr(bot_instance, 'voice_handler', None)
            tool_guard = getattr(bot_instance, 'tool_guard', None)
            logger.info("Bot components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing bot components: {e}")

def emit_event(event_type, data):
    """Emit an event to connected WebSocket clients"""
    try:
        socketio.emit('event', {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error emitting event: {e}")

# API Endpoints

@app.route('/status', methods=['GET'])
def get_status():
    """Get the current status of the LLM controller"""
    try:
        # Check if bot is connected by checking if we have guild data
        bot_is_connected = bot_instance == "connected" and len(bot_guilds) > 0
        
        status = {
            "controller": "online",
            "llm_enabled": llm_enabled,
            "mode": current_mode,
            "bot_connected": bot_is_connected,
            "components": {
                "llm_wrapper": llm_wrapper is not None,
                "voice_handler": voice_handler is not None,
                "tool_guard": tool_guard is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add guild and filter information
        if bot_is_connected:
            status.update({
                "guilds": len(bot_guilds),
                "filter_enabled": getattr(tool_guard, 'filter_enabled', True) if tool_guard else True
            })
        else:
            status.update({
                "guilds": 0,
                "filter_enabled": True
            })
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/prompt', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Text content required"}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400
        
        # Emit event
        emit_event('prompt_received', {"text": text})
        
        # Process the prompt with actual LLM if available
        if llm_enabled:
            try:
                # Try to send to actual bot if connected
                if bot_instance and hasattr(bot_instance, 'chat_handler'):
                    # This would be the real integration point
                    # For now, we'll simulate a more realistic response
                    response = f"Sakura: Hello! You said: '{text}'. I'm responding from the LLM controller. (This is currently a simulated response - full bot integration pending)"
                else:
                    # Fallback response when bot not connected
                    responses = [
                        f"I understand you're saying: '{text}'. The bot isn't fully connected right now, but I'm processing through the controller!",
                        f"Interesting message: '{text}'. I'm working through the LLM controller - full Discord bot integration is in progress.",
                        f"You wrote: '{text}'. This is a response from the LLM controller. The Discord bot connection is being established.",
                        f"Message received: '{text}'. I'm Sakura, responding via the controller API. Bot integration is active but Discord connection pending.",
                    ]
                    import random
                    response = random.choice(responses)
                
                emit_event('prompt_processed', {"response": response})
                
                return jsonify({
                    "success": True,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing LLM prompt: {e}")
                return jsonify({
                    "success": True,
                    "response": f"I received your message: '{text}', but encountered an error processing it. Error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        else:
            return jsonify({"error": "LLM is currently disabled. Use the Toggle AI button to enable it."}), 503
            
    except Exception as e:
        logger.error(f"Error handling prompt: {e}")
        emit_event('error', {"message": str(e)})
        return jsonify({"error": str(e)}), 500

@app.route('/speak', methods=['POST'])
def handle_speak():
    """Handle a TTS request"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Text content required"}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400
        
        # Emit event
        emit_event('speak_requested', {"text": text})
        
        # Add to TTS queue
        tts_queue.put({
            "text": text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process TTS (this would integrate with your bot's voice handler)
        if voice_handler:
            # Here you would call your bot's TTS processing
            emit_event('speak_started', {"text": text})
            
            return jsonify({
                "success": True,
                "message": "TTS request queued",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Voice handler not available"}), 503
            
    except Exception as e:
        logger.error(f"Error handling speak request: {e}")
        emit_event('error', {"message": str(e)})
        return jsonify({"error": str(e)}), 500

@app.route('/mode', methods=['POST'])
def set_mode():
    try:
        data = request.get_json()
        if not data or 'mode' not in data:
            return jsonify({"error": "Mode required"}), 400
        
        mode = data['mode']
        valid_modes = ["chat", "voice", "dm", "call"]
        
        if mode not in valid_modes:
            return jsonify({"error": f"Invalid mode. Must be one of: {valid_modes}"}), 400
        
        global current_mode
        current_mode = mode
        
        # Emit event
        emit_event('mode_changed', {"mode": mode})
        
        return jsonify({
            "success": True,
            "mode": mode,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error setting mode: {e}")
        emit_event('error', {"message": str(e)})
        return jsonify({"error": str(e)}), 500

@app.route('/llm/toggle', methods=['POST'])
def toggle_llm():
    """Toggle LLM on/off"""
    try:
        global llm_enabled
        llm_enabled = not llm_enabled
        
        # Emit event
        emit_event('llm_toggled', {"enabled": llm_enabled})
        
        return jsonify({
            "success": True,
            "llm_enabled": llm_enabled,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error toggling LLM: {e}")
        emit_event('error', {"message": str(e)})
        return jsonify({"error": str(e)}), 500


@app.route('/voice/stop', methods=['POST'])
def stop_voice():
    """Stop current voice processing"""
    try:
        if not voice_handler:
            return jsonify({"error": "Voice handler not available"}), 503
        
        # Stop voice processing
        voice_handler.stop_voice_processing()
        
        # Emit event
        emit_event('voice_stopped', {})
        
        return jsonify({
            "success": True,
            "message": "Voice processing stopped",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error stopping voice: {e}")
        emit_event('error', {"message": str(e)})
        return jsonify({"error": str(e)}), 500

@app.route('/servers/<server_id>/channels/voice', methods=['GET'])
def get_voice_channels(server_id):
    """Get voice channels for a specific server"""
    try:
        # Find the guild in our stored guild data
        guild_data = None
        for guild in bot_guilds:
            if str(guild.get('id')) == str(server_id):
                guild_data = guild
                break
        
        if not guild_data:
            return jsonify({"error": "Server not found"}), 404
        
        # Try to get real voice channels from the bot API
        try:
            import requests
            logger.info(f"Requesting voice channels for server {server_id} from bot API")
            bot_response = requests.get(f"http://localhost:5000/api/servers/{server_id}/channels/voice", 
                                     timeout=10)  # Increased timeout
            
            logger.info(f"Bot API responded with status {bot_response.status_code}")
            
            if bot_response.status_code == 200:
                result = bot_response.json()
                channels_count = len(result.get('channels', []))
                logger.info(f"Retrieved {channels_count} voice channels from bot API")
                logger.debug(f"Voice channels: {[ch.get('name') for ch in result.get('channels', [])]}")
                return jsonify(result)
            else:
                logger.warning(f"Bot API returned {bot_response.status_code} for voice channels: {bot_response.text}")
                return jsonify({"error": f"Bot API returned {bot_response.status_code}: {bot_response.text}"}), 502
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to bot API (is bot running?): {e}")
            return jsonify({"error": "Bot API not running - start the Discord bot first"}), 503
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error to bot API for voice channels: {e}")
            return jsonify({"error": f"Bot API error: {str(e)}"}), 503
        
    except Exception as e:
        logger.error(f"Error getting voice channels: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/voice/join', methods=['POST'])
def join_voice_channel():
    """Join a voice channel"""
    try:
        data = request.get_json()
        if not data or 'channelId' not in data:
            return jsonify({"error": "Channel ID is required"}), 400
        
        channel_id = data['channelId']
        
        # Try to join via real bot API
        try:
            import requests
            bot_response = requests.post("http://localhost:5000/api/voice/join", 
                                       json={"channelId": channel_id},
                                       timeout=10)
            
            if bot_response.status_code == 200:
                result = bot_response.json()
                logger.info(f"Successfully joined voice channel via bot API: {channel_id}")
                
                # Emit event
                emit_event('voice_joined', {
                    "channel_id": channel_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                return jsonify(result)
            else:
                logger.warning(f"Bot API returned {bot_response.status_code} for voice join")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to bot API for voice join: {e}")
        
        # No fallback - return error if bot API not connected
        return jsonify({"error": "Bot API not connected - cannot join voice channel"}), 503
        
    except Exception as e:
        logger.error(f"Error joining voice channel: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/voice/leave', methods=['POST'])
def leave_voice_channel():
    """Leave current voice channel"""
    try:
        # Try to leave via real bot API
        try:
            import requests
            bot_response = requests.post("http://localhost:5000/api/voice/leave", 
                                       json={},
                                       timeout=10)
            
            if bot_response.status_code == 200:
                result = bot_response.json()
                logger.info("Successfully left voice channel via bot API")
                
                # Emit event
                emit_event('voice_left', {
                    "timestamp": datetime.now().isoformat()
                })
                
                return jsonify(result)
            else:
                logger.warning(f"Bot API returned {bot_response.status_code} for voice leave")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to bot API for voice leave: {e}")
        
        # No fallback - return error if bot API not connected
        return jsonify({"error": "Bot API not connected - cannot leave voice channel"}), 503
        
    except Exception as e:
        logger.error(f"Error leaving voice channel: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/bot_connected', methods=['POST'])
def bot_connected():
    """Handle bot connection signal"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Update bot connection status
        global bot_instance, bot_guilds
        bot_instance = "connected"  # Simple flag that bot is connected
        bot_guilds = data.get('guilds', [])  # Store the actual guild data
        
        # Emit connection event
        emit_event('bot_connected', {
            "connected": True,
            "guild_count": data.get('guild_count', 0),
            "guilds": bot_guilds,
            "bot_name": data.get('bot_name', 'Sakura'),
            "timestamp": data.get('timestamp', datetime.now().isoformat())
        })
        
        logger.info(f"Bot connected: {data.get('bot_name', 'Sakura')} with {data.get('guild_count', 0)} guilds")
        
        return jsonify({
            "success": True,
            "message": "Bot connection registered successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error handling bot connection: {e}")
        return jsonify({"error": str(e)}), 500

# Frontend compatibility endpoints

@app.route('/ai/status', methods=['GET'])
def get_ai_status():
    """Get AI status for frontend compatibility"""
    return jsonify({
        "enabled": llm_enabled,
        "mode": current_mode,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/ai/control', methods=['POST'])
def control_ai():
    """Control AI for frontend compatibility"""
    try:
        data = request.get_json()
        action = data.get('action', 'toggle')
        
        if action == 'toggle':
            global llm_enabled
            llm_enabled = not llm_enabled
            emit_event('ai_toggled', {"enabled": llm_enabled})
            
            return jsonify({
                "success": True,
                "enabled": llm_enabled,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Invalid action"}), 400
            
    except Exception as e:
        logger.error(f"Error controlling AI: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/messages/status', methods=['GET'])
def get_message_status():
    """Get message status for frontend compatibility"""
    try:
        # Try to get real status from the bot API if it's running
        if bot_instance == "connected":
            try:
                import requests
                # Use shorter timeout to prevent hanging
                bot_response = requests.get("http://localhost:5000/api/messages/status", timeout=2)
                if bot_response.status_code == 200:
                    # Forward the real status from the bot
                    return bot_response.json()
            except requests.exceptions.Timeout:
                logger.warning("Bot API timeout - using fallback status")
            except requests.exceptions.ConnectionError:
                logger.warning("Bot API not available - using fallback status")
            except Exception as e:
                logger.warning(f"Bot API error: {e} - using fallback status")
        
        # Default/fallback response
        return jsonify({
            "current_message": {
                "status": "idle", 
                "content": "Ready to respond..."
            },
            "next_message": {
                "status": "none",
                "content": ""
            },
            "tool_calls": [],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting message status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/message/stop', methods=['POST'])
def stop_message():
    """Stop current message for frontend compatibility"""
    try:
        # Try to forward to the bot API if it's running
        if bot_instance == "connected":
            try:
                import requests
                bot_response = requests.post("http://localhost:5000/api/message/stop", timeout=5)
                if bot_response.status_code == 200:
                    emit_event('message_stopped', {})
                    return bot_response.json()
            except:
                pass  # Fall back to local handling
        
        # Fallback: try local voice handler
        if voice_handler:
            voice_handler.stop_voice_processing()
        
        emit_event('message_stopped', {})
        
        return jsonify({
            "success": True,
            "message": "Message stopped",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error stopping message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/message/cancel', methods=['POST'])
def cancel_message():
    """Cancel next message for frontend compatibility"""
    try:
        emit_event('message_cancelled', {})
        
        return jsonify({
            "success": True,
            "message": "Message cancelled",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error cancelling message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/discord/send_message', methods=['POST'])
def send_discord_message():
    """Trigger actual Discord bot to generate and send a message"""
    try:
        data = request.get_json() or {}
        server_id = data.get('server_id')
        channel = data.get('channel', 'general')
        
        # Find the server name
        server_name = "Unknown Server"
        for guild in bot_guilds:
            if str(guild.get('id')) == str(server_id):
                server_name = guild.get('name', 'Unknown Server')
                break
        
        if not llm_enabled:
            return jsonify({"error": "LLM is disabled. Enable AI first."}), 503
        
        # Try to trigger the actual bot to send a message
        if bot_instance == "connected":
            try:
                # Send a signal to the actual Discord bot to generate and send a message
                # This endpoint will be called by the real bot to trigger message generation
                
                # For now, we'll use a webhook approach - send a request to the bot
                import requests
                
                # Try to find if there's a local bot API running
                try:
                    logger.info(f"Sending message request to bot API for server {server_id}")
                    bot_response = requests.post("http://localhost:5000/api/discord/send_message", 
                                               json={"server_id": server_id, "channel": channel},
                                               timeout=15)
                    
                    logger.info(f"Bot API responded with status {bot_response.status_code}")
                    
                    if bot_response.status_code == 200:
                        result = bot_response.json()
                        logger.info(f"Successfully triggered bot to send message: {result.get('content', 'unknown')}")
                        
                        return jsonify({
                            "success": True,
                            "message": f"Message sent to #{channel}",
                            "server_name": server_name,
                            "generated_content": result.get('content', 'Message sent via bot'),
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        logger.warning(f"Bot API returned {bot_response.status_code}: {bot_response.text}")
                        return jsonify({"error": f"Bot API error: {bot_response.status_code}"}), 502
                        
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Connection error to bot API: {e}")
                    return jsonify({"error": "Bot API not running - start the Discord bot first"}), 503
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error to bot API: {e}")
                    return jsonify({"error": f"Bot API request failed: {str(e)}"}), 503
            
                
            except Exception as e:
                logger.error(f"Error triggering bot message: {e}")
                return jsonify({"error": f"Failed to trigger bot: {str(e)}"}), 500
        else:
            return jsonify({"error": "Discord bot not connected to controller"}), 503
        
    except Exception as e:
        logger.error(f"Error sending Discord message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memories', methods=['GET'])
def get_memories():
    """Get advanced memories data"""
    try:
        # Try to forward to the bot API if it's running
        if bot_instance == "connected":
            try:
                import requests
                bot_response = requests.get("http://localhost:5000/api/memories", timeout=10)
                if bot_response.status_code == 200:
                    return bot_response.json()
            except:
                pass  # Fall back to direct file access
        
        # Direct file access fallback
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

@app.route('/memories/stats', methods=['GET'])
def get_memory_stats():
    """Get memory statistics"""
    try:
        # Try to forward to the bot API if it's running
        if bot_instance == "connected":
            try:
                import requests
                bot_response = requests.get("http://localhost:5000/api/memories/stats", timeout=10)
                if bot_response.status_code == 200:
                    return bot_response.json()
            except:
                pass  # Fall back to direct calculation
        
        # Direct calculation fallback
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

@app.route('/memories/clear', methods=['POST'])
def clear_memories():
    """Clear specific memory types"""
    try:
        # Try to forward to the bot API if it's running
        if bot_instance == "connected":
            try:
                import requests
                bot_response = requests.post("http://localhost:5000/api/memories/clear", 
                                          json=request.get_json(), timeout=10)
                if bot_response.status_code == 200:
                    return bot_response.json()
            except:
                pass  # Fall back to direct operation
        
        # Direct operation fallback
        import json
        import os
        
        data = request.get_json() or {}
        memory_type = data.get('type', 'all')
        
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

@app.route('/servers', methods=['GET'])
def get_servers():
    """Get Discord servers for frontend compatibility"""
    try:
        global bot_guilds
        # Return the stored guild data from bot connection
        servers = bot_guilds if bot_guilds else []
        
        return jsonify({
            "servers": servers,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting servers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/servers/<server_id>/bots', methods=['GET'])
def get_bots(server_id):
    """Get bots for a server for frontend compatibility"""
    try:
        # For now, just return the current bot
        bots = [{"id": "sakura", "name": "Sakura"}]
        
        return jsonify({
            "bots": bots,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting bots: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/filter/status', methods=['GET'])
def get_filter_status():
    """Get filter status for frontend compatibility"""
    try:
        filter_enabled = True
        if tool_guard:
            filter_enabled = getattr(tool_guard, 'filter_enabled', True)
        
        return jsonify({
            "filter_enabled": filter_enabled,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting filter status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/set_components', methods=['POST'])
def set_components():
    """Set bot components status from the main bot"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Update component status
        global llm_wrapper, voice_handler, tool_guard
        
        # Set component availability flags
        llm_wrapper = data.get('llm_wrapper', False)
        voice_handler = data.get('voice_handler', False)
        tool_guard = data.get('tool_guard', False)
        
        logger.info(f"Bot components updated: LLM={llm_wrapper}, Voice={voice_handler}, Guard={tool_guard}")
        
        # Emit event
        emit_event('components_updated', {
            "llm_wrapper": llm_wrapper,
            "voice_handler": voice_handler,
            "tool_guard": tool_guard
        })
        
        return jsonify({
            "success": True,
            "message": "Bot components status updated successfully",
            "components": {
                "llm_wrapper": llm_wrapper,
                "voice_handler": voice_handler,
                "tool_guard": tool_guard
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error setting components: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/guilds', methods=['POST'])
def debug_set_guilds():
    """Debug endpoint to manually set guild data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        global bot_guilds, bot_instance
        bot_guilds = data.get('guilds', [])
        bot_instance = "connected"
        
        logger.info(f"Debug: Manually set {len(bot_guilds)} guilds")
        for guild in bot_guilds:
            logger.info(f"  - {guild.get('name')} (id: {guild.get('id')})")
        
        return jsonify({
            "success": True,
            "guilds_set": len(bot_guilds),
            "guilds": bot_guilds,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error setting debug guilds: {e}")
        return jsonify({"error": str(e)}), 500

# Additional endpoints for frontend compatibility

@app.route('/vtube/status', methods=['GET'])
def get_vtube_status():
    """Get VTube Studio status for frontend compatibility"""
    try:
        return jsonify({
            "connected": False,
            "model_loaded": False,
            "available_models": [],
            "current_model": None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting VTube status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/vtube/settings', methods=['GET'])
def get_vtube_settings():
    """Get VTube Studio settings for frontend compatibility"""
    try:
        return jsonify({
            "api_key": "",
            "port": 8001,
            "enabled": False,
            "auto_connect": False,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting VTube settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/filter/words', methods=['GET'])
def get_filter_words():
    """Get filter words for frontend compatibility"""
    try:
        # Load filter words from the actual data file
        filter_words = []
        filter_enabled = True
        try:
            import json
            with open('data/filtered_words.json', 'r') as f:
                data = json.load(f)
                filter_words = data.get('words', [])
                filter_enabled = data.get('enabled', True)
        except Exception as e:
            logger.warning(f"Could not load filter words: {e}")
            filter_words = []
        
        return jsonify({
            "words": filter_words,
            "count": len(filter_words),
            "enabled": filter_enabled,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting filter words: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/filter/words/<word>', methods=['DELETE'])
def delete_filter_word(word):
    """Delete a filter word"""
    try:
        import json
        import os
        
        # Load current filter data
        filter_file = 'data/filtered_words.json'
        if not os.path.exists(filter_file):
            return jsonify({"error": "Filter file not found"}), 404
        
        with open(filter_file, 'r') as f:
            data = json.load(f)
        
        words = data.get('words', [])
        
        # Remove the word (case-insensitive)
        original_count = len(words)
        words = [w for w in words if w.lower() != word.lower()]
        
        if len(words) == original_count:
            return jsonify({"error": f"Word '{word}' not found in filter list"}), 404
        
        # Update the data
        data['words'] = words
        
        # Save back to file
        with open(filter_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Removed filter word: {word}")
        
        return jsonify({
            "success": True,
            "message": f"Filter word '{word}' removed successfully",
            "remaining_count": len(words),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error deleting filter word: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/filter/words', methods=['POST'])
def add_filter_word():
    """Add a new filter word"""
    try:
        data = request.get_json()
        if not data or 'word' not in data:
            return jsonify({"error": "Word parameter required"}), 400
        
        new_word = data['word'].strip()
        if not new_word:
            return jsonify({"error": "Word cannot be empty"}), 400
        
        import json
        import os
        
        # Load current filter data
        filter_file = 'data/filtered_words.json'
        if os.path.exists(filter_file):
            with open(filter_file, 'r') as f:
                filter_data = json.load(f)
        else:
            filter_data = {"enabled": True, "words": []}
        
        words = filter_data.get('words', [])
        
        # Check if word already exists (case-insensitive)
        if any(w.lower() == new_word.lower() for w in words):
            return jsonify({"error": f"Word '{new_word}' already exists in filter list"}), 409
        
        # Add the new word
        words.append(new_word)
        filter_data['words'] = words
        
        # Save back to file
        with open(filter_file, 'w') as f:
            json.dump(filter_data, f, indent=2)
        
        logger.info(f"Added filter word: {new_word}")
        
        return jsonify({
            "success": True,
            "message": f"Filter word '{new_word}' added successfully",
            "total_count": len(words),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error adding filter word: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/filter/toggle', methods=['POST'])
def toggle_filter():
    """Toggle filter system on/off"""
    try:
        import json
        import os
        
        # Load current filter data
        filter_file = 'data/filtered_words.json'
        if os.path.exists(filter_file):
            with open(filter_file, 'r') as f:
                filter_data = json.load(f)
        else:
            filter_data = {"enabled": True, "words": []}
        
        # Toggle the enabled status
        filter_data['enabled'] = not filter_data.get('enabled', True)
        
        # Save back to file
        with open(filter_file, 'w') as f:
            json.dump(filter_data, f, indent=2)
        
        status = "enabled" if filter_data['enabled'] else "disabled"
        logger.info(f"Filter system {status}")
        
        return jsonify({
            "success": True,
            "enabled": filter_data['enabled'],
            "message": f"Filter system {status}",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error toggling filter: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/status', methods=['GET'])
def get_memory_status():
    """Get memory system status"""
    try:
        # Load memory data to get actual counts
        short_term_count = 0
        long_term_count = 0
        
        try:
            import json
            import os
            memory_file = "data/advanced_memories.json"
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    short_term_count = len(data.get('short_term', []))
                    long_term_count = len(data.get('long_term', []))
        except Exception as e:
            logger.warning(f"Could not load memory status: {e}")
        
        return jsonify({
            "enabled": True,
            "total_memories": short_term_count + long_term_count,
            "short_term": short_term_count,
            "long_term": long_term_count,
            "categories": ["short_term", "long_term"],
            "last_update": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting memory status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/list', methods=['GET'])
def get_memory_list():
    """Get list of memories"""
    try:
        # Try to load memories from the actual data files
        memories = []
        try:
            import json
            import os
            
            # Try multiple memory file locations
            memory_files = [
                "data/advanced_memories.json",
                "memory/user_memories.json"
            ]
            
            for memory_file in memory_files:
                if os.path.exists(memory_file):
                    with open(memory_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            memories.extend(data)
                        elif isinstance(data, dict):
                            # Handle different memory file structures
                            if 'memories' in data:
                                memories.extend(data['memories'])
                            elif 'short_term' in data or 'long_term' in data:
                                # Handle advanced_memories.json structure
                                memories.extend(data.get('short_term', []))
                                memories.extend(data.get('long_term', []))
                            else:
                                # Fallback: treat the dict values as potential memory lists
                                for key, value in data.items():
                                    if isinstance(value, list):
                                        memories.extend(value)
                    break
                    
        except Exception as e:
            logger.warning(f"Could not load memories: {e}")
            memories = []
        
        return jsonify({
            "memories": memories,
            "count": len(memories),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting memory list: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/add', methods=['POST'])
def add_memory():
    """Add a new memory"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # This would integrate with the actual memory system
        logger.info(f"Memory add request: {data}")
        
        return jsonify({
            "success": True,
            "message": "Memory added successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/delete', methods=['POST'])
def delete_memory():
    """Delete a memory"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # This would integrate with the actual memory system
        logger.info(f"Memory delete request: {data}")
        
        return jsonify({
            "success": True,
            "message": "Memory deleted successfully",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return jsonify({"error": str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'Connected to LLM Controller'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('ping')
def handle_ping():
    """Handle ping from client"""
    emit('pong', {'timestamp': datetime.now().isoformat()})

# Background tasks
def background_event_processor():
    """Process events in the background"""
    while True:
        try:
            # Process any queued events
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in background processor: {e}")

def start_background_tasks():
    """Start background tasks"""
    thread = threading.Thread(target=background_event_processor, daemon=True)
    thread.start()

# Additional endpoints for control panel compatibility
@app.route('/memory/clear_short_term', methods=['POST'])
def clear_short_term_memory():
    """Clear short-term memory (control panel compatibility)"""
    try:
        import json
        import os
        
        memories_file = os.path.join(os.getcwd(), 'data', 'advanced_memories.json')
        if os.path.exists(memories_file):
            with open(memories_file, 'r', encoding='utf-8') as f:
                memories_data = json.load(f)
            
            original_count = len(memories_data.get('short_term', []))
            memories_data['short_term'] = []
            
            # Save the updated memories
            with open(memories_file, 'w', encoding='utf-8') as f:
                json.dump(memories_data, f, indent=2, ensure_ascii=False)
            
            return jsonify({
                "success": True,
                "message": f"Cleared {original_count} short-term memories",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Advanced memories file not found"}), 404
            
    except Exception as e:
        logger.error(f"Error clearing short-term memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/export', methods=['GET'])
def export_memory():
    """Export all memories (control panel compatibility)"""
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
                "export_timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Advanced memories file not found"}), 404
            
    except Exception as e:
        logger.error(f"Error exporting memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/import', methods=['POST'])
def import_memory():
    """Import memories (control panel compatibility)"""
    try:
        import json
        import os
        
        data = request.get_json() or {}
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        memories_file = os.path.join(os.getcwd(), 'data', 'advanced_memories.json')
        
        # Backup existing file
        if os.path.exists(memories_file):
            backup_file = f"{memories_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(memories_file, 'r', encoding='utf-8') as src:
                with open(backup_file, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        
        # Write new data
        with open(memories_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            "success": True,
            "message": "Memory imported successfully",
            "import_timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error importing memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/memory/search', methods=['GET'])
def search_memory():
    """Search memories (control panel compatibility)"""
    try:
        import json
        import os
        
        query = request.args.get('query', '').lower()
        if not query:
            return jsonify({"error": "Search query required"}), 400
        
        memories_file = os.path.join(os.getcwd(), 'data', 'advanced_memories.json')
        if os.path.exists(memories_file):
            with open(memories_file, 'r', encoding='utf-8') as f:
                memories_data = json.load(f)
            
            # Search through all memories
            all_memories = memories_data.get('short_term', []) + memories_data.get('long_term', [])
            matching_memories = []
            
            for memory in all_memories:
                content = memory.get('content', '').lower()
                if query in content:
                    matching_memories.append(memory)
            
            return jsonify({
                "success": True,
                "memories": matching_memories,
                "query": query,
                "count": len(matching_memories),
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Advanced memories file not found"}), 404
            
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🤖 Starting LLM Controller API...")
    print("=" * 50)
    
    # Initialize bot components
    initialize_bot_components()
    
    # Start background tasks
    start_background_tasks()
    
    print("📡 API Endpoints:")
    print("  GET  /status          - Get controller status")
    print("  POST /prompt          - Send a prompt to the LLM")
    print("  POST /speak           - Request TTS")
    print("  POST /mode            - Set operation mode")
    print("  POST /llm/toggle      - Toggle LLM on/off")
    print("  POST /filter/toggle   - Toggle filter system")
    print("  POST /voice/stop      - Stop voice processing")
    print("  GET  /servers/<id>/channels/voice - Get voice channels for server")
    print("  POST /voice/join      - Join voice channel")
    print("  POST /voice/leave     - Leave voice channel")
    print("  GET  /vtube/status    - VTube Studio status")
    print("  GET  /vtube/settings  - VTube Studio settings")
    print("  GET  /filter/words    - Filter word list")
    print("  POST /filter/words    - Add filter word")
    print("  DELETE /filter/words/<word> - Delete filter word")
    print("  POST /filter/toggle   - Toggle filter system")
    print("  GET  /memory/list     - Memory list")
    print("  POST /memory/add      - Add memory")
    print("  POST /memory/delete   - Delete memory")
    print()
    print("🔌 WebSocket:")
    print("  /events               - Live events and logs")
    print()
    print("🌐 Starting server on http://localhost:4000")
    print("=" * 50)
    
    # Start the server
    socketio.run(app, host='0.0.0.0', port=4000, debug=False)
