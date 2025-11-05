# Sakura Bot - Advanced Discord AI Chat Bot

Sakura is a sophisticated Discord bot featuring advanced AI capabilities including voice interaction, text-to-speech, memory systems, and intelligent agentic tool routing. Built to demonstrate the advanced capabilities of current AI technology.

## This repository was made to demonstrate current avilable AI technologies thus, has reduced capabilities or is not maintained/updated frequently compared to its development repository.

## Features

### Core Capabilities
- **Voice Interaction**: Real-time voice chat with WebRTC VAD and Whisper STT
- **Text-to-Speech**: Azure Cognitive Services integration for natural voice synthesis
- **Advanced Memory**: Persistent memory system with context-aware conversations
- **Intelligent Routing**: Smart message routing with tool execution capabilities
- **Web Search**: Integrated Brave search for real-time information
- **Filter System**: Content moderation and filtering capabilities
- **Multi-LLM Support**: Ollama and llama.cpp integration with CUDA acceleration

### Technical Features
- **Flask Backend**: High-performance async API server
- **WebSocket Support**: Real-time communication for voice and chat via SocketIO
- **Modular Architecture**: Extensible plugin system for tools and handlers
- **Cloudflare Deployment**: Ready for edge deployment with Workers and Cloudflare Tunnel
- **Security**: Built-in rate limiting, API key authentication, and access controls

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for local LLM inference)
- Discord Bot Token ([Create a bot](https://discord.com/developers/applications))
- Azure Speech Services Key (for TTS) - [Get free key](https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/)
- Brave Search API Key (optional, for web search) - [Get API key](https://brave.com/search/api/)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Sakura-Bot-master

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

#### Quick Setup (Recommended)

Run the setup script to automatically configure your environment:

```bash
python setup_unified.py
```

This will:
- Check Python version and CUDA availability
- Create a `.env` file with all required configuration variables
- Optionally download a router model

#### Manual Configuration

1. Copy the configuration template:

```bash
cp config/config.env .env
```

2. Edit `.env` with your settings:

```env
# Discord Configuration (REQUIRED)
DISCORD_TOKEN=your_discord_bot_token
APPLICATION_ID=your_discord_application_id

# LLM Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=Sakura

# Router Settings (Optional - for local routing model)
ROUTER_MODEL=/path/to/your/model.gguf
N_GPU_LAYERS=999
N_CTX=1024
N_THREADS=6
N_BATCH=256

# Azure Speech Services (Required for TTS)
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region

# Web Search (Optional)
BRAVE_SEARCH_API_KEY=your_brave_search_api_key

# API Security (Optional but recommended)
API_KEY=your_secure_api_key_here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password_here
SECRET_KEY=your_secret_key_here

# Streaming Configuration
ENABLE_STREAMING=true
STREAMING_CHUNK_SIZE=50
STREAMING_TTS_ENABLED=true
```

### 4. Start the Bot

The bot consists of multiple components that can run separately:

#### Option 1: Full System (Recommended)

```bash
# Start LLM Controller (runs on port 4000)
python llm_controller.py

# In another terminal, start the Discord bot (includes API on port 5000)
python main.py
```

#### Option 2: With Router Server

```bash
# Start router server (runs on port 8000)
python router_server.py

# Start LLM Controller (runs on port 4000)
python llm_controller.py

# Start Discord bot (runs on port 5000)
python main.py
```

#### Option 3: Windows Batch Script

On Windows, you can use the provided batch script:

```bash
start_llm_system.bat
```

This starts all components automatically.

## Architecture

### Core Components

```
Sakura/
├── chat_handler.py          # Main conversation logic
├── voice_handler.py         # Voice interaction management
├── tts_handler.py           # Text-to-speech processing
├── whisper_handler.py       # Speech-to-text with Whisper
├── memory_system.py         # Advanced memory management
├── filter_handler.py        # Content moderation
├── websearch.py             # Web search integration
├── llm_wrapper.py          # LLM backend abstraction
├── decider.py              # Decision-making logic
└── discord_DM.py           # Direct message handling

router/
├── intent_router.py         # Message routing logic
└── tool_guard.py           # Tool execution security and rate limiting

tools/
├── tool_executor.py         # Tool execution engine
├── search_tool.py           # Search functionality
└── dm_tool.py              # Direct messaging tools
```

### API Endpoints

The bot exposes several Flask API endpoints for integration:

#### Bot API (Port 5000)
- `GET /api/status` - Health check and bot status
- `GET /api/ai/status` - AI system status
- `POST /api/discord/send_message` - Send Discord message
- `GET /api/servers` - List Discord servers
- `GET /api/servers/{server_id}/channels` - List channels
- `POST /api/message/stop` - Stop current message generation
- `GET /api/messages/status` - Message queue status
- `GET /api/memories` - View bot memories
- `GET /api/memories/stats` - Memory statistics
- `POST /api/memories/clear` - Clear memories
- `GET /api/filter/status` - Filter system status

#### LLM Controller API (Port 4000)
- `GET /status` - Controller status
- `POST /prompt` - Process prompt
- `POST /speak` - Generate speech
- `POST /mode` - Set mode
- `POST /llm/toggle` - Toggle LLM on/off
- `POST /filter/toggle` - Toggle filter on/off
- `POST /voice/stop` - Stop voice generation
- `WebSocket /events` - Real-time events via SocketIO

#### Router API (Port 8000)
- `POST /route` - Route messages to appropriate handlers

### Authentication

API endpoints support API key authentication via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your_api_key_here" http://localhost:5000/api/status
```

Set the `API_KEY` environment variable in your `.env` file to enable authentication.

## Voice Features

### Speech-to-Text
- **Whisper Integration**: High-accuracy speech recognition
- **WebRTC VAD**: Voice activity detection for efficient processing
- **Real-time Processing**: Low-latency audio processing
- **Multiple Models**: Support for different Whisper model sizes

### Text-to-Speech
- **Azure Speech**: Natural-sounding voice synthesis
- **Multiple Voices**: Support for different voice profiles
- **SSML Support**: Advanced speech markup for better pronunciation
- **Streaming TTS**: Real-time voice generation

## Memory System

### Advanced Memory Features
- **Context Awareness**: Maintains conversation context across sessions
- **User Profiles**: Personalized memory for individual users
- **Long-term Storage**: Persistent memory across bot restarts
- **Memory Filtering**: Intelligent memory management and cleanup
- **Relationship Tracking**: Tracks relationships between users and the bot

Memory data is stored in:
- `data/people.json` - User profiles and information
- `data/advanced_memories.json` - Conversation memories

## Web Interface

# The frontend repository has been kept private to protect sensitive information with the frontend code, to use the bot in a discord call use !join

The bot includes a comprehensive web interface located in `sakura-frontend/`:

- **Real-time Chat**: Web-based chat interface
- **Voice Controls**: Voice interaction controls
- **Memory Management**: View and manage bot memories
- **Filter Configuration**: Content filtering settings
- **Server Management**: Discord server and channel management
- **VTuber Integration**: VTube Studio integration controls

### Frontend Deployment

The frontend can be deployed to Cloudflare Pages. See `sakura-frontend/README.md` for detailed deployment instructions.

For local development:
```bash
cd sakura-frontend
python -m http.server 8000
```

Then access at `http://localhost:8000` and configure `config.js` to point to your local API endpoints.

## Deployment

### Local Development

```bash
# Start all components
python llm_controller.py  # Terminal 1
python main.py            # Terminal 2
```

### Production with Cloudflare Tunnel

1. Install Cloudflare Tunnel:
```bash
# Download cloudflared from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
```

2. Configure tunnels in `config/` directory:
   - `llm_tunnel.yml` - LLM Controller tunnel
   - `bot_tunnel.yml` - Bot API tunnel
   - `discord_tunnel.yml` - Discord bot tunnel

3. Update tunnel configuration files with your tunnel IDs and domain

4. Start tunnels:
```bash
cloudflared tunnel --config config/llm_tunnel.yml run
cloudflared tunnel --config config/bot_tunnel.yml run
```

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DISCORD_TOKEN` | Discord bot token | - | Yes |
| `APPLICATION_ID` | Discord application ID | - | No |
| `OLLAMA_HOST` | Ollama server URL | http://localhost:11434 | No |
| `OLLAMA_MODEL` | Ollama model name | Sakura | No |
| `AZURE_SPEECH_KEY` | Azure Speech Services key | - | Yes (for TTS) |
| `AZURE_SPEECH_REGION` | Azure Speech region | - | Yes (for TTS) |
| `BRAVE_SEARCH_API_KEY` | Brave Search API key | - | No |
| `ROUTER_MODEL` | Path to GGUF router model | - | No |
| `N_GPU_LAYERS` | GPU layers for inference | 999 | No |
| `N_CTX` | Context window size | 1024 | No |
| `N_THREADS` | CPU threads | 6 | No |
| `N_BATCH` | Batch size | 256 | No |
| `API_KEY` | API authentication key | - | No (recommended) |
| `ADMIN_USERNAME` | Admin username | admin | No |
| `ADMIN_PASSWORD` | Admin password | - | No |
| `ENABLE_STREAMING` | Enable streaming responses | true | No |
| `STREAMING_CHUNK_SIZE` | Streaming chunk size | 50 | No |

## Troubleshooting

### Common Issues

**Bot not responding:**
1. Check Discord token in `.env` file
2. Verify bot permissions in Discord server (requires `Send Messages`, `Connect`, `Speak` permissions)
3. Check console logs for errors
4. Ensure the bot is invited to your server with proper permissions

**Voice not working:**
1. Ensure Azure Speech key is configured in `.env`
2. Check microphone permissions in Discord
3. Verify WebRTC VAD installation
4. Check that Opus library is loaded (check console for Opus loading messages)

**LLM errors:**
1. Verify Ollama is running (`curl http://localhost:11434/api/tags`)
2. Check model name matches your Ollama model
3. Ensure CUDA drivers are installed (if using GPU)
4. Check router model path if using local router

**API connection issues:**
1. Verify LLM Controller is running on port 4000
2. Check firewall settings
3. Ensure API key is set if authentication is enabled
4. Check CORS configuration if accessing from web interface

### Performance Optimization

**For better voice latency:**
- Use local Whisper model instead of cloud API
- Optimize WebRTC VAD settings
- Reduce audio buffer sizes
- Use faster Azure Speech voice models

**For faster text processing:**
- Use smaller GGUF models for routing
- Increase GPU layer allocation
- Optimize context window size
- Use quantized models (Q4, Q5) for faster inference

**For reduced memory usage:**
- Reduce `N_CTX` context window size
- Lower `N_GPU_LAYERS` if VRAM limited
- Use smaller Whisper models
- Clear old memories periodically

## Development

### Project Structure

```
Sakura-Bot-master/
├── main.py                 # Main Discord bot entry point
├── llm_controller.py       # LLM Controller API server
├── router_server.py        # Router server
├── api_integration.py      # Flask API integration
├── config.py               # Configuration loader
├── setup_unified.py        # Setup script
├── config/                 # Configuration files
│   ├── config.env          # Environment template
│   └── *.yml               # Cloudflare tunnel configs
├── data/                   # Data storage
│   ├── people.json         # User profiles
│   └── advanced_memories.json  # Conversation memories
├── Sakura/                 # Core bot modules
├── router/                 # Routing logic
├── tools/                  # Tool implementations
├── sakura-frontend/        # Web interface
└── tests/                  # Test files
```

### Adding New Tools

1. Create tool class in `tools/` directory
2. Implement required methods from `ToolExecutor` interface
3. Register with `ToolExecutor` in `tools/tool_executor.py`
4. Add routing logic in `router/intent_router.py`
5. Update rate limits in `router/tool_guard.py` if needed

### Custom LLM Backends

1. Extend `LLMWrapper` class in `Sakura/llm_wrapper.py`
2. Implement required interface methods
3. Update configuration for new backend
4. Test with existing handlers

### Testing

Run tests with:
```bash
python -m pytest tests/
```

## Security Notes

- **Never commit `.env` files** - They contain sensitive credentials
- **Use API keys** - Set `API_KEY` environment variable for API authentication
- **Secure your tunnels** - Use Cloudflare Access for additional security
- **Review CORS settings** - Update allowed origins in `api_integration.py` for production
- **Protect admin endpoints** - Use strong `ADMIN_PASSWORD` in production

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

**TL;DR**: You're free to use, modify, distribute, and use this software for any purpose, including commercial use, as long as you include the original copyright notice and license text.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review console logs for detailed error messages
3. Open an issue on GitHub with:
   - System specifications (OS, Python version, GPU)
   - Error messages and stack traces
   - Steps to reproduce
   - Configuration (with sensitive data removed)

---

**Note**: This bot requires significant computational resources, especially for voice processing and LLM inference. A strong GPU with high VRAM is highly recommended for optimal performance.

