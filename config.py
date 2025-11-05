import os
from dotenv import load_dotenv

load_dotenv('.env')

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
APPLICATION_ID = os.getenv('APPLICATION_ID')
COMMAND_PREFIX = '!'

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = "Sakura" #Custom LLM Model #rekt

ENABLE_STREAMING = os.getenv('ENABLE_STREAMING', 'true').lower() == 'true'
STREAMING_CHUNK_SIZE = int(os.getenv('STREAMING_CHUNK_SIZE', '50'))
STREAMING_TTS_ENABLED = os.getenv('STREAMING_TTS_ENABLED', 'true').lower() == 'true'

AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION')

BRAVE_SEARCH_API_KEY = os.getenv('BRAVE_SEARCH_API_KEY')

ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'your_secure_password_here')
SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24))

SAMPLE_RATE = 48000
CHUNK_SIZE = 960
CHANNELS = 2

# Router Settings
ROUTER_MODEL = os.getenv('ROUTER_MODEL', 'qwen2.5-1.5b-q4-router')
N_GPU_LAYERS = int(os.getenv('N_GPU_LAYERS', '999'))
N_CTX = int(os.getenv('N_CTX', '1024'))
N_THREADS = int(os.getenv('N_THREADS', '6'))
N_BATCH = int(os.getenv('N_BATCH', '256'))