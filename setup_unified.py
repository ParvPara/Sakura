#!/usr/bin/env python3
"""
Setup script for Unified Sakura Bot with Integrated Decision Router
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA not available (will use CPU)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check CUDA")
        return False

def create_env_file():
    """Create .env file with configuration"""
    env_content = """# Sakura Bot Configuration
# Discord Bot
DISCORD_TOKEN=your_discord_token_here

# Router Settings (optional)
ROUTER_MODEL=
N_GPU_LAYERS=999
N_CTX=1024
N_THREADS=6
N_BATCH=256

# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Voice Settings
TTS_ENABLED=true
STT_ENABLED=true
VOICE_ENABLED=true

# Web Search
WEBSEARCH_ENABLED=true
SEARCH_API_KEY=

# Memory
MEMORY_ENABLED=true
MEMORY_FILE=data/people.json

# Filter
FILTER_ENABLED=true
FILTER_FILE=data/filtered_words.json
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✓ .env file created")

def download_router_model():
    """Download a recommended router model"""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if model_path.exists():
        print("✓ Router model already downloaded")
        return str(model_path)
    
    print("Downloading TinyLlama 1.1B router model...")
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if run_command(f"wget -O {model_path} {url}"):
        print("✓ Router model downloaded")
        return str(model_path)
    else:
        print("❌ Router model download failed")
        return None

def update_env_with_model(model_path):
    """Update .env file with model path"""
    if not model_path:
        return
    
    try:
        with open(".env", "r") as f:
            content = f.read()
        
        # Replace empty ROUTER_MODEL with actual path
        content = content.replace("ROUTER_MODEL=", f"ROUTER_MODEL={model_path}")
        
        with open(".env", "w") as f:
            f.write(content)
        
        print("✓ .env file updated with router model path")
        
    except Exception as e:
        print(f"❌ Failed to update .env file: {e}")


def main():
    """Main setup function"""
    print("🚀 Unified Sakura Bot Setup")
    print("=" * 40)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_cuda()
    
    # Create configuration
    create_env_file()
    
    # Download router model (optional)
    print("\nRouter Model Setup (Optional):")
    print("The bot can work with or without a dedicated router model.")
    print("Without a model, it will use intelligent fallback logic.")
    
    download_model = input("Download router model? (y/n): ").lower().strip()
    if download_model == 'y':
        model_path = download_router_model()
        if model_path:
            update_env_with_model(model_path)
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your Discord token")
    print("2. Start Ollama server (if using local LLM)")
    print("3. Run the bot:")
    print("   python main.py")
    print("\nConfiguration:")
    print("- Router: Uses LLM model if available, otherwise fallback logic")
    print("- Search: Automatic web search for questions")
    print("- DM: Handles private message requests")
    print("- Chat: Normal conversation with Sakura")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
