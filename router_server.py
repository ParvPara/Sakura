#!/usr/bin/env python3
"""
Ultra-Fast Tool Router Service
Decides between SEARCH, DM, or NONE actions in ≤300ms
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from enum import Enum
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Try to import llama-cpp-python with CUDA support
try:
    from llama_cpp import Llama
    CUDA_AVAILABLE = True
except ImportError:
    print("Warning: llama-cpp-python not found. Install with: pip install llama-cpp-python[cuda]")
    CUDA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums
class ChannelType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"

class Action(str, Enum):
    SEARCH = "SEARCH"
    DM = "DM"
    NONE = "NONE"

# Pydantic models
class RouteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    channel_type: ChannelType = ChannelType.PUBLIC
    mentions: Optional[str] = None

class RouteResponse(BaseModel):
    action: Action
    latency_ms: int
    raw: str

class HealthResponse(BaseModel):
    ok: bool
    model_name: str
    cuda_available: bool

# Global model instance
llm: Optional[Llama] = None
model_name: str = "unknown"

# Configuration
class Config:
    ROUTER_MODEL = os.getenv("ROUTER_MODEL", "")
    N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "999"))
    N_CTX = int(os.getenv("N_CTX", "1024"))
    N_THREADS = int(os.getenv("N_THREADS", "6"))
    N_BATCH = int(os.getenv("N_BATCH", "256"))
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))

# FastAPI app
app = FastAPI(title="Tool Router", version="1.0.0")

def load_model() -> bool:
    """Load the LLM model with CUDA optimization"""
    global llm, model_name
    
    if not CUDA_AVAILABLE:
        logger.error("CUDA not available - install llama-cpp-python[cuda]")
        return False
    
    if not Config.ROUTER_MODEL or not os.path.exists(Config.ROUTER_MODEL):
        logger.error(f"Model file not found: {Config.ROUTER_MODEL}")
        return False
    
    try:
        logger.info(f"Loading model: {Config.ROUTER_MODEL}")
        logger.info(f"Config: GPU layers={Config.N_GPU_LAYERS}, ctx={Config.N_CTX}, threads={Config.N_THREADS}")
        
        llm = Llama(
            model_path=Config.ROUTER_MODEL,
            n_gpu_layers=Config.N_GPU_LAYERS,
            n_ctx=Config.N_CTX,
            n_threads=Config.N_THREADS,
            n_batch=Config.N_BATCH,
            verbose=False
        )
        
        model_name = os.path.basename(Config.ROUTER_MODEL)
        logger.info(f"Model loaded successfully: {model_name}")
        
        # Warmup inference
        logger.info("Running warmup inference...")
        warmup_start = time.time()
        warmup_response = llm(
            "You are a tool router. Respond with EXACTLY ONE token from this set: <SEARCH>, <DM>, <NONE>. No extra text, punctuation, or explanation.\n\nUser: hello\nRouter:",
            max_tokens=8,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop=["\n", " ", "<", ">"]
        )
        warmup_latency = (time.time() - warmup_start) * 1000
        logger.info(f"Warmup completed in {warmup_latency:.1f}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def normalize_action(raw_output: str) -> Action:
    """Normalize raw model output to valid action"""
    if not raw_output:
        return Action.NONE
    
    # Clean the output
    raw_clean = raw_output.strip()
    
    # Try to extract bracketed token
    bracket_match = re.search(r'<(SEARCH|DM|NONE)>', raw_clean)
    if bracket_match:
        return Action(bracket_match.group(1))
    
    # Try keyword matching
    raw_lower = raw_clean.lower()
    if 'search' in raw_lower:
        return Action.SEARCH
    elif 'dm' in raw_lower or 'direct' in raw_lower or 'private' in raw_lower:
        return Action.DM
    elif 'none' in raw_lower or 'chat' in raw_lower or 'talk' in raw_lower:
        return Action.NONE
    
    # Default to NONE
    return Action.NONE

def build_prompt(request: RouteRequest) -> str:
    """Build the routing prompt"""
    prompt = "You are a tool router. Respond with EXACTLY ONE token from this set: <SEARCH>, <DM>, <NONE>. No extra text, punctuation, or explanation.\n\n"
    
    # Add context
    if request.channel_type == ChannelType.PRIVATE:
        prompt += "Channel: private\n"
    else:
        prompt += "Channel: public\n"
    
    if request.mentions:
        prompt += f"Mentions: {request.mentions}\n"
    
    prompt += f"User: {request.text}\nRouter:"
    return prompt

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    if not load_model():
        logger.error("Failed to load model - server will not function properly")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        ok=llm is not None,
        model_name=model_name,
        cuda_available=CUDA_AVAILABLE
    )

@app.post("/route", response_model=RouteResponse)
async def route_message(request: RouteRequest):
    """Route a message to determine the appropriate action"""
    if not llm:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Truncate text for logging (privacy)
    log_text = request.text[:512] + "..." if len(request.text) > 512 else request.text
    logger.info(f"Routing request: {log_text} (channel: {request.channel_type})")
    
    # Build prompt
    prompt = build_prompt(request)
    
    # Time the inference
    start_time = time.time()
    
    try:
        # Run inference
        response = llm(
            prompt,
            max_tokens=8,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            stop=["\n", " ", "<", ">"]
        )
        
        # Extract response
        raw_output = response.get('choices', [{}])[0].get('text', '').strip()
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Normalize action
        action = normalize_action(raw_output)
        
        logger.info(f"Route decision: {action} (latency: {latency_ms}ms, raw: '{raw_output}')")
        
        return RouteResponse(
            action=action,
            latency_ms=latency_ms,
            raw=raw_output
        )
        
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Print configuration
    logger.info("=== Tool Router Configuration ===")
    logger.info(f"Model: {Config.ROUTER_MODEL}")
    logger.info(f"GPU Layers: {Config.N_GPU_LAYERS}")
    logger.info(f"Context: {Config.N_CTX}")
    logger.info(f"Threads: {Config.N_THREADS}")
    logger.info(f"Batch: {Config.N_BATCH}")
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    logger.info("=================================")
    
    # Start server
    uvicorn.run(
        "router_server:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )
