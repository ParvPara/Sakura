#!/usr/bin/env python3
"""
Ultra-Fast Tool Router Service (vLLM Variant)
Decides between SEARCH, DM, or NONE actions in ≤300ms using vLLM
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from enum import Enum
import re
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import requests

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
    vllm_available: bool

# Configuration
class Config:
    VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8001"))

# FastAPI app
app = FastAPI(title="Tool Router (vLLM)", version="1.0.0")

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

def check_vllm_health() -> bool:
    """Check if vLLM server is available"""
    try:
        response = requests.get(f"{Config.VLLM_ENDPOINT}/models", timeout=5)
        return response.status_code == 200
    except:
        return False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    vllm_ok = check_vllm_health()
    return HealthResponse(
        ok=vllm_ok,
        model_name=Config.MODEL_NAME,
        vllm_available=vllm_ok
    )

@app.post("/route", response_model=RouteResponse)
async def route_message(request: RouteRequest):
    """Route a message to determine the appropriate action"""
    if not check_vllm_health():
        raise HTTPException(status_code=500, detail="vLLM server not available")
    
    # Truncate text for logging (privacy)
    log_text = request.text[:512] + "..." if len(request.text) > 512 else request.text
    logger.info(f"Routing request: {log_text} (channel: {request.channel_type})")
    
    # Build prompt
    prompt = build_prompt(request)
    
    # Time the inference
    start_time = time.time()
    
    try:
        # Call vLLM API
        payload = {
            "model": Config.MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "stop": ["\n", " ", "<", ">"]
        }
        
        response = requests.post(
            f"{Config.VLLM_ENDPOINT}/chat/completions",
            json=payload,
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"vLLM API error: {response.status_code}")
        
        result = response.json()
        raw_output = result["choices"][0]["message"]["content"].strip()
        
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
    logger.info("=== Tool Router Configuration (vLLM) ===")
    logger.info(f"vLLM Endpoint: {Config.VLLM_ENDPOINT}")
    logger.info(f"Model: {Config.MODEL_NAME}")
    logger.info(f"Host: {Config.HOST}")
    logger.info(f"Port: {Config.PORT}")
    logger.info("=======================================")
    
    # Check vLLM availability
    if not check_vllm_health():
        logger.error("vLLM server not available!")
        logger.error("Start vLLM server first with:")
        logger.error("vllm serve microsoft/DialoGPT-medium --max-model-len 1024")
        exit(1)
    
    # Start server
    uvicorn.run(
        "router_server_vllm:app",
        host=Config.HOST,
        port=Config.PORT,
        log_level="info"
    )
