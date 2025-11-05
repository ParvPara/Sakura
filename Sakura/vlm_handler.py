"""
VLM Handler for Sakura Bot
Handles image analysis using local VLM models
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import requests
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

class VLMHandler:
    def __init__(self, vlm_host: str = "http://localhost:11434", vlm_model: str = "llava:7b"):
        self.vlm_host = vlm_host
        self.vlm_model = vlm_model
        self.last_summary = None
        self.last_capture_time = None
        self.capture_source = None
        
        # Performance tracking
        self.summary_count = 0
        self.last_vlm_call_time = 0
        self.rate_limit_secs = 0.5  # Minimum time between VLM calls
        
    async def analyze_image(self, image_data: bytes, source: str = "screen") -> Dict:
        """Analyze image using VLM and return structured JSON"""
        try:
            # Rate limiting check
            current_time = time.time()
            if current_time - self.last_vlm_call_time < self.rate_limit_secs:
                logger.debug("VLM call rate limited")
                return self._create_fallback_summary(source, "Rate limited")
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Choose prompt based on source
            if source == "screen":
                prompt = self._get_screen_prompt()
            else:  # camera
                prompt = self._get_camera_prompt()
            
            # Call VLM
            vlm_response = await self._call_vlm(image_b64, prompt, source)
            
            # Validate and parse JSON
            summary = self._validate_and_parse_json(vlm_response)
            
            # Store summary
            self.last_summary = summary
            self.last_capture_time = current_time
            self.last_vlm_call_time = current_time
            self.capture_source = source
            self.summary_count += 1
            
            logger.info(f"VLM analysis completed for {source} capture (#{self.summary_count})")
            return summary
            
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return self._create_fallback_summary(source, str(e))
    
    def _get_screen_prompt(self) -> str:
        """Get VLM prompt for screen analysis"""
        return """Analyze this desktop screenshot and return ONLY valid JSON. No markdown, no code blocks, no extra text.

Required JSON format:
{
  "scene_type": "screen",
  "high_level_summary": "Brief description in 18 words or less",
  "entities": [],
  "text_blocks": [{"bbox": [x, y, w, h], "text": "content"}],
  "sensitive_indicators": [],
  "confidence": 0.8,
  "timestamp_iso": "2025-09-29T17:42:00Z"
}

Rules:
- scene_type must be "screen"
- high_level_summary: literal description, ≤18 words
- entities: array of objects with type/name/details
- text_blocks: up to 3 largest text regions with bbox [x,y,w,h], each ≤160 chars, include ALL visible text
- sensitive_indicators: array of sensitive content found
- confidence: float 0.0-1.0
- timestamp_iso: current UTC time in ISO-8601 format
- Return ONLY the JSON object, nothing else"""
    
    def _get_camera_prompt(self) -> str:
        """Get VLM prompt for camera analysis"""
        return """Analyze this camera image and return ONLY valid JSON. No markdown, no code blocks, no extra text.

Required JSON format:
{
  "scene_type": "camera",
  "high_level_summary": "Brief description in 18 words or less",
  "entities": [],
  "text_blocks": [{"bbox": [x, y, w, h], "text": "content"}],
  "sensitive_indicators": [],
  "confidence": 0.8,
  "timestamp_iso": "2025-09-29T17:42:00Z"
}

Rules:
- scene_type must be "camera"
- high_level_summary: literal description, ≤18 words
- entities: array of objects with type/name/details
- text_blocks: up to 3 most readable text regions with bbox [x,y,w,h], each ≤160 chars, include ALL visible text
- sensitive_indicators: array of sensitive content found
- confidence: float 0.0-1.0
- timestamp_iso: current UTC time in ISO-8601 format
- Return ONLY the JSON object, nothing else"""
    
    async def _call_vlm(self, image_b64: str, prompt: str, source: str = "screen") -> str:
        """Call local VLM model"""
        try:
            # Check if VLM service is available first
            try:
                loop = asyncio.get_event_loop()
                health_response = await loop.run_in_executor(
                    None, 
                    lambda: requests.get(f"{self.vlm_host}/api/tags")
                )
                if health_response.status_code != 200:
                    logger.warning(f"VLM service not available: {health_response.status_code}")
                    return json.dumps(self._create_fallback_summary(source, "VLM service unavailable"))
            except requests.exceptions.RequestException:
                logger.warning("VLM service not reachable, using fallback")
                return json.dumps(self._create_fallback_summary(source, "VLM service not reachable"))
            
            # Prepare request data with reduced processing load
            data = {
                "model": self.vlm_model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 150,  # Further reduced for faster processing
                    "num_ctx": 1024,     # Smaller context window
                    "num_gpu": 1,        # Use GPU if available
                    "num_thread": 2       # Fewer CPU threads
                }
            }
            
            # Make async request without timeout - let VLM run as long as needed
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(f"{self.vlm_host}/api/generate", json=data)
            )
            
            if response.status_code == 200:
                result = response.json()
                vlm_output = result.get('response', '')
                logger.debug(f"VLM raw output: {vlm_output[:200]}...")
                return vlm_output
            elif response.status_code == 404:
                logger.warning(f"VLM model '{self.vlm_model}' not found, using fallback")
                return json.dumps(self._create_fallback_summary(source, "VLM model not found"))
            else:
                logger.error(f"VLM API call failed: {response.status_code} - {response.text}")
                return json.dumps(self._create_fallback_summary(source, f"VLM API error: {response.status_code}"))
                
        except Exception as e:
            logger.error(f"VLM API call failed: {e}")
            raise e
    
    def _create_fallback_summary(self, source: str = "screen", error: str = "VLM unavailable") -> Dict:
        """Create a fallback summary when VLM is not available"""
        current_time = datetime.now(timezone.utc).isoformat()
        return {
            "scene_type": source,
            "high_level_summary": f"Screen capture successful but VLM analysis unavailable: {error}",
            "entities": [
                {"type": "system", "name": "VLM Service", "details": "Vision analysis service not available"}
            ],
            "text_blocks": [],
            "sensitive_indicators": [],
            "confidence": 0.1,
            "timestamp_iso": current_time
        }
    
    def _validate_and_parse_json(self, vlm_response: str) -> Dict:
        """Validate and parse VLM JSON response"""
        try:
            # Ensure we have a string
            if not isinstance(vlm_response, str):
                logger.warning(f"VLM response is not a string: {type(vlm_response)}")
                return self._create_fallback_summary("unknown", "Invalid response type from VLM")
                
            # Clean response (remove any markdown or extra text)
            response_clean = vlm_response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            # Try to parse JSON - if it fails, attempt repair
            try:
                summary = json.loads(response_clean)
            except json.JSONDecodeError as e:
                logger.warning(f"VLM JSON validation failed: {e}")
                # Try to repair the JSON
                repaired_json = self._repair_json(response_clean)
                if repaired_json:
                    try:
                        summary = json.loads(repaired_json)
                        logger.info("VLM JSON successfully repaired")
                    except json.JSONDecodeError as repair_error:
                        logger.warning(f"JSON repair also failed: {repair_error}")
                        # Create a fallback with the raw response
                        current_time = datetime.now(timezone.utc).isoformat()
                        summary = {
                            "scene_type": "screen",
                            "high_level_summary": f"VLM analysis completed but JSON parsing failed. Raw response: {response_clean[:100]}...",
                            "entities": [],
                            "text_blocks": [],
                            "sensitive_indicators": [],
                            "confidence": 0.5,
                            "timestamp_iso": current_time
                        }
                else:
                    # Create a fallback with the raw response
                    current_time = datetime.now(timezone.utc).isoformat()
                    summary = {
                        "scene_type": "screen",
                        "high_level_summary": f"VLM analysis completed but JSON parsing failed. Raw response: {response_clean[:100]}...",
                        "entities": [],
                        "text_blocks": [],
                        "sensitive_indicators": [],
                        "confidence": 0.5,
                        "timestamp_iso": current_time
                    }
            
            # Validate required fields
            required_fields = [
                'scene_type', 'high_level_summary', 'entities', 
                'text_blocks', 'sensitive_indicators', 'confidence', 'timestamp_iso'
            ]
            
            for field in required_fields:
                if field not in summary:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate field types and constraints
            if not isinstance(summary['entities'], list):
                summary['entities'] = []
            if not isinstance(summary['text_blocks'], list):
                summary['text_blocks'] = []
            if not isinstance(summary['sensitive_indicators'], list):
                summary['sensitive_indicators'] = []
            
            # Ensure confidence is float
            summary['confidence'] = float(summary.get('confidence', 0.0))
            
            # Ensure timestamp is valid ISO format
            if 'timestamp_iso' not in summary or not summary['timestamp_iso']:
                summary['timestamp_iso'] = datetime.now(timezone.utc).isoformat()
            
            return summary
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"VLM JSON validation failed: {e}")
            # Try repair once
            repaired_json = self._repair_json(vlm_response)
            if repaired_json:
                try:
                    return json.loads(repaired_json)
                except:
                    pass
            return self._create_fallback_summary("unknown", "Invalid JSON from VLM")
    
    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON issues from VLM output"""
        try:
            # Ensure we have a string
            if not isinstance(json_str, str):
                logger.warning(f"JSON repair received non-string: {type(json_str)}")
                return None
                
            repaired = json_str.strip()
            
            # Remove any markdown formatting
            if repaired.startswith('```json'):
                repaired = repaired[7:]
            if repaired.startswith('```'):
                repaired = repaired[3:]
            if repaired.endswith('```'):
                repaired = repaired[:-3]
            repaired = repaired.strip()
            
            # Fix unterminated strings - simple approach
            lines = repaired.split('\n')
            fixed_lines = []
            in_string = False
            string_start_line = 0
            
            for i, line in enumerate(lines):
                if not in_string:
                    # Check if this line starts a string
                    quote_count = line.count('"')
                    if quote_count % 2 == 1:  # Odd number of quotes
                        in_string = True
                        string_start_line = i
                else:
                    # We're in a string, look for the end
                    if '"' in line:
                        in_string = False
                    else:
                        # Still in string, add closing quote
                        line = line + '"'
                        in_string = False
                
                fixed_lines.append(line)
            
            # If we're still in a string at the end, close it
            if in_string:
                fixed_lines[-1] = fixed_lines[-1] + '"'
            
            repaired = '\n'.join(fixed_lines)
            
            # Fix common JSON issues
            # Fix missing commas between objects
            repaired = re.sub(r'}\s*{', '}, {', repaired)
            repaired = re.sub(r']\s*\[', '], [', repaired)
            
            # Fix missing commas between properties
            repaired = re.sub(r'"\s*\n\s*"', '",\n    "', repaired)
            repaired = re.sub(r'}\s*\n\s*"', '},\n    "', repaired)
            
            # Remove trailing commas
            repaired = re.sub(r',\s*}', '}', repaired)
            repaired = re.sub(r',\s*]', ']', repaired)
            
            # Fix unquoted keys
            repaired = re.sub(r'(\w+):', r'"\1":', repaired)
            
            # Convert single quotes to double quotes
            repaired = repaired.replace("'", '"')
            
            # Try to parse the repaired JSON
            json.loads(repaired)
            return repaired
            
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            return None
    
    
    def get_last_summary(self) -> Optional[Dict]:
        """Get the last valid summary"""
        return self.last_summary
    
    def is_summary_stale(self, stale_after_secs: int = 120) -> bool:
        """Check if last summary is stale"""
        if not self.last_capture_time:
            return True
        return (time.time() - self.last_capture_time) > stale_after_secs
    
    def get_summary_age_seconds(self) -> int:
        """Get age of last summary in seconds"""
        if not self.last_capture_time:
            return 999999
        return int(time.time() - self.last_capture_time)
    
    def get_status(self) -> Dict:
        """Get VLM handler status"""
        return {
            "has_summary": self.last_summary is not None,
            "last_capture_time": self.last_capture_time,
            "capture_source": self.capture_source,
            "is_stale": self.is_summary_stale(),
            "age_seconds": self.get_summary_age_seconds(),
            "last_confidence": self.last_summary.get("confidence", 0.0) if self.last_summary else 0.0
        }
