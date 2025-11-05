"""
Live Vision Controller for Sakura Bot
Manages continuous vision capture with event-driven freshness
"""

import asyncio
import logging
import time
import threading
import io
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone
import cv2
import numpy as np
from PIL import Image, ImageGrab
import hashlib

logger = logging.getLogger(__name__)

class LiveVisionController:
    def __init__(self, vlm_handler, vision_tool):
        self.vlm_handler = vlm_handler
        self.vision_tool = vision_tool
        
        # Configuration
        self.live_enabled = False
        self.fresh_secs = 5
        self.min_interval_ms = 500
        self.max_rate_per_sec = 1.5
        self.consent_required = False  # Simplified for local use
        self.consent_granted = True
        self.redaction_enabled = True
        self.store_raw_frames = False
        self.privacy_hotkey = "Ctrl+Shift+P"
        
        # State
        self.latest_summary = None
        self.captured_at_unix = None
        self.source = "screen"
        self.is_fresh = False
        self.age_secs = 0
        
        # Capture loop
        self.capture_task = None
        self.capture_running = False
        self.last_capture_time = 0
        self.frame_gate = None
        self.preprocessor = None
        
        # Event subscriptions
        self.window_change_callback = None
        self.clipboard_change_callback = None
        
        # Performance tracking
        self.summary_count = 0
        self.last_summary_time = 0
        
    async def start_live_vision(self) -> bool:
        """Start continuous vision capture"""
        try:
            if self.live_enabled:
                logger.warning("Live vision already running")
                return True
            
            if not self.consent_granted:
                logger.error("Consent not granted for live vision")
                return False
            
            self.live_enabled = True
            self.capture_running = True
            
            # Initialize components
            self.frame_gate = FrameGate(self.min_interval_ms)
            self.preprocessor = PreProcessor(redaction_enabled=self.redaction_enabled)
            
            # Start capture loop
            self.capture_task = asyncio.create_task(self._capture_loop())
            
            logger.info("Live vision started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start live vision: {e}")
            return False
    
    async def stop_live_vision(self) -> bool:
        """Stop continuous vision capture"""
        try:
            if not self.live_enabled:
                return True
            
            self.live_enabled = False
            self.capture_running = False
            
            if self.capture_task:
                self.capture_task.cancel()
                try:
                    await self.capture_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Live vision stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop live vision: {e}")
            return False
    
    async def privacy_pause(self) -> bool:
        """Immediate privacy pause - halt capture and clear buffers"""
        try:
            await self.stop_live_vision()
            self.latest_summary = None
            self.captured_at_unix = None
            self.is_fresh = False
            self.age_secs = 0
            
            logger.info("Privacy pause activated")
            return True
            
        except Exception as e:
            logger.error(f"Privacy pause failed: {e}")
            return False
    
    async def _capture_loop(self):
        """Main capture loop with intelligent gating"""
        try:
            while self.capture_running:
                current_time = time.time()
                
                # Rate limiting
                if current_time - self.last_summary_time < (1.0 / self.max_rate_per_sec):
                    await asyncio.sleep(0.1)
                    continue
                
                # Capture frame
                frame_data = await self._capture_frame()
                if not frame_data:
                    await asyncio.sleep(0.1)
                    continue
                
                # Frame gating
                should_process = await self.frame_gate.should_process_frame(frame_data)
                if not should_process:
                    await asyncio.sleep(0.1)
                    continue
                
                # Preprocess frame
                processed_frame = await self.preprocessor.process_frame(frame_data)
                if not processed_frame:
                    await asyncio.sleep(0.1)
                    continue
                
                # Generate VLM summary
                summary = await self.vlm_handler.analyze_image(processed_frame, self.source)
                if summary:
                    await self._update_summary(summary)
                
                self.last_summary_time = current_time
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                
        except asyncio.CancelledError:
            logger.info("Capture loop cancelled")
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
    
    async def _capture_frame(self) -> Optional[bytes]:
        """Capture a single frame"""
        try:
            if self.source == "screen":
                return await self._capture_screen()
            elif self.source == "camera":
                return await self._capture_camera()
            else:
                return None
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    async def _capture_screen(self) -> Optional[bytes]:
        """Fast screen capture"""
        try:
            # Use PIL ImageGrab for cross-platform capture
            screenshot = ImageGrab.grab()
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            screenshot.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    async def _capture_camera(self) -> Optional[bytes]:
        """Fast camera capture"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Camera capture failed: {e}")
            return None
    
    async def _update_summary(self, summary: Dict):
        """Update latest summary with freshness calculation"""
        self.latest_summary = summary
        self.captured_at_unix = time.time()
        self.age_secs = 0
        self.is_fresh = True
        self.summary_count += 1
        
        logger.debug(f"Updated summary #{self.summary_count}: {summary.get('high_level_summary', 'No summary')}")
    
    def get_latest_summary(self) -> Optional[Dict]:
        """Get latest summary with freshness check"""
        if not self.latest_summary or not self.captured_at_unix:
            return None
        
        # Update age and freshness
        self.age_secs = int(time.time() - self.captured_at_unix)
        self.is_fresh = self.age_secs <= self.fresh_secs
        
        return self.latest_summary
    
    def get_vision_context(self) -> Optional[str]:
        """Get vision context for LLM with staleness info"""
        summary = self.get_latest_summary()
        if not summary:
            return None
        
        context = f"VLM_SUMMARY_JSON:\n{summary}"
        
        if not self.is_fresh:
            context += f"\n\nWARNING: Vision context is stale (age: {self.age_secs}s, fresh threshold: {self.fresh_secs}s)"
        
        return context
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive live vision status"""
        return {
            "live_enabled": self.live_enabled,
            "capture_running": self.capture_running,
            "fresh_secs": self.fresh_secs,
            "min_interval_ms": self.min_interval_ms,
            "max_rate_per_sec": self.max_rate_per_sec,
            "redaction_enabled": self.redaction_enabled,
            "has_summary": self.latest_summary is not None,
            "captured_at_unix": self.captured_at_unix,
            "age_secs": self.age_secs,
            "is_fresh": self.is_fresh,
            "source": self.source,
            "summary_count": self.summary_count,
            "last_confidence": self.latest_summary.get("confidence", 0.0) if self.latest_summary else 0.0
        }
    
    def set_fresh_threshold(self, seconds: int):
        """Set freshness threshold"""
        self.fresh_secs = seconds
        logger.info(f"Fresh threshold set to {seconds} seconds")
    
    def set_capture_source(self, source: str):
        """Set capture source preference"""
        if source in ["screen", "camera"]:
            self.source = source
            logger.info(f"Capture source set to {source}")
        else:
            logger.warning(f"Invalid capture source: {source}")
    
    def toggle_redaction(self, enabled: bool):
        """Toggle redaction on/off"""
        self.redaction_enabled = enabled
        if self.preprocessor:
            self.preprocessor.redaction_enabled = enabled
        logger.info(f"Redaction {'enabled' if enabled else 'disabled'}")


class FrameGate:
    """Intelligent frame gating to reduce redundant processing"""
    
    def __init__(self, min_interval_ms: int):
        self.min_interval_ms = min_interval_ms
        self.last_processed_time = 0
        self.last_frame_hash = None
        self.last_histogram = None
        
    async def should_process_frame(self, frame_data: bytes) -> bool:
        """Determine if frame should be processed"""
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Enforce minimum interval
        if current_time - self.last_processed_time < self.min_interval_ms:
            return False
        
        # Calculate frame hash for change detection
        frame_hash = hashlib.md5(frame_data).hexdigest()
        
        # Skip if identical to last frame
        if frame_hash == self.last_frame_hash:
            return False
        
        # Update state
        self.last_processed_time = current_time
        self.last_frame_hash = frame_hash
        
        return True


class PreProcessor:
    """Frame preprocessing with redaction and optimization"""
    
    def __init__(self, redaction_enabled: bool = True):
        self.redaction_enabled = redaction_enabled
        self.target_size = (1280, 720)  # Downscale target
        
    async def process_frame(self, frame_data: bytes) -> Optional[bytes]:
        """Preprocess frame for VLM analysis"""
        try:
            # Load image
            img = Image.open(io.BytesIO(frame_data))
            
            # Downscale for performance
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Apply redaction if enabled
            if self.redaction_enabled:
                img = await self._apply_redaction(img)
            
            # Convert back to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            return img_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            return None
    
    async def _apply_redaction(self, img: Image.Image) -> Image.Image:
        """Apply redaction overlays for privacy"""
        try:
            # Convert to numpy for processing
            img_array = np.array(img)
            
            # Simple redaction: blur sensitive regions
            # This is a placeholder - in production, you'd use proper detection
            # For now, we'll just return the original image
            
            return img
            
        except Exception as e:
            logger.error(f"Redaction failed: {e}")
            return img
