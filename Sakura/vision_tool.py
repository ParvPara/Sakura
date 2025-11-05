"""
Vision Tool for Sakura Bot
Handles image capture (screen/camera) and VLM integration
"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Tuple, List
import io
from PIL import Image, ImageGrab
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class VisionTool:
    def __init__(self, vlm_handler):
        self.vlm_handler = vlm_handler
        self.enabled = False
        self.stale_after_secs = 120
        self.capture_source_preference = "screen"  # "screen" | "camera" | "auto"
        
    async def capture_screen(self, monitor_index: int = None) -> Optional[bytes]:
        """Capture screen screenshot"""
        try:
            # Use PIL ImageGrab for cross-platform screen capture
            if monitor_index is not None:
                # Capture specific monitor (0 = primary, 1 = secondary, etc.)
                screenshot = ImageGrab.grab(bbox=None, all_screens=False, xdisplay=None)
            else:
                # Capture primary monitor (default behavior)
                screenshot = ImageGrab.grab()
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            screenshot.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            logger.info(f"Screen capture successful (monitor: {monitor_index or 'primary'})")
            return img_data
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    async def capture_camera(self) -> Optional[bytes]:
        """Capture from camera"""
        try:
            # Try to open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Camera not available")
                return None
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Failed to capture from camera")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            logger.info("Camera capture successful")
            return img_data
            
        except Exception as e:
            logger.error(f"Camera capture failed: {e}")
            return None
    
    async def capture_image(self, source: str = None, monitor: str = "0") -> Tuple[bool, str, Optional[bytes]]:
        """Capture image from specified source or preference"""
        if source is None:
            source = self.capture_source_preference
        
        if source == "screen":
            # Convert monitor string to int for screen capture
            monitor_index = int(monitor) if monitor.isdigit() else None
            img_data = await self.capture_screen(monitor_index)
            if img_data:
                return True, f"Screen capture successful (Monitor {monitor})", img_data
            else:
                return False, "Screen capture failed", None
                
        elif source == "camera":
            img_data = await self.capture_camera()
            if img_data:
                return True, "Camera capture successful", img_data
            else:
                return False, "Camera capture failed", None
                
        else:
            return False, f"Unknown capture source: {source}", None
    
    async def process_vision_request(self, source: str = None, monitor: str = "0") -> Dict:
        """Process a vision request - capture and analyze"""
        try:
            if not self.enabled:
                return {
                    "success": False,
                    "error": "Vision tool is disabled",
                    "summary": None
                }
            
            # Capture image with monitor selection
            success, message, img_data = await self.capture_image(source, monitor)
            if not success:
                return {
                    "success": False,
                    "error": message,
                    "summary": None
                }
            
            # Analyze with VLM
            summary = await self.vlm_handler.analyze_image(img_data, source or self.capture_source_preference)
            
            return {
                "success": True,
                "message": message,
                "summary": summary,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Vision request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": None
            }
    
    def should_trigger_vision(self, message: str) -> bool:
        """Check if message should trigger vision capture"""
        if not self.enabled:
            return False
        
        # Vision trigger keywords
        vision_keywords = [
            "see", "look", "what's on my screen", "screenshot",
            "camera", "photo", "image", "what do you see"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in vision_keywords)
    
    def get_vision_context(self) -> Optional[str]:
        """Get vision context for LLM if available and not stale"""
        summary = self.vlm_handler.get_last_summary()
        if not summary:
            return None
        
        if self.vlm_handler.is_summary_stale(self.stale_after_secs):
            return None
        
        # Format summary for LLM context
        return f"VLM_SUMMARY_JSON:\n{summary}"
    
    def get_vision_context_with_staleness(self) -> Tuple[Optional[str], bool]:
        """Get vision context and staleness status"""
        summary = self.vlm_handler.get_last_summary()
        logger.info(f"Vision tool get_last_summary: {summary is not None}")
        if not summary:
            logger.info("No vision summary available - need to capture image first")
            return None, False
        
        is_stale = self.vlm_handler.is_summary_stale(self.stale_after_secs)
        context = f"VLM_SUMMARY_JSON:\n{summary}"
        logger.info(f"Vision context created, is_stale: {is_stale}")
        
        return context, is_stale
    
    def set_enabled(self, enabled: bool):
        """Enable/disable vision tool"""
        self.enabled = enabled
        logger.info(f"Vision tool {'enabled' if enabled else 'disabled'}")
    
    def set_stale_threshold(self, seconds: int):
        """Set staleness threshold"""
        self.stale_after_secs = seconds
        logger.info(f"Vision staleness threshold set to {seconds} seconds")
    
    def set_capture_preference(self, preference: str):
        """Set capture source preference"""
        if preference in ["screen", "camera", "auto"]:
            self.capture_source_preference = preference
            logger.info(f"Capture preference set to {preference}")
        else:
            logger.warning(f"Invalid capture preference: {preference}")
    
    def get_available_monitors(self) -> List[Dict]:
        """Get list of available monitors"""
        try:
            monitors = []
            
            # Try to get monitor information using PIL
            try:
                # Get all screens using ImageGrab
                all_screens = ImageGrab.grab(bbox=None, all_screens=True)
                if all_screens:
                    # For now, we'll return basic monitor info
                    # In a more advanced implementation, you could use platform-specific APIs
                    monitors.append({
                        "id": 0,
                        "name": "Primary Monitor",
                        "width": all_screens.width,
                        "height": all_screens.height,
                        "is_primary": True
                    })
            except Exception as e:
                logger.warning(f"Could not detect monitors via PIL: {e}")
            
            # Try to detect additional monitors using platform-specific methods
            try:
                import platform
                system = platform.system().lower()
                
                if system == "windows":
                    # Try multiple methods for Windows monitor detection
                    try:
                        # Method 1: Try win32api if available
                        try:
                            import win32api
                            import win32con
                            
                            def get_monitor_friendly_name(device_name):
                                try:
                                    device_key = device_name.replace('\\\\.\\', '')
                                    device_num = 0
                                    
                                    while True:
                                        try:
                                            device = win32api.EnumDisplayDevices(device_name, device_num, 0)
                                            if device.DeviceString:
                                                return device.DeviceString
                                            device_num += 1
                                        except:
                                            break
                                    
                                    adapter = win32api.EnumDisplayDevices(None, int(device_key.replace('DISPLAY', '')) - 1, 0)
                                    if adapter.DeviceString:
                                        return adapter.DeviceString
                                    
                                    return None
                                except Exception as e:
                                    return None
                            
                            monitors_list = []
                            monitor_enum = win32api.EnumDisplayMonitors()
                            
                            for i, (hMonitor, hdcMonitor, rect) in enumerate(monitor_enum):
                                try:
                                    monitor_info = win32api.GetMonitorInfo(hMonitor)
                                    device_name = monitor_info.get('Device', '')
                                    width = rect[2] - rect[0]
                                    height = rect[3] - rect[1]
                                    is_primary = monitor_info['Flags'] & win32con.MONITORINFOF_PRIMARY != 0
                                    
                                    friendly_name = get_monitor_friendly_name(device_name)
                                    
                                    if friendly_name:
                                        if is_primary:
                                            display_name = f"{friendly_name} (Primary) [{width}x{height}]"
                                        else:
                                            display_name = f"{friendly_name} [{width}x{height}]"
                                    else:
                                        if is_primary:
                                            display_name = f"Primary Display [{width}x{height}]"
                                        else:
                                            display_name = f"Display {i+1} [{width}x{height}]"
                                    
                                    monitors_list.append({
                                        "id": i,
                                        "name": display_name,
                                        "width": width,
                                        "height": height,
                                        "is_primary": is_primary,
                                        "device": device_name
                                        })
                                except Exception as e:
                                    logger.warning(f"Error getting monitor {i} info: {e}")
                            
                            if monitors_list:
                                monitors = monitors_list
                                monitor_names = [m['name'] for m in monitors]
                                logger.info(f"Detected {len(monitors)} monitors via win32api: {monitor_names}")
                                
                        except ImportError:
                            logger.info("win32api not available, trying alternative methods")
                            
                            # Method 2: Use tkinter to detect screen info
                            try:
                                import tkinter as tk
                                root = tk.Tk()
                                
                                # Get primary screen info
                                primary_width = root.winfo_screenwidth()
                                primary_height = root.winfo_screenheight()
                                
                                monitors = [{
                                    "id": 0,
                                    "name": "Primary Display",
                                    "width": primary_width,
                                    "height": primary_height,
                                    "is_primary": True
                                }]
                                
                                # Try to detect additional monitors using tkinter
                                # This is a simplified approach - tkinter doesn't directly support multi-monitor
                                # but we can make educated guesses based on common setups
                                
                                # Check if we have a high-resolution display (likely multiple monitors)
                                if primary_width > 1920 or primary_height > 1080:
                                    # Likely has multiple monitors, add a second one
                                    monitors.append({
                                        "id": 1,
                                        "name": "Secondary Display",
                                        "width": 1920,  # Common secondary resolution
                                        "height": 1080,
                                        "is_primary": False
                                    })
                                
                                root.destroy()
                                logger.info(f"Detected {len(monitors)} monitors via tkinter")
                                
                            except Exception as e:
                                logger.warning(f"Tkinter monitor detection failed: {e}")
                                
                            # Method 3: Use PowerShell to detect monitors (most reliable)
                            try:
                                import subprocess
                                import json
                                
                                # PowerShell script to get monitor information
                                ps_script = """
                                Get-WmiObject -Class Win32_DesktopMonitor | ForEach-Object {
                                    $monitor = $_
                                    $adapter = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -ne $null }
                                    [PSCustomObject]@{
                                        Name = $monitor.Name
                                        Width = $monitor.ScreenWidth
                                        Height = $monitor.ScreenHeight
                                        Primary = $monitor.Availability -eq 3
                                    }
                                } | ConvertTo-Json
                                """
                                
                                result = subprocess.run([
                                    'powershell', '-Command', ps_script
                                ], capture_output=True, text=True, timeout=10)
                                
                                if result.returncode == 0 and result.stdout.strip():
                                    try:
                                        ps_monitors = json.loads(result.stdout)
                                        if isinstance(ps_monitors, list):
                                            for i, monitor in enumerate(ps_monitors):
                                                if monitor.get('Width') and monitor.get('Height'):
                                                    monitors.append({
                                                        "id": i,
                                                        "name": monitor.get('Name', f'Monitor {i}'),
                                                        "width": monitor.get('Width'),
                                                        "height": monitor.get('Height'),
                                                        "is_primary": monitor.get('Primary', i == 0)
                                                    })
                                            
                                            if len(monitors) > 1:
                                                logger.info(f"Detected {len(monitors)} monitors via PowerShell")
                                    except json.JSONDecodeError:
                                        logger.warning("Failed to parse PowerShell monitor output")
                                        
                            except Exception as e:
                                logger.warning(f"PowerShell monitor detection failed: {e}")
                                
                            # Method 4: Use WMI if available
                            try:
                                import wmi
                                c = wmi.WMI()
                                
                                # Get display adapters
                                for adapter in c.Win32_VideoController():
                                    if adapter.Name and adapter.CurrentHorizontalResolution and adapter.CurrentVerticalResolution:
                                        monitors.append({
                                            "id": len(monitors),
                                            "name": adapter.Name or f'Display {len(monitors)}',
                                            "width": adapter.CurrentHorizontalResolution,
                                            "height": adapter.CurrentVerticalResolution,
                                            "is_primary": len(monitors) == 0
                                        })
                                
                                if len(monitors) > 1:
                                    logger.info(f"Detected {len(monitors)} monitors via WMI")
                                    
                            except ImportError:
                                logger.info("WMI not available")
                            except Exception as e:
                                logger.warning(f"WMI monitor detection failed: {e}")
                                
                    except Exception as e:
                        logger.warning(f"Windows monitor detection failed: {e}")
                
                elif system == "darwin":  # macOS
                    try:
                        # Use system_profiler to get display info
                        import subprocess
                        result = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-json'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            import json
                            data = json.loads(result.stdout)
                            # Parse display information
                            # This is a simplified version - you'd need to parse the actual JSON structure
                            pass
                    except Exception as e:
                        logger.warning(f"macOS monitor detection failed: {e}")
                
                elif system == "linux":
                    try:
                        # Use xrandr to get display info
                        import subprocess
                        result = subprocess.run(['xrandr', '--query'], capture_output=True, text=True)
                        if result.returncode == 0:
                            lines = result.stdout.split('\n')
                            monitor_id = 0
                            for line in lines:
                                if ' connected' in line and 'primary' in line:
                                    # Parse primary monitor
                                    parts = line.split()
                                    if len(parts) >= 3:
                                        resolution = parts[2].split('+')[0]
                                        if 'x' in resolution:
                                            width, height = resolution.split('x')
                                            monitors.append({
                                                "id": monitor_id,
                                                "name": f"Monitor {monitor_id}",
                                                "width": int(width),
                                                "height": int(height),
                                                "is_primary": True
                                            })
                                            monitor_id += 1
                    except Exception as e:
                        logger.warning(f"Linux monitor detection failed: {e}")
                        
            except Exception as e:
                logger.warning(f"Platform-specific monitor detection failed: {e}")
            
            # Fallback: if no monitors detected, provide default
            if not monitors:
                monitors = [{
                    "id": 0,
                    "name": "Primary Monitor",
                    "width": 1920,
                    "height": 1080,
                    "is_primary": True
                }]
            
            logger.info(f"Detected {len(monitors)} monitors: {[m['name'] for m in monitors]}")
            return monitors
            
        except Exception as e:
            logger.error(f"Error detecting monitors: {e}")
            # Return default monitor as fallback
            return [{
                "id": 0,
                "name": "Primary Monitor",
                "width": 1920,
                "height": 1080,
                "is_primary": True
            }]

    def get_status(self) -> Dict:
        """Get vision tool status"""
        vlm_status = self.vlm_handler.get_status()
        return {
            "enabled": self.enabled,
            "stale_after_secs": self.stale_after_secs,
            "capture_preference": self.capture_source_preference,
            "vlm_status": vlm_status,
            "available_monitors": self.get_available_monitors()
        }
    
