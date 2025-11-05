"""
Singing Handler for Sakura Bot
Handles WAV file playback and LLM disabling during singing
"""

import asyncio
import os
import logging
from typing import Optional, List, Dict
import discord
from discord.ext import commands
import io
import wave
import struct

logger = logging.getLogger(__name__)

class SingingHandler:
    def __init__(self, bot, voice_handler):
        self.bot = bot
        self.voice_handler = voice_handler
        self.is_singing = False
        self.current_song = None
        self.singing_task = None
        self.songs_directory = "songs"  # Directory containing WAV files
        
        # Ensure songs directory exists
        os.makedirs(self.songs_directory, exist_ok=True)
        
    async def start_singing(self, song_filename: str) -> bool:
        """Start singing a specific song"""
        try:
            if self.is_singing:
                logger.warning("Already singing, stopping current song first")
                await self.stop_singing()
            
            # Check if song file exists
            song_path = os.path.join(self.songs_directory, song_filename)
            if not os.path.exists(song_path):
                logger.error(f"Song file not found: {song_path}")
                return False
            
            # Check if bot is in a voice channel
            # Get voice client from the bot's voice clients
            voice_client = None
            if hasattr(self.bot, 'voice_clients') and self.bot.voice_clients:
                voice_client = self.bot.voice_clients[0]
            
            if not voice_client:
                logger.error("Bot not in a voice channel")
                return False
            
            # Check voice connection health
            if not voice_client.is_connected():
                logger.error("Voice client is not connected")
                return False
            
            self.is_singing = True
            self.current_song = song_filename
            
            # Disable LLM during singing
            if hasattr(self.bot, 'chat_handler') and hasattr(self.bot.chat_handler, 'llm_wrapper'):
                self.bot.chat_handler.llm_wrapper.llm_state.enabled = False
                logger.info("LLM disabled for singing")
            
            # Also disable global LLM state
            if hasattr(self.bot, 'llm_enabled'):
                self.bot.llm_enabled = False
                logger.info("Global LLM state disabled for singing")
            
            # Start singing task
            self.singing_task = asyncio.create_task(self._play_song(song_path, voice_client))
            
            logger.info(f"Started singing: {song_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting singing: {e}")
            await self.stop_singing()
            return False
    
    async def stop_singing(self) -> bool:
        """Stop current singing"""
        try:
            if not self.is_singing:
                return True
            
            self.is_singing = False
            self.current_song = None
            
            # Cancel singing task if running
            if self.singing_task and not self.singing_task.done():
                self.singing_task.cancel()
                try:
                    await self.singing_task
                except asyncio.CancelledError:
                    pass
            
            # Re-enable LLM
            if hasattr(self.bot, 'chat_handler') and hasattr(self.bot.chat_handler, 'llm_wrapper'):
                self.bot.chat_handler.llm_wrapper.llm_state.enabled = True
                logger.info("LLM re-enabled after singing")
            else:
                logger.warning("Could not re-enable LLM: chat_handler or llm_wrapper not found")
            
            # Also re-enable global LLM state
            if hasattr(self.bot, 'llm_enabled'):
                self.bot.llm_enabled = True
                logger.info("Global LLM state re-enabled after singing")
            else:
                logger.warning("Could not re-enable global LLM state: llm_enabled attribute not found")
            
            # Force update the bot's AI status
            if hasattr(self.bot, 'set_llm_enabled'):
                self.bot.set_llm_enabled(True)
                logger.info("Bot LLM state force-enabled via set_llm_enabled")
            
            # Verify LLM is actually enabled
            llm_enabled_check = False
            if hasattr(self.bot, 'llm_enabled'):
                llm_enabled_check = self.bot.llm_enabled
            elif hasattr(self.bot, 'chat_handler') and hasattr(self.bot.chat_handler, 'llm_wrapper'):
                llm_enabled_check = self.bot.chat_handler.llm_wrapper.llm_state.enabled
            
            logger.info(f"LLM enabled status after re-enabling: {llm_enabled_check}")
            
            # Stop voice client if playing
            if hasattr(self.bot, 'voice_clients') and self.bot.voice_clients:
                voice_client = self.bot.voice_clients[0]
                if voice_client and voice_client.is_playing():
                    voice_client.stop()
            
            logger.info("Stopped singing")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping singing: {e}")
            return False
    
    async def _play_song(self, song_path: str, voice_client):
        """Play the song file in voice channel"""
        try:
            logger.info(f"Starting to play song: {song_path}")
            
            # Check if voice client is ready
            if not voice_client:
                logger.error("Voice client not available")
                await self.stop_singing()
                return
            
            if not voice_client.is_connected():
                logger.error("Voice client not connected")
                await self.stop_singing()
                return
            
            # Create audio source from file with Discord-compatible options
            # Use more stable audio processing to prevent pitch spikes
            audio_source = discord.FFmpegPCMAudio(
                song_path,
                before_options="-re -fflags +genpts",  # Read input at native frame rate with proper timestamps
                options="-f s16le -ar 48000 -ac 2 -bufsize 64k"  # Convert to Discord's preferred format with buffer control
            )
            
            logger.info("Audio source created, starting playback...")
            
            # Play the audio
            voice_client.play(audio_source)
            
            logger.info("Audio playback started")
            
            # Wait for playback to complete with connection monitoring
            while voice_client.is_playing() and self.is_singing:
                # Check if voice client is still connected
                if not voice_client.is_connected():
                    logger.error("Voice client disconnected during playback")
                    # Try to reconnect before giving up
                    if not await self._check_voice_connection():
                        logger.error("Failed to reconnect, stopping singing")
                        await self.stop_singing()
                        return
                
                # Check for connection health
                try:
                    # Small delay to prevent excessive checking
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error during playback monitoring: {e}")
                    await self.stop_singing()
                    return
            
            logger.info("Audio playback completed")
            
            # Auto-stop when song ends
            if self.is_singing:
                await self.stop_singing()
                
        except Exception as e:
            logger.error(f"Error playing song: {e}")
            import traceback
            traceback.print_exc()
            await self.stop_singing()
    
    def get_available_songs(self) -> List[Dict[str, str]]:
        """Get list of available songs"""
        try:
            songs = []
            if os.path.exists(self.songs_directory):
                for filename in os.listdir(self.songs_directory):
                    if filename.lower().endswith('.wav'):
                        # Extract song name from filename (remove extension)
                        name = os.path.splitext(filename)[0]
                        songs.append({
                            'filename': filename,
                            'name': name.replace('_', ' ').title()
                        })
            return songs
        except Exception as e:
            logger.error(f"Error getting available songs: {e}")
            return []
    
    def get_wav_duration(self, file_path: str) -> float:
        """Get duration of WAV file in seconds"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                return duration
        except Exception as e:
            logger.error(f"Error getting WAV duration: {e}")
            return 0.0
    
    def get_singing_status(self) -> Dict[str, any]:
        """Get current singing status"""
        duration = 0.0
        if self.current_song:
            song_path = os.path.join(self.songs_directory, self.current_song)
            duration = self.get_wav_duration(song_path)
        
        # Check both local and global LLM state
        llm_enabled = True
        if hasattr(self.bot, 'llm_enabled'):
            llm_enabled = self.bot.llm_enabled
        elif hasattr(self.bot, 'chat_handler') and hasattr(self.bot.chat_handler, 'llm_wrapper'):
            llm_enabled = self.bot.chat_handler.llm_wrapper.llm_state.enabled
        
        # Check voice connection health
        voice_connected = False
        if hasattr(self.bot, 'voice_clients') and self.bot.voice_clients:
            voice_client = self.bot.voice_clients[0]
            voice_connected = voice_client and voice_client.is_connected()
        
        return {
            'is_singing': self.is_singing,
            'current_song': self.current_song,
            'duration': duration,
            'llm_enabled': not self.is_singing and llm_enabled,
            'voice_connected': voice_connected
        }
    
    async def _check_voice_connection(self) -> bool:
        """Check and attempt to recover voice connection if needed"""
        try:
            if not hasattr(self.bot, 'voice_clients') or not self.bot.voice_clients:
                return False
            
            voice_client = self.bot.voice_clients[0]
            if not voice_client:
                return False
            
            # Check if connected
            if voice_client.is_connected():
                return True
            
            # Attempt to reconnect if disconnected
            logger.warning("Voice client disconnected, attempting to reconnect...")
            try:
                # Get the channel from the voice client
                channel = voice_client.channel
                if channel:
                    await voice_client.connect(reconnect=True, timeout=10.0)
                    logger.info("Voice client reconnected successfully")
                    return True
            except Exception as e:
                logger.error(f"Failed to reconnect voice client: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking voice connection: {e}")
            return False
