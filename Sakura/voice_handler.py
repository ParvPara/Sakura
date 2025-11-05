import discord
from discord.ext import commands
from discord.ext import voice_recv
import asyncio
import webrtcvad
import time
import collections
import numpy as np
from utils.audio_utils import prepare_audio_for_whisper
import config
import io

class WhisperSink(voice_recv.AudioSink):
    def __init__(self, bot, text_channel: discord.TextChannel):
        super().__init__()
        self.bot = bot
        self.text_channel = text_channel
        
        # Buffer configuration
        self.buffer_duration = 20.0
        self.sample_rate = 16000
        self.buffer_max_samples = int(self.buffer_duration * self.sample_rate)
        
        # Low-latency but safe endpointing
        self.silence_threshold = 0.35  # seconds of silence before flush (tweak 0.30–0.45)
        
        # WebRTC VAD (20 ms frames recommended; mode 2 is stable, 3 is very aggressive)
        self.vad = webrtcvad.Vad(2)
        self.vad_frame_ms = 20
        self.vad_frame_samples = int(self.sample_rate * self.vad_frame_ms / 1000)  # 320
        self.min_audio_length = 0.55
        self.min_samples_needed = int(self.min_audio_length * self.sample_rate)
        
        # State
        self.buffers = {}         # uid -> np.float32 (rolling 20s)
        self.vad_buffers = {}     # uid -> bytes/int16 deque for quick frame slicing
        self.last_voice_time = {} # uid -> time.time()
        self.voice_started_time = {}
        self.is_speaking = {}
        self.users = {}           # uid -> discord.Member for watcher-triggered flush
        
        # VAD smoothing
        self.speech_run_ms = collections.defaultdict(int)
        self.silence_run_ms = collections.defaultdict(int)
        self.noise_gate = 0.01  # mean abs amplitude gate to ignore ultra-low "speech"
        
        self.flush_lock = asyncio.Lock()
        self.packet_count = 0
        self.tts_playing = False
        
        # Audio accumulation for overlapping input
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        self.accumulated_audio = {}
        self.last_audio_packet_time = {}  # Track last audio packet time for 2.5s flush timer
        
        # Multi-user coordination
        self.last_response_time = 0
        self.min_response_interval = 3.0  # Minimum 3 seconds between responses
        self.processing_user = None  # Track which user is currently being processed
        
        # Start a background watcher to trigger flush even when no packets arrive
        self._watcher_task = self.bot.loop.create_task(self._silence_watcher())
        
        print(f"[DEBUG] WhisperSink created for channel {text_channel.name} - VAD smoothing enabled, 20s rolling buffer with watcher")

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData):
        self.packet_count += 1
        
        if self.packet_count <= 10:
            print(f"[PACKET-RAW] #{self.packet_count} - user={user.name if user else 'None'}, pcm={len(data.pcm) if data and data.pcm else 'None'} bytes, bot_id={self.bot.user.id if self.bot and self.bot.user else 'None'}")
        elif self.packet_count % 25 == 0:
            print(f"[PACKET-RAW] #{self.packet_count} received")
        
        if user is None or data.pcm is None or user.id == self.bot.user.id:
            if self.packet_count <= 10:
                print(f"[PACKET-SKIP] #{self.packet_count} - Early return triggered")
            return
        
        # Skip processing if TTS is currently playing (more comprehensive check)
        if self.tts_playing or self._is_any_tts_playing():
            return
        
        # If we're currently processing audio, accumulate it instead of processing immediately
        if self.is_processing:
            uid = user.id
            now = time.time()
            if uid not in self.accumulated_audio:
                self.accumulated_audio[uid] = np.array([], dtype=np.float32)
            
            # Update last audio packet time even when accumulating
            self.last_audio_packet_time[uid] = now
            
            try:
                mono16 = self._stereo48k_to_mono16k(data.pcm)
                self.accumulated_audio[uid] = np.concatenate([self.accumulated_audio[uid], mono16])
                print(f"[DEBUG] Accumulating audio for {user.name} while processing (total: {self.accumulated_audio[uid].size} samples)")
            except Exception as e:
                print(f"[DEBUG] Error accumulating audio from {user.name}: {e}")
            return
        
        if len(data.pcm) < 4:
            return
        
        if len(data.pcm) % 2 != 0:
            return
        
        if len(data.pcm) > 4096:
            return
        
        try:
            mono16 = self._stereo48k_to_mono16k(data.pcm)
        except Exception as e:
            print(f"[DEBUG] Error processing audio from {user.name if user else 'Unknown'}: {e}")
            return
            
        uid = user.id
        now = time.time()
        self.users[uid] = user
        
        # Update last audio packet time for 2.5s flush timer
        self.last_audio_packet_time[uid] = now
        
        # init
        if uid not in self.buffers:
            self.buffers[uid] = np.array([], dtype=np.float32)
            self.vad_buffers[uid] = bytearray()
            self.last_voice_time[uid] = now
            self.voice_started_time[uid] = 0.0
            self.is_speaking[uid] = False
            self.speech_run_ms[uid] = 0
            self.silence_run_ms[uid] = 0
            print(f"[DEBUG] Initialized buffers for {user.name} (Discord username, 20s window, VAD smoothing)")
        
        # append float32 audio to 20s rolling window
        buf = self.buffers[uid]
        self.buffers[uid] = np.concatenate([buf, mono16]) if buf.size else mono16
        if self.buffers[uid].size > self.buffer_max_samples:
            excess = self.buffers[uid].size - self.buffer_max_samples
            self.buffers[uid] = self.buffers[uid][excess:]
        
        # append int16 bytes for VAD processing
        pcm16 = (np.clip(mono16, -1.0, 1.0) * 32768.0).astype(np.int16)
        self.vad_buffers[uid].extend(pcm16.tobytes())
        # keep VAD buffer roughly in sync with 20s (bytes == samples*2)
        max_bytes = self.buffer_max_samples * 2
        if len(self.vad_buffers[uid]) > max_bytes:
            # drop oldest bytes
            drop = len(self.vad_buffers[uid]) - max_bytes
            del self.vad_buffers[uid][:drop]
        
        # --- VAD smoothing over incoming 20ms subframes ---
        frame_bytes = self.vad_frame_samples * 2
        chunk = self.vad_buffers[uid]
        # Only analyze the new tail we just appended (iterate over last packet length)
        new_bytes = pcm16.nbytes
        start_index = max(0, len(chunk) - new_bytes)
        for i in range(start_index, len(chunk), frame_bytes):
            frame = chunk[i:i+frame_bytes]
            if len(frame) < frame_bytes:
                break
            # amplitude gate (ignore near-silence even if VAD says speech)
            amp = np.mean(np.abs(np.frombuffer(frame, dtype=np.int16))) / 32768.0
            vad_speech = False
            try:
                vad_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                pass
            is_speech = vad_speech and (amp > self.noise_gate)
            
            if is_speech:
                if not self.is_speaking[uid]:
                    self.voice_started_time[uid] = now
                    self.is_speaking[uid] = True
                    # reset runs
                    self.silence_run_ms[uid] = 0
                    self.speech_run_ms[uid] = 0
                    print(f"[DEBUG] Voice started: {user.name}")
                self.last_voice_time[uid] = now
                self.speech_run_ms[uid] += self.vad_frame_ms
                self.silence_run_ms[uid] = 0
            else:
                self.silence_run_ms[uid] += self.vad_frame_ms
                # don't flush here; the watcher will handle silence cutoff

    async def _flush(self, user, audio: np.ndarray):
        if audio.size == 0:
            print(f"[DEBUG] Skipping empty audio from {user.name}")
            return
        
        # Check if voice processing is enabled
        if not self.bot.voice_enabled:
            print("[DEBUG] Voice processing disabled, skipping audio processing")
            return
        
        # Multi-user coordination: prevent rapid-fire responses
        current_time = time.time()
        if current_time - self.last_response_time < self.min_response_interval:
            print(f"[DEBUG] Cooldown active - {self.min_response_interval - (current_time - self.last_response_time):.1f}s remaining")
            return
        
        # If another user is currently being processed, skip this one
        if self.processing_user and self.processing_user != user.id:
            print(f"[DEBUG] Another user ({self.processing_user}) is being processed, skipping {user.name}")
            return
        
        # Check if we have accumulated audio for this user
        uid = user.id
        if uid in self.accumulated_audio and self.accumulated_audio[uid].size > 0:
            # Combine current audio with accumulated audio
            combined_audio = np.concatenate([audio, self.accumulated_audio[uid]])
            print(f"[DEBUG] Combining current audio ({audio.size} samples) with accumulated audio ({self.accumulated_audio[uid].size} samples) for {user.name}")
            audio = combined_audio
            # Clear accumulated audio
            self.accumulated_audio[uid] = np.array([], dtype=np.float32)
        
        audio_duration = audio.size / self.sample_rate
        print(f"[DEBUG] Processing audio from {user.name}: {audio.size} samples ({audio_duration:.2f}s)")
            
        # Use lock to prevent overlapping processing but allow concurrent listening
        async with self.flush_lock:
            # Set processing flag to accumulate new audio
            self.is_processing = True
            self.processing_user = user.id  # Track which user we're processing
            try:
                # Start timing for latency measurement
                start_time = time.time()
                
                processed_audio = prepare_audio_for_whisper(audio)
                
                whisper_start = time.time()
                result = await self.bot.whisper_handler.transcribe_audio(processed_audio)
                whisper_time = time.time() - whisper_start
                
                print(f"[DEBUG] Whisper result: '{result['text']}' (confidence: {result['confidence']}, took {whisper_time:.2f}s)")
                
                if result["text"].strip() and result['confidence'] > 0.3:  # Filter low confidence
                        # Check if STT is enabled
                        if not self.bot.stt_enabled:
                            print("[DEBUG] STT disabled, skipping voice processing")
                            return
                        
                        # Resolve Discord username to real name from memory
                        discord_username = user.name
                        print(f"[DEBUG] Using Discord username: {discord_username} (not display name: {getattr(user, 'display_name', 'N/A')})")
                        real_name = self._resolve_username_to_real_name(discord_username)
                        
                        chat_start = time.time()
                        
                        # Check if LLM is enabled
                        if not self.bot.llm_enabled:
                            print("[DEBUG] LLM disabled, skipping response generation")
                            return
                        
                        # Use the new agentic system for voice messages
                        print(f"[DEBUG] Processing voice message through agentic system: '{result['text']}'")
                        
                        # Get user ID from voice channel
                        user_id = None
                        voice_channel = self.bot.voice_clients[0].channel if self.bot.voice_clients else None
                        
                        if voice_channel:
                            for member in voice_channel.members:
                                if member.name == user.name:
                                    user_id = member.id
                                    break
                            
                            # If not found by name, try by display name
                            if not user_id:
                                for member in voice_channel.members:
                                    if member.display_name == user.display_name:
                                        user_id = member.id
                                        break
                        
                        # Fallback: search all guild members if still not found
                        if not user_id:
                            for member in self.text_channel.guild.members:
                                if (member.name == user.name or member.display_name == user.display_name) and member.voice:
                                    user_id = member.id
                                    break
                        
                        if user_id:
                            discord_member = None
                            for member in self.text_channel.guild.members:
                                if member.id == user_id:
                                    discord_member = member
                                    break
                            
                            if discord_member:
                                await self.bot.handle_voice_message(discord_member, result["text"])
                            else:
                                print(f"[DEBUG] Could not find Discord member object for user_id {user_id}")
                            return  # Exit early since handle_voice_message handles everything
                        else:
                            print(f"[DEBUG] Could not identify user {real_name} in voice channel")
                            # Don't continue with old chat system - just log the issue
                            print(f"[DEBUG] Skipping voice message processing due to user identification failure")
                            return
                        
                else:
                    if result["text"].strip():
                        print(f"[DEBUG] Low confidence transcription ({result['confidence']:.2f}), skipping")
                    else:
                        print(f"[DEBUG] Empty transcription, skipping response")
                                
            except Exception as e:
                print(f"[DEBUG] Error in audio processing: {e}")
                import traceback
                traceback.print_exc()
                if hasattr(self.bot, 'chat_handler') and hasattr(self.bot.chat_handler, 'current_message'):
                    self.bot.chat_handler.current_message = {"status": "error", "content": str(e)}
            finally:
                # Clear processing flag to allow new audio processing
                self.is_processing = False
                self.processing_user = None
                self.last_response_time = time.time()  # Update response time for cooldown

    async def _flush_streaming(self, user, audio: np.ndarray):
        """Enhanced flush method with streaming TTS support for ultra-low latency"""
        if audio.size == 0:
            print(f"[DEBUG] Skipping empty audio from {user.name}")
            return
        
        # Check if voice processing is enabled
        if not self.bot.voice_enabled:
            print("[DEBUG] Voice processing disabled, skipping audio processing")
            return
        
        # Check if we have accumulated audio for this user
        uid = user.id
        if uid in self.accumulated_audio and self.accumulated_audio[uid].size > 0:
            # Combine current audio with accumulated audio
            combined_audio = np.concatenate([audio, self.accumulated_audio[uid]])
            print(f"[DEBUG] Combining current audio ({audio.size} samples) with accumulated audio ({self.accumulated_audio[uid].size} samples) for {user.name}")
            audio = combined_audio
            # Clear accumulated audio
            self.accumulated_audio[uid] = np.array([], dtype=np.float32)
        
        audio_duration = audio.size / self.sample_rate
        print(f"[DEBUG] Processing audio from {user.name}: {audio.size} samples ({audio_duration:.2f}s)")
            
        async with self.flush_lock:
            self.is_processing = True
            try:
                # Start timing for latency measurement
                start_time = time.time()
                
                processed_audio = prepare_audio_for_whisper(audio)
                
                whisper_start = time.time()
                result = await self.bot.whisper_handler.transcribe_audio(processed_audio)
                whisper_time = time.time() - whisper_start
                
                print(f"[DEBUG] Whisper result: '{result['text']}' (confidence: {result['confidence']}, took {whisper_time:.2f}s)")
                
                if result["text"].strip() and result['confidence'] > 0.3:  # Filter low confidence
                        # Check if STT is enabled
                        if not self.bot.stt_enabled:
                            print("[DEBUG] STT disabled, skipping voice processing")
                            return
                        
                        # Resolve Discord username to real name from memory
                        discord_username = user.name
                        print(f"[DEBUG] Using Discord username: {discord_username} (not display name: {getattr(user, 'display_name', 'N/A')})")
                        real_name = self._resolve_username_to_real_name(discord_username)
                        
                        chat_start = time.time()
                        
                        # Check if LLM is enabled
                        if not self.bot.llm_enabled:
                            print("[DEBUG] LLM disabled, skipping response generation")
                            return
                        
                        # Use real-time streaming for ultra-low latency
                        response_chunks = []
                        accumulated_response = ""
                        tts_started = False
                        
                        async for chunk in self.bot.chat_handler.get_response_streaming(
                            self.text_channel.guild.id,
                            self.text_channel.id,
                            real_name,
                            result["text"]
                        ):
                            response_chunks.append(chunk)
                            accumulated_response += chunk
                            print(f"[DEBUG] LLM chunk: '{chunk}'")
                            
                            # Start TTS immediately for each meaningful chunk
                            if len(chunk.strip()) > 3 and not tts_started:
                                tts_started = True
                                
                                # Check if TTS is enabled
                                if not self.bot.tts_enabled:
                                    print("[DEBUG] TTS disabled, skipping audio generation")
                                    continue
                                
                                # Start TTS immediately for this chunk
                                tts_start = time.time()
                                tts_chunks = await self.bot.tts_handler.generate_speech_chunked(chunk, chunk_size=50)
                                
                                if tts_chunks:
                                    audio_queue = await self.bot.tts_handler.generate_speech_streaming(tts_chunks)
                                    tts_time = time.time() - tts_start
                                    print(f"[DEBUG] TTS chunk prepared: {len(tts_chunks)} (took {tts_time:.2f}s)")
                                    
                                    voice_client = self.text_channel.guild.voice_client
                                    if voice_client and voice_client.is_connected():
                                        # Update status to TTS speaking
                                        if hasattr(self.bot, 'chat_handler') and self.bot.chat_handler:
                                            self.bot.chat_handler.current_message = {"status": "tts_speaking", "content": accumulated_response}
                                        
                                        # Start playing immediately and continue processing
                                        asyncio.create_task(self._play_streaming_tts(voice_client, audio_queue))
                                        print(f"[DEBUG] Started real-time TTS playback")
                                    else:
                                        print(f"[DEBUG] Cannot play streaming TTS: voice_client={bool(voice_client)}, connected={voice_client.is_connected() if voice_client else False}")
                                
                                # Continue processing remaining chunks while TTS plays
                                continue
                        
                        chat_time = time.time() - chat_start
                        full_response = ''.join(response_chunks)
                        print(f"[DEBUG] Full Ollama response: '{full_response}' (took {chat_time:.2f}s)")
                        
                        # If we didn't start TTS with any chunk, start it now with the full response
                        if not tts_started and full_response:
                            # Check if TTS is enabled
                            if not self.bot.tts_enabled:
                                print("[DEBUG] TTS disabled, skipping audio generation")
                                return
                            
                            # Use streaming TTS for ultra-low latency
                            tts_start = time.time()
                            
                            # Split response into TTS-friendly chunks
                            tts_chunks = await self.bot.tts_handler.generate_speech_chunked(full_response, chunk_size=50)
                            
                            if tts_chunks:
                                # Generate streaming audio
                                audio_queue = await self.bot.tts_handler.generate_speech_streaming(tts_chunks)
                                
                                tts_time = time.time() - tts_start
                                print(f"[DEBUG] TTS chunks prepared: {len(tts_chunks)} (took {tts_time:.2f}s)")
                                
                                voice_client = self.text_channel.guild.voice_client
                                if voice_client and voice_client.is_connected():
                                    # Play streaming TTS
                                    await self._play_streaming_tts(voice_client, audio_queue)
                                else:
                                    print(f"[DEBUG] Cannot play streaming TTS: voice_client={bool(voice_client)}, connected={voice_client.is_connected() if voice_client else False}")
                            else:
                                print("[DEBUG] No TTS chunks generated")
                        
                        total_time = time.time() - start_time
                        tts_time = locals().get('tts_time', 0.0)  # Default to 0 if not defined
                        print(f"[DEBUG] Total processing time: {total_time:.2f}s (Whisper: {whisper_time:.2f}s, Chat: {chat_time:.2f}s, TTS: {tts_time:.2f}s)")
                else:
                    if result["text"].strip():
                        print(f"[DEBUG] Low confidence transcription ({result['confidence']:.2f}), skipping")
                    else:
                        print(f"[DEBUG] Empty transcription, skipping response")
                                
            except Exception as e:
                print(f"[DEBUG] Error in streaming audio processing: {e}")
                import traceback
                traceback.print_exc()
                if hasattr(self.bot, 'chat_handler') and hasattr(self.bot.chat_handler, 'current_message'):
                    self.bot.chat_handler.current_message = {"status": "error", "content": str(e)}
            finally:
                # Clear processing flag to allow new audio processing
                self.is_processing = False
                self.processing_user = None
                self.last_response_time = time.time()  # Update response time for cooldown

    async def _play_streaming_tts(self, voice_client, audio_queue):
        """Play streaming TTS audio for ultra-low latency"""
        try:
            # Check if already playing TTS
            if self.tts_playing:
                print("[DEBUG] TTS already playing, skipping new request")
                return
            
            # Set TTS playing flag
            self.tts_playing = True
            print("[DEBUG] Streaming TTS started - audio processing paused")
            
            # Check if voice client is already playing
            if voice_client.is_playing():
                print("[DEBUG] Voice client already playing, stopping current audio")
                voice_client.stop()
            
            # Process audio chunks as they arrive
            chunk_count = 0
            while True:
                try:
                    # Get next audio chunk with timeout
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
                    
                    if audio_data is None:  # End of stream
                        print(f"[DEBUG] Streaming TTS completed after {chunk_count} chunks")
                        break
                    
                    chunk_count += 1
                    print(f"[DEBUG] Playing TTS chunk {chunk_count}: {len(audio_data)} bytes")
                    
                    # Create audio source for this chunk
                    audio_buffer = io.BytesIO(audio_data)
                    audio_buffer.seek(0)
                    
                    try:
                        # Try to play WAV audio directly
                        audio_source = discord.FFmpegPCMAudio(
                            audio_buffer,
                            pipe=True
                        )
                    except discord.errors.ClientException:
                            import os
                            possible_paths = [
                                "D:\\ffmpeg\\bin\\ffmpeg.exe",
                            ]
                            ffmpeg_path = None
                            for path in possible_paths:
                                if os.path.exists(path):
                                    ffmpeg_path = path
                                    break
                            
                            if ffmpeg_path:
                                audio_source = discord.FFmpegPCMAudio(
                                    audio_buffer,
                                    pipe=True,
                                    executable=ffmpeg_path
                                )
                            else:
                                print("[DEBUG] FFmpeg not found. Please restart terminal or add FFmpeg to PATH")
                                break
                    
                    # Play this chunk
                    voice_client.play(
                        audio_source,
                        after=lambda error: print(f'Chunk {chunk_count} finished: {error}' if error else f'Chunk {chunk_count} finished')
                    )
                    
                    # Wait for this chunk to finish before playing next
                    while voice_client.is_playing():
                        await asyncio.sleep(0.1)
                    
                except asyncio.TimeoutError:
                    print("[DEBUG] Timeout waiting for TTS audio chunk")
                    break
                except Exception as e:
                    print(f"[DEBUG] Error playing TTS chunk {chunk_count}: {e}")
                    break
            
        except Exception as e:
            print(f"[DEBUG] Error in streaming TTS playback: {e}")
        finally:
            # Clear TTS playing flag
            self.tts_playing = False
            print("[DEBUG] Streaming TTS finished - audio processing resumed")
            
            # Update message status to completed
            if hasattr(self.bot, 'chat_handler') and self.bot.chat_handler:
                if self.bot.chat_handler.current_message and self.bot.chat_handler.current_message.get('status') == 'tts_speaking':
                    self.bot.chat_handler.current_message = {"status": "completed", "content": self.bot.chat_handler.current_message.get('content', '')}

    async def _silence_watcher(self):
        """
        Background task that checks silence timers independently of incoming packets.
        This fixes the "awkward buffer stage" when Discord stops sending packets during silence.
        """
        try:
            while True:
                await asyncio.sleep(0.05)  # 50ms tick
                now = time.time()
                # Skip flushing if TTS is currently playing (comprehensive check)
                if self.tts_playing or self._is_any_tts_playing():
                    continue
                    
                for uid in list(self.buffers.keys()):
                    if not self.is_speaking.get(uid, False):
                        continue
                    
                    # Check for 2.5s flush timer (additional to existing silence threshold)
                    last_audio_time = getattr(self, 'last_audio_packet_time', {}).get(uid, 0)
                    if (now - last_audio_time) >= 2.5:  # 2.5 second flush timer
                        # Skip if we're already processing to prevent duplicate responses
                        if self.is_processing:
                            continue
                            
                        # Ensure we had at least some speech
                        spoke_sec = max(0.0, self.last_voice_time[uid] - self.voice_started_time.get(uid, now))
                        if (self.buffers[uid].size >= self.min_samples_needed) and (spoke_sec >= self.min_audio_length):
                            user = self.users.get(uid)
                            if not user:
                                # best effort: skip if we lost the user handle
                                self.is_speaking[uid] = False
                                continue

                            # Take ONLY the last 20s window
                            payload = self.buffers[uid][-self.buffer_max_samples:].copy()

                            # Reset for next utterance
                            self.buffers[uid] = np.array([], dtype=np.float32)
                            self.vad_buffers[uid].clear()
                            self.is_speaking[uid] = False
                            self.speech_run_ms[uid] = 0
                            self.silence_run_ms[uid] = 0
                            self.voice_started_time[uid] = 0.0

                            print(f"[DEBUG] 2.5s flush timer → flushing {payload.size/self.sample_rate:.2f}s for {user.name}")
                            asyncio.run_coroutine_threadsafe(self._flush(user, payload), self.bot.loop)
                            continue
                    
                    # Trigger on real silence (timer-based, independent of new packets)
                    if (now - self.last_voice_time.get(uid, now)) >= self.silence_threshold:
                        # Skip if we're already processing to prevent duplicate responses
                        if self.is_processing:
                            continue
                            
                        # Ensure we had at least some speech
                        spoke_sec = max(0.0, self.last_voice_time[uid] - self.voice_started_time.get(uid, now))
                        if (self.buffers[uid].size >= self.min_samples_needed) and (spoke_sec >= self.min_audio_length):
                            user = self.users.get(uid)
                            if not user:
                                # best effort: skip if we lost the user handle
                                self.is_speaking[uid] = False
                                continue

                            # Take ONLY the last 20s window
                            payload = self.buffers[uid][-self.buffer_max_samples:].copy()

                            # Reset for next utterance
                            self.buffers[uid] = np.array([], dtype=np.float32)
                            self.vad_buffers[uid].clear()
                            self.is_speaking[uid] = False
                            self.speech_run_ms[uid] = 0
                            self.silence_run_ms[uid] = 0
                            self.voice_started_time[uid] = 0.0

                            print(f"[DEBUG] Silence {self.silence_threshold*1000:.0f}ms → flushing {payload.size/self.sample_rate:.2f}s for {user.name}")
                            asyncio.run_coroutine_threadsafe(self._flush(user, payload), self.bot.loop)
        except asyncio.CancelledError:
            return

    def _stereo48k_to_mono16k(self, pcm_bytes: bytes) -> np.ndarray:
        """Fast decimation-based 48k->16k resampler (3x decimation)"""
        if not pcm_bytes or len(pcm_bytes) == 0:
            raise ValueError("Empty audio data")
        
        if len(pcm_bytes) % 2 != 0:
            raise ValueError("Invalid PCM data length")
            
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if pcm.size == 0:
            raise ValueError("No audio samples found")
            
        if pcm.size % 2 == 1:
            pcm = pcm[:-1]
            
        if pcm.size == 0:
            raise ValueError("No valid audio samples after processing")
            
        stereo = pcm.reshape(-1, 2)
        mono48 = stereo.mean(axis=1)

        # 48k -> 16k decimator (fast and good enough for STT)
        mono16 = mono48[::3]
        
        return mono16.astype(np.float32)

    def cleanup(self):
        if hasattr(self, "_watcher_task"):
            self._watcher_task.cancel()
        self.buffers.clear()
        self.vad_buffers.clear()
        self.last_voice_time.clear()
        self.voice_started_time.clear()
        self.is_speaking.clear()
        self.speech_run_ms.clear()
        self.silence_run_ms.clear()
        self.users.clear()
        self.tts_playing = False
        self.is_processing = False
        self.accumulated_audio.clear()
        self.processing_user = None
        self.last_response_time = 0
    
    def is_tts_playing(self) -> bool:
        """Check if TTS is currently playing"""
        return self.tts_playing
    
    def _is_any_tts_playing(self) -> bool:
        """Comprehensive check if ANY TTS is playing in the voice channel"""
        try:
            # Check local TTS flag
            if self.tts_playing:
                return True
            
            # Check if voice client is playing audio
            voice_client = self.text_channel.guild.voice_client
            if voice_client and voice_client.is_playing():
                return True
            
            # Check if bot's TTS handler is active
            if hasattr(self.bot, 'tts_handler') and hasattr(self.bot.tts_handler, 'is_speaking'):
                if self.bot.tts_handler.is_speaking:
                    return True
            
            # Check streaming TTS contexts
            if hasattr(self.bot, 'active_tts_contexts'):
                for tts_context in self.bot.active_tts_contexts:
                    if tts_context.is_speaking:
                        return True
            
            return False
        except Exception as e:
            print(f"[DEBUG] Error checking TTS state: {e}")
            return False
    
    def _resolve_username_to_real_name(self, discord_username: str) -> str:
        """Resolve Discord username to real name from memory system"""
        try:
            # Use the chat handler's memory system
            person = self.bot.chat_handler.memory_system.get_person(discord_username)
            if person:
                print(f"[DEBUG] Resolved {discord_username} to real name: {person.name}")
                return person.name
            
            # Try to find by Discord username in all people
            all_people = self.bot.chat_handler.memory_system.people
            print(f"[DEBUG] Searching {len(all_people)} people for Discord username: {discord_username}")
            
            for person in all_people.values():
                if hasattr(person, 'discord_username') and person.discord_username == discord_username:
                    print(f"[DEBUG] Found person by Discord username: {person.name} (Discord: {person.discord_username})")
                    return person.name
            
            # If no mapping found, return the Discord username
            print(f"[DEBUG] No real name mapping found for {discord_username}, using username")
            return discord_username
            
        except Exception as e:
            print(f"[DEBUG] Error resolving username {discord_username}: {e}")
            return discord_username

class VoiceHandler:
    def __init__(self, bot):
        self.bot = bot
        self.sinks = {}

    async def process_voice(self, ctx):
        print(f"[DEBUG] process_voice called for channel {ctx.channel.name}")
        if not ctx.voice_client:
            print(f"[DEBUG] No voice client found")
            return

        if not isinstance(ctx.voice_client, voice_recv.VoiceRecvClient):
            print(f"[DEBUG] ERROR: Voice client is not VoiceRecvClient! Type: {type(ctx.voice_client)}")
            print(f"[DEBUG] This will cause Opus errors. Reconnecting with VoiceRecvClient...")
            channel = ctx.author.voice.channel if ctx.author.voice else None
            if channel:
                await ctx.voice_client.disconnect()
                await channel.connect(cls=voice_recv.VoiceRecvClient, timeout=20.0, reconnect=True)
            else:
                print(f"[DEBUG] Cannot reconnect - no voice channel found")
                return

        channel_id = ctx.channel.id
        if channel_id not in self.sinks:
            print(f"[DEBUG] Creating new WhisperSink for channel {ctx.channel.name}")
            sink = WhisperSink(self.bot, ctx.channel)
            self.sinks[channel_id] = sink
        else:
            print(f"[DEBUG] WhisperSink already exists, reusing for channel {ctx.channel.name}")
            sink = self.sinks[channel_id]
        
        print(f"[DEBUG] Calling ctx.voice_client.listen(sink)...")
        ctx.voice_client.listen(sink)
        print(f"[DEBUG] Started listening with WhisperSink - sink should now receive packets")

    async def cleanup(self, guild_id):
        for channel_id, sink in list(self.sinks.items()):
            if sink.text_channel and sink.text_channel.guild.id == guild_id:
                sink.cleanup()
                del self.sinks[channel_id]
    
    def stop_voice_processing(self):
        """Stop all voice processing and TTS playback"""
        try:
            # Stop TTS playback in all guilds
            for guild in self.bot.guilds:
                if guild.voice_client and guild.voice_client.is_playing():
                    guild.voice_client.stop()
                    print(f"[DEBUG] TTS stopped in guild {guild.name}")
            
            # Stop any active voice processing
            for sink in self.sinks.values():
                if hasattr(sink, 'is_processing') and sink.is_processing:
                    sink.is_processing = False
                    print(f"[DEBUG] Voice processing stopped in sink")
            
            print("[DEBUG] Voice processing stopped")
        except Exception as e:
            print(f"[DEBUG] Error stopping voice processing: {e}")