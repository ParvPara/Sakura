import azure.cognitiveservices.speech as speechsdk
import config
import io
import asyncio
import threading
import queue
import time
import re

class TTSHandler:
    def __init__(self):
        # Initialize Azure Speech Service
        self.speech_config = speechsdk.SpeechConfig(
            subscription=config.AZURE_SPEECH_KEY, 
            region=config.AZURE_SPEECH_REGION
        )
        
        # Set voice to Sara neural with 20% pitch increase
        self.voice_name = "en-US-SaraNeural"
        self.max_chars = 1000
        
        self.ssml_template = '''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice_name}">
                <prosody pitch="+20%" rate="{rate}">{text}</prosody>
            </voice>
        </speak>'''
        
        # Current synthesis task for cancellation
        self._current_synthesis = None
        self._synthesis_lock = threading.Lock()
        
        # Streaming TTS support
        self._streaming_synthesizer = None
        self._streaming_lock = threading.Lock()
        self._audio_queue = queue.Queue()
        self._is_streaming = False
        
        self._current_speech_rate = "+0%" #0% = default rate
        self._context_aware_enabled = True
        
    def _analyze_text_context(self, text: str) -> dict:
        """Analyze text for context-aware TTS adjustments"""
        context = {
            'speech_rate': "+0%",  # Default rate
            'has_fast_speech_trigger': False,
            'has_question': False,
            'has_exclamation': False,
            'has_ellipsis': False,
            'has_pause_triggers': False
        }
        
        text_lower = text.lower()
        
        # Check for fast speech triggers
        fast_speech_triggers = [
            "i'm going to talk fast",
            "i am going to talk fast",
            "talking fast",
            "speaking quickly",
            "rambling",
            "going on a tangent",
            "getting excited",
            "getting worked up"
        ]
        
        for trigger in fast_speech_triggers:
            if trigger in text_lower:
                context['has_fast_speech_trigger'] = True
                context['speech_rate'] = "+50%"  # Much faster
                break
        
        # Check for punctuation that affects pacing
        if '?' in text:
            context['has_question'] = True
            context['has_pause_triggers'] = True
            
        if '!' in text:
            context['has_exclamation'] = True
            context['has_pause_triggers'] = True
            
        if '...' in text or '…' in text:
            context['has_ellipsis'] = True
            context['has_pause_triggers'] = True
        
        return context

    def _add_ssml_pauses(self, text: str) -> str:
        """Add SSML pause tags for natural speech patterns"""
        # Add pauses after questions and exclamations (more natural)
        text = re.sub(r'(\?)(\s*)', r'\1<break time="600ms"/>\2', text)
        text = re.sub(r'(!)(\s*)', r'\1<break time="500ms"/>\2', text)
        
        # Add longer pauses for ellipsis
        text = re.sub(r'(\.{3,}|…)(\s*)', r'\1<break time="800ms"/>\2', text)
        
        # Add medium pauses for periods (but not in abbreviations or acronyms)
        text = re.sub(r'(?<!\w)\.(?!\w)(\s*)', r'.\1<break time="400ms"/>', text)
        
        # Add slight pauses for commas
        text = re.sub(r',(\s*)', r',<break time="200ms"/>\1', text)
        
        # Add pauses for texting-style elements
        # Pause after acronyms like "lol", "omg", "wtf"
        text = re.sub(r'\b(lol|lmao|omg|wtf|tbh|fr|ngl|imo|btw|idk|ttyl|brb)\b(\s*)', r'\1<break time="150ms"/>\2', text, flags=re.IGNORECASE)
        
        # Pause after spam letters for emphasis
        text = re.sub(r'([a-z])\1{3,}(\s*)', r'\1\1\1\1<break time="200ms"/>\2', text, flags=re.IGNORECASE)
        
        # Pause after Japanese expressions
        text = re.sub(r'\b(ohayo|konnichiwa|arigato|sugoi|kawaii|ne|yo|wa)\b(\s*)', r'\1<break time="200ms"/>\2', text, flags=re.IGNORECASE)
        
        # Debug: check if breaks were added
        if '<break' in text:
            print(f"[TTS] Added SSML breaks: {text.count('<break')} pauses")
        
        return text

    def _normalize_interjections(self, text: str) -> str:
        """Normalize interjections and filler words to prevent voice changes"""
        replacements = {
            r'\byeah\b': 'yes',
            r'\byah\b': 'yes',
            r'\byep\b': 'yes',
            r'\buh\b': 'uh',
            r'\bugh\b': 'uh',
            r'\bumm\b': 'um',
            r'\buhmm\b': 'um',
            r'\bhmm\b': 'hm',
            r'\bhmmm\b': 'hm',
            r'\bahh\b': 'ah',
            r'\bahhh\b': 'ah',
            r'\bohh\b': 'oh',
            r'\bohhh\b': 'oh',
            r'\behh\b': 'eh',
            r'\bmeh\b': 'eh',
        }
        
        normalized = text
        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _process_text_for_tts(self, text: str) -> tuple:
        """Process text for context-aware TTS with dynamic rate and pauses"""
        if not self._context_aware_enabled:
            return text, "+0%"
        
        # Normalize interjections first
        text = self._normalize_interjections(text)
        
        # Analyze context
        context = self._analyze_text_context(text)
        
        # Update current speech rate
        self._current_speech_rate = context['speech_rate']
        
        # Add SSML pauses if needed
        processed_text = text
        if context['has_pause_triggers']:
            processed_text = self._add_ssml_pauses(text)
        
        print(f"[TTS] Context analysis - Rate: {context['speech_rate']}, Fast trigger: {context['has_fast_speech_trigger']}, Pauses: {context['has_pause_triggers']}")
        
        return processed_text, context['speech_rate']

    async def generate_speech(self, text: str) -> io.BytesIO:
        try:
            if not text:
                return None

            if len(text) > self.max_chars:
                text = text[:self.max_chars] + "..."

            print(f"[TTS] Starting synthesis for: '{text[:50]}...'")
            
            if not hasattr(config, 'AZURE_SPEECH_KEY') or not config.AZURE_SPEECH_KEY:
                print("[TTS] Error: AZURE_SPEECH_KEY not configured")
                return None
            
            if not hasattr(config, 'AZURE_SPEECH_REGION') or not config.AZURE_SPEECH_REGION:
                print("[TTS] Error: AZURE_SPEECH_REGION not configured")
                return None

            # Check if content should be filtered for TTS
            # Note: The original text is preserved for display, only TTS gets filtered
            tts_text = text
            if hasattr(config, 'FILTER_ENABLED') and config.FILTER_ENABLED:
                # Import filter handler here to avoid circular imports
                try:
                    from Sakura.filter_handler import FilterHandler
                    filter_handler = FilterHandler()
                    if filter_handler.is_filtered(text):
                        print(f"[TTS] Content filtered, TTS will speak 'Filtered.' instead of: '{text[:50]}...'")
                        tts_text = "Filtered."
                except Exception as e:
                    print(f"[TTS] Warning: Could not check filter status: {e}")
                    # Continue with original text if filter check fails

            # Process text for context-aware TTS
            processed_text, speech_rate = self._process_text_for_tts(tts_text)
            
            # Try SSML synthesis with context-aware adjustments
            loop = asyncio.get_running_loop()
            audio_data = await loop.run_in_executor(None, self._synthesize_with_context, processed_text, speech_rate)
            
            if not audio_data:
                return None

            buffer = io.BytesIO(audio_data)
            buffer.seek(0)
            return buffer

        except Exception as e:
            print(f"[TTS] Azure Speech error: {e}")
            return None
    
    async def generate_and_play_speech(self, text: str, bot=None):
        """Generate speech and play it through Discord voice client"""
        try:
            print(f"[TTS] generate_and_play_speech called with: '{text[:50]}...'")
            
            # Generate the audio
            audio_data = await self.generate_speech(text)
            if not audio_data:
                print(f"[TTS] Failed to generate speech for: '{text[:50]}...'")
                return False
            
            print(f"[TTS] Speech generated successfully for: '{text[:50]}...'")
            
            # If bot is provided, try to play through voice client
            if bot and hasattr(bot, 'voice_clients') and bot.voice_clients:
                voice_client = bot.voice_clients[0]
                
                if voice_client and voice_client.is_connected():
                    print(f"[TTS] Playing audio through voice client")
                    
                    # Stop any current audio
                    if voice_client.is_playing():
                        print("[TTS] Stopping current audio to play new TTS")
                        voice_client.stop()
                    
                    # Create audio source
                    import discord
                    try:
                        audio_source = discord.FFmpegPCMAudio(
                            audio_data,
                            pipe=True
                        )
                    except discord.errors.ClientException:
                        # Try with explicit FFmpeg path
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
                                audio_data,
                                pipe=True,
                                executable=ffmpeg_path
                            )
                        else:
                            print("[TTS] FFmpeg not found. Cannot play audio.")
                            return False
                    
                    # Play the audio
                    voice_client.play(
                        audio_source,
                        after=lambda e: print(f"[TTS] Playback finished: {e}") if e else print("[TTS] Playback completed successfully")
                    )
                    
                    print(f"[TTS] Audio playback started")
                    return True
                else:
                    print(f"[TTS] No voice client connected, cannot play audio")
                    return False
            else:
                print(f"[TTS] No bot or voice clients provided, audio generated but not played")
                return True
                
        except Exception as e:
            print(f"[TTS] Error in generate_and_play_speech: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def generate_speech_streaming(self, text_chunks: list) -> asyncio.Queue:
        """Generate streaming speech for ultra-low latency with event-driven audio chunks"""
        try:
            if not text_chunks:
                audio_queue = asyncio.Queue()
                await audio_queue.put(None)
                return audio_queue

            print(f"[TTS] Starting streaming synthesis for {len(text_chunks)} chunks")
            
            if not hasattr(config, 'AZURE_SPEECH_KEY') or not config.AZURE_SPEECH_KEY:
                print("[TTS] Error: AZURE_SPEECH_KEY not configured")
                audio_queue = asyncio.Queue()
                await audio_queue.put(None)
                return audio_queue
            
            if not hasattr(config, 'AZURE_SPEECH_REGION') or not config.AZURE_SPEECH_REGION:
                print("[TTS] Error: AZURE_SPEECH_REGION not configured")
                audio_queue = asyncio.Queue()
                await audio_queue.put(None)
                return audio_queue

            # Check for filtered content in chunks
            filtered_chunks = []
            for chunk in text_chunks:
                if hasattr(config, 'FILTER_ENABLED') and config.FILTER_ENABLED:
                    try:
                        from Sakura.filter_handler import FilterHandler
                        filter_handler = FilterHandler()
                        if filter_handler.is_filtered(chunk):
                            print(f"[TTS] Streaming chunk filtered, TTS will speak 'Filtered.' instead")
                            filtered_chunks.append("Filtered.")
                        else:
                            filtered_chunks.append(chunk)
                    except Exception as e:
                        print(f"[TTS] Warning: Could not check filter status for chunk: {e}")
                        filtered_chunks.append(chunk)
                else:
                    filtered_chunks.append(chunk)

            # Setup streaming synthesizer if not already done
            if not self._streaming_synthesizer:
                await self._setup_streaming_synthesizer_async()
            
            if not self._streaming_synthesizer:
                print("[TTS] Failed to create streaming synthesizer, falling back to non-streaming")
                # Fallback to non-streaming synthesis
                audio_queue = asyncio.Queue()
                
                for i, chunk in enumerate(filtered_chunks, 1):
                    print(f"[TTS] Fallback processing chunk {i}/{len(filtered_chunks)}: '{chunk[:30]}...'")
                    
                    # Process chunk for context-aware TTS
                    processed_chunk, chunk_rate = self._process_text_for_tts(chunk)
                    
                    # Use non-streaming synthesis as fallback with context awareness
                    audio_data = await asyncio.get_running_loop().run_in_executor(
                        None, self._synthesize_with_context, processed_chunk, chunk_rate
                    )
                    
                    if audio_data:
                        print(f"[TTS] Fallback chunk {i} synthesized: {len(audio_data)} bytes")
                        await audio_queue.put(audio_data)
                    else:
                        print(f"[TTS] Fallback chunk {i} failed to synthesize")
                
                print("[TTS] Fallback synthesis completed")
                await audio_queue.put(None)  # End of stream
                return audio_queue
            
            # Use improved streaming synthesis with larger chunks
            print(f"[TTS] Starting streaming synthesis for {len(text_chunks)} chunks")
            
            # Combine chunks into larger sentences to reduce choppiness
            combined_chunks = []
            current_chunk = ""
            
            for chunk in text_chunks:
                if not chunk.strip():
                    continue
                    
                # If current chunk is getting too long, start a new one
                if len(current_chunk) + len(chunk) > 200:
                    if current_chunk:
                        combined_chunks.append(current_chunk.strip())
                    current_chunk = chunk
                else:
                    current_chunk += " " + chunk if current_chunk else chunk
            
            # Add the last chunk
            if current_chunk:
                combined_chunks.append(current_chunk.strip())
            
            print(f"[TTS] Combined {len(text_chunks)} chunks into {len(combined_chunks)} larger chunks")
            
            # Use reliable non-streaming synthesis for each combined chunk
            audio_queue = asyncio.Queue()
            
            for i, chunk in enumerate(combined_chunks, 1):
                if chunk.strip():
                    print(f"[TTS] Processing combined chunk {i}/{len(combined_chunks)}: '{chunk[:50]}...'")
                    
                    # Process chunk for context-aware TTS
                    processed_chunk, chunk_rate = self._process_text_for_tts(chunk)
                    
                    # Use non-streaming synthesis for reliability with context awareness
                    audio_data = await asyncio.get_running_loop().run_in_executor(
                        None, self._synthesize_with_context, processed_chunk, chunk_rate
                    )
                    
                    if audio_data:
                        print(f"[TTS] Combined chunk {i} synthesized: {len(audio_data)} bytes")
                        await audio_queue.put(audio_data)
                    else:
                        print(f"[TTS] Combined chunk {i} failed to synthesize")
            
            print("[TTS] Streaming synthesis completed")
            await audio_queue.put(None)  # End of stream
            return audio_queue
            
        except Exception as e:
            print(f"[TTS] Error in streaming synthesis: {e}")
            # Return empty queue on error
            audio_queue = asyncio.Queue()
            await audio_queue.put(None)
            return audio_queue

    async def generate_speech_chunked(self, text: str, chunk_size: int = 50) -> list:
        """
        Split text into chunks and generate speech for each chunk
        Useful for streaming LLM responses
        """
        try:
            if not text:
                return []

            # Check if content should be filtered for TTS
            # Note: The original text is preserved for display, only TTS gets filtered
            tts_text = text
            if hasattr(config, 'FILTER_ENABLED') and config.FILTER_ENABLED:
                try:
                    from Sakura.filter_handler import FilterHandler
                    filter_handler = FilterHandler()
                    if filter_handler.is_filtered(text):
                        print(f"[TTS] Chunked content filtered, TTS will speak 'Filtered.' instead of: '{text[:50]}...'")
                        tts_text = "Filtered."
                except Exception as e:
                    print(f"[TTS] Warning: Could not check filter status for chunked text: {e}")
                    # Continue with original text if filter check fails

            # Split text into chunks (try to break at sentence boundaries)
            import re
            sentences = re.split(r'[.!?]+', tts_text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would exceed chunk size, start new chunk
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += ". " + sentence
                    else:
                        current_chunk = sentence
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            print(f"[TTS] Split text into {len(chunks)} chunks for streaming")
            return chunks
            
        except Exception as e:
            print(f"[TTS] Error chunking text: {e}")
            return [text]

    def _synthesize_speech(self, ssml_text: str) -> bytes:
        try:
            with self._synthesis_lock:
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config, 
                    audio_config=None
                )
                
                self._current_synthesis = synthesizer
                
                result = synthesizer.speak_ssml_async(ssml_text).get()
                
                self._current_synthesis = None
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print(f"[TTS] Successfully synthesized {len(result.audio_data)} bytes of audio")
                    return result.audio_data
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    print(f"[TTS] Speech synthesis canceled: {cancellation.reason}")
                    if cancellation.reason == speechsdk.CancellationReason.Error:
                        print(f"[TTS] Error details: {cancellation.error_details}")
                    return None
                else:
                    print(f"[TTS] Speech synthesis failed: {result.reason}")
                    return None
                    
        except Exception as e:
            print(f"[TTS] Synthesis error: {e}")
            return None

    def _synthesize_with_context(self, text: str, speech_rate: str = "+0%") -> bytes:
        """Synthesize text with context-aware speech rate and pauses"""
        try:
            with self._synthesis_lock:
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config, 
                    audio_config=None
                )
                
                self._current_synthesis = synthesizer
                
                # Create SSML with dynamic rate and pauses
                ssml = self._create_context_aware_ssml(text, speech_rate)
                
                print(f"[TTS] Using context-aware SSML - Rate: {speech_rate}, Pitch: +25%")
                result = synthesizer.speak_ssml_async(ssml).get()
                
                self._current_synthesis = None
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print(f"[TTS] Successfully synthesized {len(result.audio_data)} bytes with context-aware settings")
                    return result.audio_data
                elif result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    print(f"[TTS] Context-aware SSML synthesis canceled: {cancellation.reason}")
                    if cancellation.reason == speechsdk.CancellationReason.Error:
                        print(f"[TTS] Error details: {cancellation.error_details}")
                        print(f"[TTS] Falling back to simple text synthesis")
                        return self._synthesize_simple_text(text)
                    return None
                else:
                    print(f"[TTS] Context-aware SSML synthesis failed: {result.reason}, falling back to simple text")
                    return self._synthesize_simple_text(text)
                    
        except Exception as e:
            print(f"[TTS] Context-aware SSML synthesis error: {e}, falling back to simple text")
            return self._synthesize_simple_text(text)

    def _synthesize_with_pitch(self, text: str) -> bytes:
        """Legacy method for backward compatibility"""
        return self._synthesize_with_context(text, "+0%")

    def _synthesize_simple_text(self, text: str) -> bytes:
        try:
            with self._synthesis_lock:
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config, 
                    audio_config=None
                )
                
                self._current_synthesis = synthesizer
                self.speech_config.speech_synthesis_voice_name = self.voice_name
                
                result = synthesizer.speak_text_async(text).get()
                self._current_synthesis = None
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print(f"[TTS] Successfully synthesized {len(result.audio_data)} bytes (fallback)")
                    return result.audio_data
                else:
                    print(f"[TTS] Simple text synthesis failed: {result.reason}")
                    return None
                    
        except Exception as e:
            print(f"[TTS] Simple synthesis error: {e}")
            return None

    def stop_current_speech(self):
        """
        Stop current speech generation
        """
        try:
            with self._synthesis_lock:
                if self._current_synthesis:
                    # Azure Speech SDK doesn't have a direct stop method for synthesis
                    # The synthesis will be interrupted when the synthesizer is destroyed
                    self._current_synthesis = None
                    print("[TTS] Speech synthesis stop requested")
        except Exception as e:
            print(f"[TTS] Error stopping speech generation: {e}")

    def get_remaining_characters(self) -> int:
        """
        Azure Speech Service has generous limits, return a high number
        Note: Actual quota tracking would require separate Azure API calls
        """
        try:
            return 100000
        except Exception as e:
            print(f"[TTS] Error getting remaining characters: {e}")
            return 0

    def _escape_ssml(self, text: str) -> str:
        """
        Escape special characters for SSML
        """
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("\"", "&quot;")
        text = text.replace("'", "&apos;")
        return text

    def _setup_streaming_synthesizer(self):
        """Setup Azure streaming synthesizer (synchronous version)"""
        try:
            with self._streaming_lock:
                # Configure for streaming - use WAV format that Discord can play
                self.speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
                )
                
                # Set voice explicitly to avoid speaker activation issues
                self.speech_config.speech_synthesis_voice_name = self.voice_name
                
                # Create synthesizer with custom audio config for streaming
                audio_config = speechsdk.audio.AudioOutputConfig(stream=None)
                self._streaming_synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config
                )
                
                # Test the synthesizer with a small sample to activate it
                test_ssml = self._create_context_aware_ssml("Test", "+0%")
                
                test_result = self._streaming_synthesizer.speak_ssml_async(test_ssml).get()
                if test_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print("[TTS] Streaming synthesizer created and activated successfully")
                else:
                    print(f"[TTS] Warning: Test synthesis failed: {test_result.reason}")
                    # Continue anyway, might work for actual content
                
        except Exception as e:
            print(f"[TTS] Error setting up streaming synthesizer: {e}")
            self._streaming_synthesizer = None

    async def _setup_streaming_synthesizer_async(self):
        """Setup Azure streaming synthesizer with event-driven audio chunks (async version)"""
        try:
            with self._streaming_lock:
                # Configure for streaming - use WAV format that Discord can play
                self.speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
                )
                
                # Set voice explicitly to avoid speaker activation issues
                self.speech_config.speech_synthesis_voice_name = self.voice_name
                
                # IMPORTANT: give it an output stream (pull), so you control playback
                pull_stream = speechsdk.audio.PullAudioOutputStream()
                audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
                
                synth = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config
                )
                self._streaming_synthesizer = synth
                
                # Setup event-driven audio streaming
                self._evt_loop = asyncio.get_running_loop()
                self._audio_async_queue = asyncio.Queue()
                
                def on_synthesizing(evt: speechsdk.SpeechSynthesisEventArgs):
                    # evt.result.audio_data is a bytes chunk
                    if evt.result and evt.result.audio_data:
                        # bridge from SDK thread to asyncio world
                        self._evt_loop.call_soon_threadsafe(
                            self._audio_async_queue.put_nowait,
                            evt.result.audio_data
                        )
                
                def on_completed(evt: speechsdk.SessionEventArgs):
                    # signal end of utterance
                    self._evt_loop.call_soon_threadsafe(self._audio_async_queue.put_nowait, None)
                
                self._streaming_synthesizer.synthesizing.connect(on_synthesizing)
                self._streaming_synthesizer.synthesis_completed.connect(on_completed)
                self._streaming_synthesizer.synthesis_canceled.connect(on_completed)
                
                # Test the synthesizer with a small sample to activate it
                test_ssml = self._create_context_aware_ssml("Test", "+0%")
                
                test_result = self._streaming_synthesizer.speak_ssml_async(test_ssml).get()
                if test_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    print("[TTS] Streaming synthesizer created and activated successfully")
                else:
                    print(f"[TTS] Warning: Test synthesis failed: {test_result.reason}")
                    # Continue anyway, might work for actual content
                
        except Exception as e:
            print(f"[TTS] Error setting up streaming synthesizer: {e}")
            self._streaming_synthesizer = None

    def _synthesize_chunk(self, text: str) -> bytes:
        """Synthesize a single text chunk"""
        try:
            if not self._streaming_synthesizer:
                return None
                
            # Create context-aware SSML for this chunk
            processed_text, speech_rate = self._process_text_for_tts(text)
            ssml = self._create_context_aware_ssml(processed_text, speech_rate)
            
            # Synthesize the chunk (non-blocking)
            future = self._streaming_synthesizer.start_speaking_ssml_async(ssml)
            result = future.get()  # waits only for this utterance's completion signal
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            else:
                print(f"[TTS] Chunk synthesis failed: {result.reason}")
                return None
                
        except Exception as e:
            print(f"[TTS] Error synthesizing chunk: {e}")
            return None

    def _create_context_aware_ssml(self, text: str, speech_rate: str = "+0%") -> str:
        """Create SSML with context-aware speech rate and pauses - text already has breaks from _process_text_for_tts"""
        # The text already has <break> tags added by _process_text_for_tts, so we just need to
        # escape XML characters in text content but preserve SSML tags
        processed_text = text
        
        # Escape XML characters in text content but preserve SSML tags
        # First, temporarily replace break tags with placeholders
        import uuid
        import re
        break_placeholder = f"__BREAK_{uuid.uuid4().hex[:8]}__"
        break_tags = []
        
        # Extract all break tags
        break_pattern = r'<break[^>]*>'
        break_matches = re.findall(break_pattern, processed_text)
        
        # Replace break tags with placeholders
        for i, match in enumerate(break_matches):
            placeholder = f"{break_placeholder}{i}"
            processed_text = processed_text.replace(match, placeholder, 1)
            break_tags.append(match)
        
        # Now escape XML characters in the remaining text
        processed_text = processed_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # Restore break tags
        for i, tag in enumerate(break_tags):
            placeholder = f"{break_placeholder}{i}"
            processed_text = processed_text.replace(placeholder, tag)
        
        # Debug: check if break tags were preserved
        break_count = processed_text.count('<break')
        if break_count > 0:
            print(f"[TTS] Preserved {break_count} SSML break tags for reliable pauses")
        else:
            print(f"[TTS] No SSML break tags found in processed text: {text[:100]}...")
        
        # ENHANCED SSML structure - using hybrid silence + prosody rate changes for dramatic pauses
        # This approach combines silence tags with rate changes for maximum pause effect
        return f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{self.voice_name}">
            <prosody pitch="+25%" rate="{speech_rate}">
                {processed_text}
            </prosody>
            </voice>
        </speak>'''

    def _create_ssml(self, text: str) -> str:
        """Legacy method for backward compatibility"""
        return self._create_context_aware_ssml(text, "+0%")

    def _cleanup_streaming_synthesizer(self):
        """Cleanup streaming synthesizer"""
        try:
            with self._streaming_lock:
                if self._streaming_synthesizer:
                    self._streaming_synthesizer = None
                    print("[TTS] Streaming synthesizer cleaned up")
        except Exception as e:
            print(f"[TTS] Error cleaning up streaming synthesizer: {e}")
    