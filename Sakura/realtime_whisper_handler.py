import numpy as np
import asyncio
from faster_whisper import WhisperModel
from typing import Optional, AsyncGenerator, Dict
import time
from collections import deque

class RealtimeWhisperHandler:
    def __init__(
        self,
        model_size: str = "medium.en",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 1,
        temperature: float = 0.0,
        english_only: bool = True,
        chunk_duration: float = 1.0,
        overlap_duration: float = 0.3,
    ):
        if device == "auto":
            try:
                import torch
                print(f"[DEBUG] PyTorch version: {torch.__version__}")
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16" if compute_type == "auto" else compute_type
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[DEBUG] Using CUDA for Realtime Whisper - GPU: {gpu_name}")
                    print(f"[DEBUG] CUDA version: {torch.version.cuda}")
                else:
                    device = "cpu"
                    compute_type = "int8" if compute_type == "auto" else compute_type
                    print("[DEBUG] CUDA not available, using CPU for Realtime Whisper")
            except ImportError as e:
                device = "cpu"
                compute_type = "int8" if compute_type == "auto" else compute_type
                print(f"[DEBUG] PyTorch not available, using CPU for Realtime Whisper: {e}")
        elif device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("[DEBUG] CUDA requested but not available, falling back to CPU")
                    device = "cpu"
                    compute_type = "int8" if compute_type == "auto" else compute_type
                else:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[DEBUG] CUDA explicitly requested - GPU: {gpu_name}")
                    compute_type = "float16" if compute_type == "auto" else compute_type
            except ImportError:
                print("[DEBUG] PyTorch not available, falling back to CPU")
                device = "cpu"
                compute_type = "int8" if compute_type == "auto" else compute_type
        
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"[DEBUG] Realtime Whisper loaded: {model_size} on {device} with {compute_type}")
        except Exception as e:
            print(f"[DEBUG] Failed to load Whisper with {device}, falling back to CPU: {e}")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print(f"[DEBUG] Realtime Whisper loaded: {model_size} on CPU with int8")
            
        self.beam_size = beam_size
        self.temperature = temperature
        self.english_only = english_only
        
        self.sample_rate = 16000
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(self.overlap_duration * self.sample_rate)
        
        self.streaming_buffer = {}
        self.last_transcription = {}
        self.stable_text = {}
        self.processing_locks = {}
        
        print(f"[DEBUG] Realtime Whisper config: chunk={chunk_duration}s, overlap={overlap_duration}s")

    async def transcribe_audio(self, audio_data: np.ndarray) -> dict:
        try:
            if audio_data is None or audio_data.size == 0:
                return {"text": "", "language": "unknown", "confidence": 0.0}

            sr = 16000
            audio_duration = len(audio_data) / sr
            
            if audio_duration < 0.10:
                return {"text": "", "language": "unknown", "confidence": 0.0}

            loop = asyncio.get_running_loop()
            text, conf = await loop.run_in_executor(None, self._transcribe_sync, audio_data)

            return {
                "text": text,
                "language": "en" if self.english_only else ("en" if text else "unknown"),
                "confidence": float(conf),
            }

        except Exception as e:
            print(f"[REALTIME-STT] Error: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0}

    async def transcribe_streaming(
        self, 
        user_id: str, 
        audio_chunk: np.ndarray
    ) -> AsyncGenerator[Dict, None]:
        if user_id not in self.streaming_buffer:
            self.streaming_buffer[user_id] = np.array([], dtype=np.float32)
            self.last_transcription[user_id] = ""
            self.stable_text[user_id] = ""
            self.processing_locks[user_id] = asyncio.Lock()
        
        async with self.processing_locks[user_id]:
            self.streaming_buffer[user_id] = np.concatenate([
                self.streaming_buffer[user_id], 
                audio_chunk
            ])
            
            max_buffer_samples = int(30.0 * self.sample_rate)
            if self.streaming_buffer[user_id].size > max_buffer_samples:
                excess = self.streaming_buffer[user_id].size - max_buffer_samples
                self.streaming_buffer[user_id] = self.streaming_buffer[user_id][excess:]
            
            buffer_duration = self.streaming_buffer[user_id].size / self.sample_rate
            
            if buffer_duration >= self.chunk_duration:
                chunk_to_process = self.streaming_buffer[user_id][-self.chunk_samples:].copy()
                
                loop = asyncio.get_running_loop()
                start_time = time.time()
                text, confidence = await loop.run_in_executor(
                    None, 
                    self._transcribe_sync, 
                    chunk_to_process
                )
                processing_time = time.time() - start_time
                
                if text and confidence > 0.2:
                    new_text = self._stabilize_text(user_id, text)
                    
                    if new_text and new_text != self.last_transcription[user_id]:
                        self.last_transcription[user_id] = new_text
                        
                        yield {
                            "text": new_text,
                            "partial": True,
                            "confidence": confidence,
                            "processing_time": processing_time,
                            "is_final": False
                        }

    def _stabilize_text(self, user_id: str, new_text: str) -> str:
        if not new_text:
            return ""
        
        new_text = new_text.strip()
        
        if not self.stable_text[user_id]:
            self.stable_text[user_id] = new_text
            return new_text
        
        stable = self.stable_text[user_id]
        
        if new_text.startswith(stable):
            addition = new_text[len(stable):].strip()
            if addition:
                self.stable_text[user_id] = new_text
                return new_text
            return stable
        
        common_words = self._find_common_suffix(stable, new_text)
        if common_words:
            self.stable_text[user_id] = new_text
            return new_text
        
        self.stable_text[user_id] = new_text
        return new_text
    
    def _find_common_suffix(self, text1: str, text2: str) -> str:
        words1 = text1.split()
        words2 = text2.split()
        
        common = []
        min_len = min(len(words1), len(words2))
        
        for i in range(1, min_len + 1):
            if words1[-i] == words2[-i]:
                common.insert(0, words1[-i])
            else:
                break
        
        return " ".join(common)

    async def finalize_transcription(self, user_id: str) -> dict:
        if user_id not in self.streaming_buffer:
            return {"text": "", "language": "unknown", "confidence": 0.0}
        
        async with self.processing_locks[user_id]:
            final_audio = self.streaming_buffer[user_id].copy()
            
            self.streaming_buffer[user_id] = np.array([], dtype=np.float32)
            self.last_transcription[user_id] = ""
            self.stable_text[user_id] = ""
        
        if final_audio.size > 0:
            result = await self.transcribe_audio(final_audio)
            result["is_final"] = True
            return result
        
        return {"text": "", "language": "unknown", "confidence": 0.0, "is_final": True}

    def reset_streaming(self, user_id: str):
        if user_id in self.streaming_buffer:
            self.streaming_buffer[user_id] = np.array([], dtype=np.float32)
            self.last_transcription[user_id] = ""
            self.stable_text[user_id] = ""

    def _transcribe_sync(self, audio_f32: np.ndarray):
        if audio_f32.dtype != np.float32:
            audio_f32 = audio_f32.astype(np.float32)
        audio_f32 = np.clip(audio_f32, -1.0, 1.0)

        segments, info = self.model.transcribe(
            audio_f32,
            beam_size=self.beam_size,
            vad_filter=False,
            temperature=self.temperature,
            language="en" if self.english_only else None,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        out_text = []
        logprobs = []
        for seg in segments:
            if seg.text:
                out_text.append(seg.text)
            if getattr(seg, "avg_logprob", None) is not None:
                logprobs.append(seg.avg_logprob)

        text = " ".join(t.strip() for t in out_text).strip()
        if logprobs:
            import math
            avg_lp = sum(logprobs) / len(logprobs)
            conf = 1.0 / (1.0 + math.exp(-(avg_lp + 1.5)))
        else:
            conf = 1.0 if text else 0.0

        return text, conf
