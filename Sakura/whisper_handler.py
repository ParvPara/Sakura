import numpy as np
import asyncio
from faster_whisper import WhisperModel

class WhisperHandler:
    """
    Drop-in replacement for your current OpenAI Whisper handler.
    Expects audio_data as float32 numpy array in [-1, 1], 16 kHz mono.
    Returns: {"text": str, "language": "en", "confidence": float}
    """
    def __init__(
        self,
        model_size: str = "medium.en",   
        device: str = "auto",            
        compute_type: str = "auto",      
        beam_size: int = 1,
        temperature: float = 0.0,
        english_only: bool = True    
    ):
        if device == "auto":
            try:
                import torch
                print(f"[DEBUG] PyTorch version: {torch.__version__}")
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16" if compute_type == "auto" else compute_type
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[DEBUG] Using CUDA for Whisper - GPU: {gpu_name}")
                    print(f"[DEBUG] CUDA version: {torch.version.cuda}")
                else:
                    device = "cpu"
                    compute_type = "int8" if compute_type == "auto" else compute_type
                    print("[DEBUG] CUDA not available, using CPU for Whisper")
            except ImportError as e:
                device = "cpu"
                compute_type = "int8" if compute_type == "auto" else compute_type
                print(f"[DEBUG] PyTorch not available, using CPU for Whisper: {e}")
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
            print(f"[DEBUG] Whisper loaded: {model_size} on {device} with {compute_type}")
        except Exception as e:
            print(f"[DEBUG] Failed to load Whisper with {device}, falling back to CPU: {e}")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            print(f"[DEBUG] Whisper loaded: {model_size} on CPU with int8")
            
        self.beam_size = beam_size
        self.temperature = temperature
        self.english_only = english_only

    async def transcribe_audio(self, audio_data: np.ndarray) -> dict:
        try:
            if audio_data is None or audio_data.size == 0:
                return {"text": "", "language": "unknown", "confidence": 0.0}

            sr = 16000
            audio_duration = len(audio_data) / sr
            print(f"[DEBUG] Audio duration: {audio_duration:.2f}s, samples: {len(audio_data)}")
            
            if audio_duration < 0.10:
                print(f"[DEBUG] Audio too short ({audio_duration:.2f}s), skipping")
                return {"text": "", "language": "unknown", "confidence": 0.0}

            loop = asyncio.get_running_loop()
            text, conf = await loop.run_in_executor(None, self._transcribe_sync, audio_data)

            return {
                "text": text,
                "language": "en" if self.english_only else ("en" if text else "unknown"),
                "confidence": float(conf),
            }

        except Exception as e:
            print(f"[STT] faster-whisper error: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0}

    # -------- sync worker --------
    def _transcribe_sync(self, audio_f32: np.ndarray):
        """
        Perform the actual faster-whisper call.
        Returns (text, confidence). Confidence is a heuristic based on avg_logprob.
        """
        # Ensure float32 numpy in [-1,1]
        if audio_f32.dtype != np.float32:
            audio_f32 = audio_f32.astype(np.float32)
        # Clip just in case
        audio_f32 = np.clip(audio_f32, -1.0, 1.0)

        segments, info = self.model.transcribe(
            audio_f32,
            beam_size=self.beam_size,
            vad_filter=False,                 # you already handle segmentation upstream
            temperature=self.temperature,
            language="en" if self.english_only else None,
            condition_on_previous_text=False, # lower latency / fewer cross-segment deps
            word_timestamps=False,
        )

        # Concatenate text and compute a simple confidence heuristic
        out_text = []
        logprobs = []
        for seg in segments:
            if seg.text:
                out_text.append(seg.text)
            # faster-whisper exposes avg_logprob on segments; map to [0,1] for a rough confidence
            if getattr(seg, "avg_logprob", None) is not None:
                logprobs.append(seg.avg_logprob)

        text = " ".join(t.strip() for t in out_text).strip()
        if logprobs:
            # avg logprob ~[-5,0]; map to [0,1] via sigmoid-ish transform
            import math
            avg_lp = sum(logprobs) / len(logprobs)
            conf = 1.0 / (1.0 + math.exp(-(avg_lp + 1.5)))  # shift center
        else:
            conf = 1.0 if text else 0.0

        return text, conf