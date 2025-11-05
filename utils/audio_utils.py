import numpy as np
import config
from scipy import signal

def convert_audio_to_numpy(audio_data: bytes) -> np.ndarray:
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0
    return audio_array

def prepare_audio_for_whisper(audio_array: np.ndarray) -> np.ndarray:
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    audio_array = np.clip(audio_array, -1.0, 1.0)
    
    return audio_array

def detect_silence(audio_array: np.ndarray, threshold: float = 0.01) -> bool:
    return np.abs(audio_array).mean() < threshold