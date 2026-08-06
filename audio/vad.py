import numpy as np
import torch

from silero_vad import load_silero_vad, get_speech_timestamps

from config import SAMPLE_RATE, VAD_THRESHOLD


print("[VAD] Loading...")


vad_model = load_silero_vad()


print("[VAD] Ready")



def contains_speech(
    audio: np.ndarray
) -> bool:


    if audio.dtype != np.float32:

        audio = (
            audio.astype(np.float32)
            /
            32768.0
        )


    waveform = torch.from_numpy(
        audio
    )


    timestamps = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD
    )


    return len(timestamps) > 0