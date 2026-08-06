import numpy as np

from silero_vad import load_silero_vad, VADIterator

from config import SAMPLE_RATE, VAD_THRESHOLD

from core.console import log


log("VAD", "Loading...")

_model = load_silero_vad()

# Silero's streaming detector is stateful and requires EXACTLY
# 512-sample chunks at 16kHz per call - confirmed against the real
# package, passing any other length raises inside the TorchScript
# model. We buffer whatever length comes in from the mic (1280
# samples, sized for openWakeWord) and feed the detector fixed
# 512-sample slices sliced off that buffer.
#
# This replaces the previous approach of calling get_speech_timestamps()
# fresh on each isolated mic block: that function's own default
# min_speech_duration_ms is 250ms, and it throws out any reported
# segment shorter than that - so a single ~30-80ms block, even one
# that's entirely speech, could never be reported. That's the actual
# reason the assistant couldn't hear commands after the wake word.

_WINDOW = 512

_iterator = VADIterator(
    _model,
    threshold=VAD_THRESHOLD,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=100
)

_buffer = np.array([], dtype=np.float32)

_speaking = False


log("VAD", "Ready")



def contains_speech(
    audio: np.ndarray
) -> bool:

    global _buffer, _speaking


    if audio.dtype != np.float32:

        audio = (
            audio.astype(np.float32)
            /
            32768.0
        )


    _buffer = np.concatenate(
        [_buffer, audio]
    )


    while len(_buffer) >= _WINDOW:

        chunk = _buffer[:_WINDOW]

        _buffer = _buffer[_WINDOW:]


        event = _iterator(chunk)


        if event is not None:

            if "start" in event:

                _speaking = True


            if "end" in event:

                _speaking = False



    return _speaking



def reset():

    # Clear buffered audio and the detector's recurrent state between
    # separate listen_for_command() sessions, so a previous command's
    # tail can't bleed into the next one.

    global _buffer, _speaking

    _buffer = np.array([], dtype=np.float32)

    _speaking = False

    _iterator.reset_states()