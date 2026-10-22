import queue
import threading
import time

import numpy as np
import sounddevice as sd

from piper import PiperVoice

from config import VOICE_MODEL, BEEP_FREQUENCY, BEEP_DURATION

from core.console import log


# The voice used to be loaded by a fresh `piper` subprocess on EVERY
# speak() call - a 63MB ONNX model read from disk and initialised each
# time, roughly 0.5-1s before a single sample came out. It's now loaded
# once, and audio goes straight to a sound device stream that stays
# open, so there's no per-phrase startup cost and no `aplay` dependency.
#
# Two worker threads form a small pipeline:
#
#   say(text) -> [text queue] -> synth thread -> [audio queue] -> player
#
# so sentence N+1 is synthesised while sentence N is still playing,
# and the main loop can keep reading the LLM stream in the meantime.


log("TTS", "Loading voice...")

_voice = PiperVoice.load(VOICE_MODEL)

# Read from the voice's own .onnx.json, so swapping voices can never
# again cause sped-up / slowed-down speech from a stale config value.
SAMPLE_RATE = _voice.config.sample_rate


def _synthesize(text):

    # Yields int16 numpy arrays, one per sentence piper detects.

    for chunk in _voice.synthesize(text):

        yield chunk.audio_int16_array


# The first inference call is noticeably slower (ONNX session warmup),
# so pay it now instead of on the first real reply.
for _ in _synthesize("ok"):
    pass


_output = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)

_output.start()


def _make_beep():

    t = np.arange(
        int(SAMPLE_RATE * BEEP_DURATION)
    ) / SAMPLE_RATE

    tone = np.sin(2 * np.pi * BEEP_FREQUENCY * t)

    # short fade in/out so it doesn't click
    fade = min(len(tone) // 4, int(SAMPLE_RATE * 0.01))

    if fade:
        ramp = np.linspace(0.0, 1.0, fade)
        tone[:fade] *= ramp
        tone[-fade:] *= ramp[::-1]

    return (tone * 0.3 * 32767).astype(np.int16)


_BEEP = _make_beep()


_text_queue = queue.Queue()

_audio_queue = queue.Queue()

_first_play = None


def _synth_worker():

    while True:

        text = _text_queue.get()

        try:

            for audio in _synthesize(text):

                _audio_queue.put(audio)

        except Exception as error:

            log("ERROR", "TTS synthesis failed:", error)

        finally:

            _text_queue.task_done()


def _player_worker():

    global _first_play

    while True:

        audio = _audio_queue.get()

        try:

            if _first_play is None:
                _first_play = time.time()

            # write() blocks until the samples are handed to the
            # device, which naturally paces playback.
            _output.write(audio.reshape(-1, 1))

        except Exception as error:

            log("ERROR", "TTS playback failed:", error)

        finally:

            _audio_queue.task_done()


threading.Thread(target=_synth_worker, daemon=True).start()

threading.Thread(target=_player_worker, daemon=True).start()


log("TTS", f"Ready ({SAMPLE_RATE} Hz)")



def say(text):

    # Non-blocking: queue text and return immediately.

    if not text:
        return

    log("TTS", text)

    _text_queue.put(text)


def beep():

    # Non-blocking: queue the pre-rendered wake beep.

    _audio_queue.put(_BEEP)


def wait():

    # Block until everything queued so far has finished playing.
    # Text first (so all audio is queued), then audio.

    _text_queue.join()

    _audio_queue.join()

    # write() returns once samples are buffered, not once they've
    # left the speaker - wait out the device latency too, so the mic
    # flush that follows really does drop our own tail.
    time.sleep(_output.latency)


def speak(text):

    # Blocking convenience wrapper: say it and wait until it's done.

    say(text)

    wait()


def reset_timer():

    global _first_play

    _first_play = None


def first_play_time():

    # Wall-clock time the first audio of the current reply started
    # playing (None if nothing played yet). Used for honest
    # "first audio" timings - measured at the speaker, not when the
    # text was ready.

    return _first_play
