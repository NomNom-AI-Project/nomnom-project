from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS
)

from core.console import log


log("STT", "Loading...")


model = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
    cpu_threads=WHISPER_CPU_THREADS
)


log("STT", "Ready")



def transcribe(
    audio
):

    # `audio` is a float32 numpy waveform passed straight from the
    # mic. faster-whisper accepts a raw waveform, so we no longer
    # write a temp WAV to disk and read it back - that saved a full
    # encode/write/read/decode round trip per command, and also fixed
    # a leak, since those temp files were created with delete=False
    # and never cleaned up.

    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE,
        beam_size=1,

        # Our own VAD already trimmed this clip to just the speech,
        # so Whisper doesn't need to search for it again.
        vad_filter=False,

        # We transcribe each command independently; carrying context
        # between them costs time and invites hallucinated
        # continuations.
        condition_on_previous_text=False,

        # We only want the text - skipping timestamp alignment saves
        # a little decoding work.
        without_timestamps=True
    )


    text = ""


    for segment in segments:

        text += segment.text


    return text.strip()