from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    WHISPER_LANGUAGE
)

from core.console import log


log("STT", "Loading...")


model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)


log("STT", "Ready")



def transcribe(
    audio_file
):

    segments, info = model.transcribe(
        audio_file,
        language=WHISPER_LANGUAGE,
        beam_size=1
    )


    text = ""


    for segment in segments:

        text += segment.text


    return text.strip()