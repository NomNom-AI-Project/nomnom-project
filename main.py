from dotenv import load_dotenv

load_dotenv()

import time
import tempfile

import numpy as np
import soundfile as sf


from config import (
    SAMPLE_RATE,
    AUDIO_BLOCK_SIZE,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION,
    MAX_SPEECH_DURATION,
    COMMAND_WAIT_TIMEOUT,
    ASSISTANT_NAME
)

from audio.microphone import (
    start_microphone,
    get_audio,
    clear_buffer
)

from audio.wakeword import detect

from audio.vad import contains_speech, reset as reset_vad

from audio.stt import transcribe

from audio.tts import speak


from ai.brain import think, warmup

from ai.intent import needs_vision

from vision.camera import camera

from vision.analyzer import analyze


from core.state import (
    set_state,
    AssistantState
)

from core.console import log, banner


# Derive frame-based thresholds from the configured durations instead
# of hardcoding magic numbers tied to one specific block size.

FRAME_SECONDS = AUDIO_BLOCK_SIZE / SAMPLE_RATE

SILENCE_LIMIT_FRAMES = max(1, round(SILENCE_DURATION / FRAME_SECONDS))

MIN_SPEECH_FRAMES = max(1, round(MIN_SPEECH_DURATION / FRAME_SECONDS))

MAX_SPEECH_FRAMES = max(1, round(MAX_SPEECH_DURATION / FRAME_SECONDS))



def save_audio(chunks):

    filename = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name


    audio = np.concatenate(
        chunks
    )


    audio_float = (
        audio.astype(
            np.float32
        )
        /
        32768.0
    )


    sf.write(
        filename,
        audio_float,
        SAMPLE_RATE
    )


    return filename




def listen_for_command():

    log("MIC", "Preparing...")


    clear_buffer()


    time.sleep(
        0.8
    )


    log("MIC", "Listening...")


    reset_vad()


    chunks = []

    speech_frames = 0

    silence_frames = 0

    speech_started = False


    wait_deadline = time.time() + COMMAND_WAIT_TIMEOUT



    while True:

        if (
            not speech_started
            and time.time() > wait_deadline
        ):

            log("MIC", "No speech")

            return None


        audio = get_audio()


        speaking = contains_speech(
            audio
        )


        if speaking:

            speech_started = True

            speech_frames += 1

            silence_frames = 0


        elif speech_started:

            silence_frames += 1



        if not speech_started:

            continue


        chunks.append(
            audio
        )


        if len(chunks) >= MAX_SPEECH_FRAMES:

            log("MIC", "Max duration reached")

            break


        if silence_frames > SILENCE_LIMIT_FRAMES:

            if speech_frames >= MIN_SPEECH_FRAMES:

                break


            # too short to be a real command (cough, click, door) -
            # reset and keep waiting instead of sending noise to STT

            chunks = []

            speech_started = False

            speech_frames = 0

            silence_frames = 0



    return save_audio(
        chunks
    )




def process_command(text):

    if needs_vision(
        text
    ):

        log("AI", "Vision requested")


        vision_result = analyze(
            text
        )


        context = (
            "Informazioni dalla fotocamera:\n"
            f"{vision_result}\n\n"
            "Rispondi usando queste informazioni visive."
        )


        return think(
            text,
            extra_context=context
        )


    return think(
        text
    )




def main():

    banner(ASSISTANT_NAME)


    start_microphone()


    camera.start()


    log("SYSTEM", "Warming up models...")


    warmup()


    log("SYSTEM", "Ready")



    while True:

        try:

            set_state(
                AssistantState.IDLE
            )


            log("WAKE", "Waiting...")


            # Wakeword loop

            while True:

                audio = get_audio()


                if detect(
                    audio
                ):

                    break



            log("WAKE", "Activated")


            set_state(
                AssistantState.SPEAKING
            )


            clear_buffer()


            speak(
                "Eccomi, dimmi."
            )


            set_state(
                AssistantState.LISTENING
            )


            audio_file = listen_for_command()



            if audio_file is None:

                continue



            set_state(
                AssistantState.THINKING
            )


            text = transcribe(
                audio_file
            )


            log("USER", text)



            if not text:

                continue



            start = time.time()


            try:

                answer = process_command(
                    text
                )


            except Exception as error:

                log(
                    "ERROR",
                    "process_command:",
                    error
                )

                answer = (
                    "Scusa, ho avuto un problema "
                    "a elaborare la richiesta."
                )


            elapsed = (
                time.time()
                -
                start
            )



            log("AI", answer)

            log("TIME", f"{elapsed:.2f}s")



            set_state(
                AssistantState.SPEAKING
            )


            speak(
                answer
            )



            # remove leftover TTS audio

            clear_buffer()



        except KeyboardInterrupt:


            log("SYSTEM", "Stopped")


            camera.stop()

            break



        except Exception as error:


            log("ERROR", error)


            time.sleep(
                1
            )





if __name__ == "__main__":

    main()