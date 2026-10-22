from dotenv import load_dotenv

load_dotenv()

import time

import numpy as np


from config import (
    SAMPLE_RATE,
    AUDIO_BLOCK_SIZE,
    SILENCE_DURATION,
    MIN_SPEECH_DURATION,
    MAX_SPEECH_DURATION,
    COMMAND_WAIT_TIMEOUT,
    MIC_SETTLE_DELAY,
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

from audio.tts import (
    speak,
    say,
    beep,
    wait as wait_for_speech,
    reset_timer as reset_audio_timer,
    first_play_time
)


from ai.brain import think_stream, warmup

from ai.intent import needs_vision

from vision.camera import camera

from vision.analyzer import analyze_stream


from core.state import (
    set_state,
    AssistantState
)

from core.console import log, banner

from core.text import stream_sentences, clean_for_speech


# Derive frame-based thresholds from the configured durations instead
# of hardcoding magic numbers tied to one specific block size.

FRAME_SECONDS = AUDIO_BLOCK_SIZE / SAMPLE_RATE

SILENCE_LIMIT_FRAMES = max(1, round(SILENCE_DURATION / FRAME_SECONDS))

MIN_SPEECH_FRAMES = max(1, round(MIN_SPEECH_DURATION / FRAME_SECONDS))

MAX_SPEECH_FRAMES = max(1, round(MAX_SPEECH_DURATION / FRAME_SECONDS))



def build_waveform(chunks):

    # Hand Whisper a float32 waveform directly instead of writing a
    # temp WAV and reading it back.

    audio = np.concatenate(
        chunks
    )


    return (
        audio.astype(np.float32)
        /
        32768.0
    )




def listen_for_command():

    # Settle first, THEN flush. The tail of our own greeting is still
    # arriving from the sound card during this delay, so clearing
    # afterwards is what actually drops it - which is why the delay
    # can now be much shorter than before.

    time.sleep(
        MIC_SETTLE_DELAY
    )


    clear_buffer()

    reset_vad()


    log("MIC", "Listening...")


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



    return build_waveform(
        chunks
    )




def respond(text):

    # Returns a stream of text pieces. Vision questions are answered
    # by the vision model directly - no second pass through the brain
    # model, which used to double the wait and force a model swap.

    if needs_vision(
        text
    ):

        log("AI", "Vision requested")


        return analyze_stream(
            text
        )


    return think_stream(
        text
    )




def speak_response(pieces):

    # Each sentence is queued for speech the moment it's complete.
    # say() doesn't block, so the LLM keeps streaming while earlier
    # sentences are synthesised and played in the background.

    start = time.time()

    reset_audio_timer()

    spoken = []


    for sentence in stream_sentences(
        pieces
    ):

        clean = clean_for_speech(
            sentence
        )


        if not clean:
            continue


        spoken.append(clean)


        say(clean)



    wait_for_speech()


    # Measured when sound actually reached the speaker, not when the
    # text was ready - so it includes synthesis time too.

    played = first_play_time()

    if played is not None:

        log(
            "TIME",
            f"first audio in {played - start:.2f}s"
        )


    return " ".join(spoken)




def main():

    banner(ASSISTANT_NAME)


    start_microphone()


    camera.start()


    # Non-blocking: models load in background threads while the user
    # is still walking up to say the wake word.

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


            # Instant pre-rendered beep instead of a spoken greeting.
            # listen_for_command() flushes the mic after it, so the
            # beep itself never reaches the VAD.

            beep()

            wait_for_speech()


            set_state(
                AssistantState.LISTENING
            )


            waveform = listen_for_command()



            if waveform is None:

                continue



            set_state(
                AssistantState.THINKING
            )


            heard = time.time()


            text = transcribe(
                waveform
            )


            log(
                "TIME",
                f"transcribed in {time.time() - heard:.2f}s"
            )


            log("USER", text)



            if not text:

                continue



            start = time.time()


            set_state(
                AssistantState.SPEAKING
            )


            try:

                answer = speak_response(
                    respond(text)
                )


            except Exception as error:

                log(
                    "ERROR",
                    "respond:",
                    error
                )

                answer = ""

                speak(
                    "Scusa, ho avuto un problema "
                    "a elaborare la richiesta."
                )



            log("AI", answer)


            # The number that matters: from the moment your command was
            # captured to the moment it started answering.

            played = first_play_time()

            if played is not None:

                log(
                    "TIME",
                    f"command end -> first audio {played - heard:.2f}s"
                )

            log(
                "TIME",
                f"total {time.time() - start:.2f}s"
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