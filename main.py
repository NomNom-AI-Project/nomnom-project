from dotenv import load_dotenv

load_dotenv()

import time
import tempfile

import numpy as np
import soundfile as sf


from audio.microphone import (
    start_microphone,
    get_audio,
    clear_buffer
)

from audio.wakeword import detect

from audio.vad import contains_speech

from audio.stt import transcribe

from audio.tts import speak


from ai.brain import think

from ai.intent import needs_vision

from vision.camera import camera

from vision.analyzer import analyze


from core.state import (
    set_state,
    AssistantState
)



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
        16000
    )


    return filename




def listen_for_command():

    print("[MIC] Preparing...")


    clear_buffer()


    time.sleep(
        0.8
    )


    print("[MIC] Listening...")


    chunks = []

    speech_started = False

    silence_frames = 0


    timeout = time.time() + 15



    while time.time() < timeout:


        audio = get_audio()



        if contains_speech(
            audio
        ):

            if not speech_started:

                print(
                    "[MIC] Speech detected"
                )

                speech_started = True


            silence_frames = 0


        elif speech_started:

            silence_frames += 1



        if speech_started:

            chunks.append(
                audio
            )



        # stop after silence

        if (
            speech_started
            and silence_frames > 25
        ):

            break



    if not speech_started:

        print(
            "[MIC] No speech"
        )

        return None



    return save_audio(
        chunks
    )




def process_command(text):


    if needs_vision(
        text
    ):

        print(
            "[AI] Vision requested"
        )


        vision_result = analyze(
            text
        )


        prompt = f"""
User request:

{text}


Camera analysis:

{vision_result}


Answer using the camera information.
"""


        return think(
            prompt
        )



    return think(
        text
    )





def main():

    print(
        """
=========================
        NomNom AI
=========================
"""
    )


    start_microphone()


    camera.start()


    print(
        "[SYSTEM] Ready"
    )



    while True:

        try:


            set_state(
                AssistantState.IDLE
            )


            print(
                "[WAKE] Waiting..."
            )


            # Wakeword loop

            while True:

                audio = get_audio()


                if detect(
                    audio
                ):

                    break



            print(
                "[WAKE] Activated"
            )


            set_state(
                AssistantState.SPEAKING
            )


            clear_buffer()


            speak(
                "Eccomi, dimmi."
            )


            time.sleep(
                1
            )


            clear_buffer()



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


            print(
                "[USER]",
                text
            )



            if not text:

                continue



            start = time.time()


            answer = process_command(
                text
            )


            elapsed = (
                time.time()
                -
                start
            )



            print(
                "[AI]",
                answer
            )


            print(
                f"[TIME] {elapsed:.2f}s"
            )



            set_state(
                AssistantState.SPEAKING
            )


            speak(
                answer
            )



            # remove leftover TTS audio

            clear_buffer()



        except KeyboardInterrupt:


            print(
                "\n[SYSTEM] Stopped"
            )


            camera.stop()

            break



        except Exception as error:


            print(
                "[ERROR]",
                error
            )


            time.sleep(
                1
            )





if __name__ == "__main__":

    main()