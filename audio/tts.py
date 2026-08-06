import subprocess
import tempfile
import os

from config import VOICE_MODEL



def speak(
    text
):

    if not text:
        return


    print(
        "[TTS]",
        text
    )


    file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )


    filename = file.name

    file.close()



    try:

        subprocess.run(
            [
                "piper",
                "--model",
                VOICE_MODEL,
                "--output_file",
                filename
            ],
            input=text.encode(
                "utf-8"
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


        subprocess.run(
            [
                "aplay",
                filename
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


    finally:

        try:

            os.remove(
                filename
            )

        except:

            pass