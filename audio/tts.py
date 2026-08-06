import subprocess

from config import VOICE_MODEL, PIPER_SAMPLE_RATE

from core.console import log



def speak(
    text
):

    if not text:
        return


    log("TTS", text)


    try:

        piper = subprocess.Popen(
            [
                "piper",
                "--model",
                VOICE_MODEL,
                "--output-raw"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )


        # piper.stdin / piper.stdout are typed Optional[IO[bytes]] by
        # subprocess's stubs, since they're None unless PIPE was
        # passed - which we did. Narrow into locals once so every use
        # below is provably non-None, instead of re-checking each time.

        piper_stdin = piper.stdin
        piper_stdout = piper.stdout

        assert piper_stdin is not None
        assert piper_stdout is not None


        player = subprocess.Popen(
            [
                "aplay",
                "-r", str(PIPER_SAMPLE_RATE),
                "-f", "S16_LE",
                "-t", "raw",
                "-"
            ],
            stdin=piper_stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


        # let piper receive SIGPIPE if aplay exits early, instead of
        # blocking on a full pipe buffer

        piper_stdout.close()


        piper_stdin.write(
            text.encode("utf-8")
        )

        piper_stdin.close()


        player.wait()

        piper.wait()


    except Exception as error:

        log(
            "ERROR",
            "TTS failed:",
            error
        )