import sounddevice as sd
import numpy as np
import queue

from config import SAMPLE_RATE


BLOCK_SIZE = 512


audio_queue = queue.Queue(
    maxsize=100
)


stream = None



def callback(
    indata,
    frames,
    time,
    status
):

    audio = (
        indata[:,0]
        .copy()
        .astype(np.int16)
    )


    try:

        audio_queue.put_nowait(
            audio
        )

    except queue.Full:

        pass




def start_microphone():

    global stream


    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        callback=callback
    )


    stream.start()

    print("[MIC] Ready")




def get_audio():

    return audio_queue.get()




def clear_buffer():

    removed = 0

    while not audio_queue.empty():

        try:

            audio_queue.get_nowait()

            removed += 1

        except:

            break


    print(
        f"[MIC] Cleared {removed} frames"
    )