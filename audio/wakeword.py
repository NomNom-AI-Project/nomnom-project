import os
import numpy as np
import openwakeword

from openwakeword.model import Model

from config import (
    WAKEWORD_MODEL,
    WAKE_THRESHOLD
)


print("[WAKE] Loading...")


available_models = openwakeword.get_pretrained_model_paths()


model_path = None


for path in available_models:

    if WAKEWORD_MODEL.lower() in path.lower():

        model_path = path
        break



if model_path is None:

    raise RuntimeError(
        f"Wake word model '{WAKEWORD_MODEL}' not found.\n"
        f"Available models:\n{available_models}"
    )



print(
    "[WAKE] Model:",
    os.path.basename(model_path)
)



wake_model = Model(
    wakeword_model_paths=[
        model_path
    ]
)



print("[WAKE] Ready")



def detect(
    audio: np.ndarray
):

    if audio.dtype != np.int16:

        audio = audio.astype(
            np.int16
        )


    prediction = wake_model.predict(
        audio
    )


    if isinstance(
        prediction,
        tuple
    ):

        prediction = prediction[0]


    if not isinstance(
        prediction,
        dict
    ):

        return False



    best_name = None
    best_score = 0.0


    for name, score in prediction.items():

        try:

            score = float(score)

        except:

            continue


        if score > best_score:

            best_score = score
            best_name = name



    if best_score >= WAKE_THRESHOLD:

        print(
            f"[WAKE] Detected {best_name}: {best_score:.2f}"
        )


        wake_model.reset()


        return True



    return False