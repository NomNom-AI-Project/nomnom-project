import base64
import cv2
import ollama


from config import VISION_MODEL

from vision.camera import camera



def encode_image(frame):

    success, buffer = cv2.imencode(
        ".jpg",
        frame
    )


    if not success:

        return None


    return base64.b64encode(
        buffer
    ).decode(
        "utf-8"
    )



def analyze(
    question
):

    frame = camera.get_frame()


    if frame is None:

        return (
            "Non riesco ad accedere "
            "alla webcam."
        )



    image = encode_image(
        frame
    )


    if image is None:

        return (
            "Errore acquisizione immagine."
        )



    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role":"user",
                "content":f"""
Guarda questa immagine.

Domanda:
{question}

Rispondi in italiano.
Descrivi solo ciò che vedi.
Non inventare dettagli.
""",
                "images":[
                    image
                ]
            }
        ],
        options={
            "temperature":0.2,
            "num_predict":120
        }
    )


    return response[
        "message"
    ][
        "content"
    ].strip()