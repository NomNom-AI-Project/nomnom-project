import base64
import cv2
import ollama


from config import (
    VISION_MODEL,
    CLOUD_VISION_MODEL,
    OLLAMA_KEEP_ALIVE,
    NUM_CTX,
    VISION_NUM_PREDICT,
    VISION_IMAGE_MAX_SIZE,
    VISION_JPEG_QUALITY,
    ASSISTANT_NAME
)

from vision.camera import camera

from ai.brain import remember

from ai import cloud



# The vision model now answers the user DIRECTLY, in the assistant's
# own voice, instead of describing the scene for a second model to
# rephrase. Previously one camera question cost three sequential model
# calls (intent -> llava -> brain), which on a memory-constrained
# machine also meant Ollama evicting and reloading a 7B model mid
# conversation. It's one call now.

PROMPT = f"""
Sei {ASSISTANT_NAME}, un assistente vocale.

Guardi l'immagine dalla tua webcam e rispondi
direttamente all'utente.

Regole:
- Rispondi in italiano.
- Una o due frasi, adatte alla voce.
- Descrivi solo ciò che vedi davvero.
- Non inventare dettagli.
- Non descrivere l'immagine come "immagine":
  parla come se stessi guardando tu.

Domanda dell'utente:
"""



def encode_image(frame):

    height, width = frame.shape[:2]

    longest = max(height, width)


    # Downscale before encoding: these models resize to ~336px
    # internally anyway, so a full-size frame only costs JPEG encode
    # time and a larger base64 payload.

    if longest > VISION_IMAGE_MAX_SIZE:

        scale = VISION_IMAGE_MAX_SIZE / longest


        frame = cv2.resize(
            frame,
            (
                int(width * scale),
                int(height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )


    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            VISION_JPEG_QUALITY
        ]
    )


    if not success:

        return None


    return base64.b64encode(
        buffer
    ).decode(
        "utf-8"
    )



def _cloud_stream(
    question,
    image
):

    # OpenAI-style image message: the frame travels as a data URI.

    return cloud.stream_chat(
        CLOUD_VISION_MODEL,
        [
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":f"{PROMPT}{question}"
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:image/jpeg;base64,{image}"
                        }
                    }
                ]
            }
        ],
        VISION_NUM_PREDICT,
        0.2
    )



def _local_stream(
    question,
    image
):

    stream = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role":"user",
                "content":f"{PROMPT}{question}",
                "images":[
                    image
                ]
            }
        ],
        options={
            "temperature":0.2,
            "num_predict":VISION_NUM_PREDICT,
            "num_ctx":NUM_CTX
        },
        keep_alive=OLLAMA_KEEP_ALIVE,
        stream=True
    )


    for chunk in stream:

        piece = chunk["message"]["content"]


        if piece:

            yield piece



def analyze_stream(
    question
):

    frame = camera.get_frame()


    if frame is None:

        yield (
            "Non riesco ad accedere "
            "alla webcam."
        )

        return



    image = encode_image(
        frame
    )


    if image is None:

        yield "Errore acquisizione immagine."

        return



    answer = ""


    pieces = cloud.with_fallback(
        lambda: _cloud_stream(question, image),
        lambda: _local_stream(question, image)
    )


    for piece in pieces:

        answer += piece

        yield piece



    # The vision model replies to the user directly, so record the
    # exchange in conversation memory ourselves - otherwise follow-up
    # questions would have no idea what was just discussed.

    remember(
        question,
        answer.strip()
    )



def analyze(
    question
):

    return "".join(
        analyze_stream(
            question
        )
    ).strip()