import threading

import ollama

from config import (
    CLOUD_MODEL,
    BRAIN_MODEL,
    VISION_MODEL,
    OLLAMA_KEEP_ALIVE,
    NUM_CTX,
    BRAIN_NUM_PREDICT,
    WARMUP_BRAIN,
    WARMUP_VISION
)

from core.memory import memory

from ai import cloud

from core.console import log



def think_stream(
    text,
    extra_context=None
):

    # Streams the reply piece by piece as the model generates it, so
    # the caller can start speaking the first sentence instead of
    # waiting for the whole answer to finish. This is the single
    # biggest improvement to *perceived* response time.

    # The exchange is only written to memory once the reply has been
    # fully generated. Previously the user message was stored first, so
    # an Ollama error mid-stream left two user turns in a row in the
    # history, which confuses the model on the next request.

    content = (
        f"{text}\n\n{extra_context}"
        if extra_context
        else text
    )


    messages = memory.get() + [
        {
            "role":"user",
            "content":content
        }
    ]


    answer = ""


    pieces = cloud.with_fallback(
        lambda: cloud.stream_chat(
            CLOUD_MODEL,
            messages,
            BRAIN_NUM_PREDICT,
            0.3
        ),
        lambda: _local_stream(
            messages
        )
    )


    for piece in pieces:

        answer += piece

        yield piece



    # Only the clean user text is stored, so history doesn't fill up
    # with camera-analysis scaffolding.

    remember(
        text,
        answer.strip()
    )



def _local_stream(
    messages
):

    stream = ollama.chat(
        model=BRAIN_MODEL,
        messages=messages,
        options={
            "temperature":0.3,
            "num_predict":BRAIN_NUM_PREDICT,
            "num_ctx":NUM_CTX
        },
        keep_alive=OLLAMA_KEEP_ALIVE,
        stream=True
    )


    for chunk in stream:

        piece = chunk["message"]["content"]


        if piece:

            yield piece



def think(
    text,
    extra_context=None
):

    # Non-streaming convenience wrapper, kept for any caller that just
    # wants the finished string.

    return "".join(
        think_stream(
            text,
            extra_context
        )
    ).strip()



def remember(
    user_text,
    ai_text
):

    # Stores one finished exchange. Also used by the vision path,
    # which answers from the vision model directly.

    memory.add_user(user_text)

    memory.add_ai(ai_text)



def _load(
    model_name
):

    try:

        ollama.chat(
            model=model_name,
            messages=[
                {
                    "role":"user",
                    "content":"ok"
                }
            ],
            options={
                "num_predict":1
            },
            keep_alive=OLLAMA_KEEP_ALIVE
        )

        log("WARMUP", f"{model_name} ready")


    except Exception as error:

        log(
            "WARMUP",
            f"{model_name} failed:",
            error
        )



def warmup():

    # Load models in the background so startup isn't blocked. By the
    # time the user finishes saying their first command, the model is
    # usually already resident - which removes the cold-load stall
    # from the first (and, for vision, often the only) request.

    cloud.warmup()


    targets = []


    if WARMUP_BRAIN:

        targets.append(BRAIN_MODEL)


    # In cloud mode the local vision model is only a rarely used
    # fallback - not worth keeping several GB of RAM resident for.

    if WARMUP_VISION and not cloud.enabled():

        targets.append(VISION_MODEL)


    for model_name in targets:

        threading.Thread(
            target=_load,
            args=(model_name,),
            daemon=True
        ).start()