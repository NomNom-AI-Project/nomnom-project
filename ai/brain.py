import ollama

from config import BRAIN_MODEL, OLLAMA_KEEP_ALIVE

from core.memory import memory

from core.console import log



def think(
    text,
    extra_context=None
):

    memory.add_user(
        text
    )


    messages = memory.get()


    if extra_context:

        # Augment only the outgoing request for this call. What's
        # stored in memory stays the clean user text - so history
        # doesn't fill up with camera-analysis scaffolding.

        messages = messages[:-1] + [
            {
                "role":"user",
                "content":f"{text}\n\n{extra_context}"
            }
        ]


    response = ollama.chat(
        model=BRAIN_MODEL,
        messages=messages,
        options={
            "temperature":0.3,
            "num_predict":120
        },
        keep_alive=OLLAMA_KEEP_ALIVE
    )


    answer = response[
        "message"
    ][
        "content"
    ].strip()



    memory.add_ai(
        answer
    )


    return answer



def warmup():

    # Force the brain model into memory at startup instead of paying
    # the cold-start load time on the user's first real question.

    try:

        ollama.chat(
            model=BRAIN_MODEL,
            messages=[
                {
                    "role":"user",
                    "content":"ciao"
                }
            ],
            options={
                "num_predict":1
            },
            keep_alive=OLLAMA_KEEP_ALIVE
        )

    except Exception as error:

        log(
            "WARMUP",
            "Failed:",
            error
        )