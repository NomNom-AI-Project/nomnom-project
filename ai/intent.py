import ollama

from config import BRAIN_MODEL, OLLAMA_KEEP_ALIVE



def needs_vision(
    text
):

    prompt = f"""

Classifica questa frase.

Rispondi SOLO:

VISION

oppure

NORMAL


Frase:

{text}

"""

    result = ollama.chat(
        model=BRAIN_MODEL,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        options={
            "temperature":0,
            "num_predict":3
        },
        keep_alive=OLLAMA_KEEP_ALIVE
    )


    answer = result[
        "message"
    ][
        "content"
    ].strip().upper()



    return answer == "VISION"