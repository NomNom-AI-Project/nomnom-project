import ollama


MODEL = "qwen2.5:3b"



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
        model=MODEL,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        options={
            "temperature":0,
            "num_predict":3
        }
    )


    answer = result[
        "message"
    ][
        "content"
    ].strip().upper()



    return answer == "VISION"