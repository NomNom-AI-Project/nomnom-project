import ollama

from config import BRAIN_MODEL

from core.memory import memory

from ai.prompts import SYSTEM_PROMPT



def think(
    text
):

    memory.add_user(
        text
    )


    messages = [
        {
            "role":"system",
            "content":SYSTEM_PROMPT
        }
    ]


    messages.extend(
        memory.get()
    )


    response = ollama.chat(
        model=BRAIN_MODEL,
        messages=messages,
        options={
            "temperature":0.3,
            "num_predict":120
        }
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