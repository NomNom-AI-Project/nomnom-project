from config import (
    MAX_MEMORY_MESSAGES,
    ASSISTANT_NAME,
    DEVELOPER_NAME,
    DEVELOPER_ALIAS
)


SYSTEM_MESSAGE = f"""
Sei {ASSISTANT_NAME},
un assistente AI personale locale.

Regole:

- Rispondi sempre in italiano.
- Sii preciso e naturale.
- Non inventare informazioni.
- Mantieni risposte adatte alla voce.
- Usa il contesto della conversazione.

Il tuo sviluppatore è:
{DEVELOPER_NAME}

Nickname:
{DEVELOPER_ALIAS}
"""


class Memory:


    def __init__(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            }
        ]


    def add_user(
        self,
        text
    ):

        self.messages.append(
            {
                "role":"user",
                "content":text
            }
        )

        self.trim()



    def add_ai(
        self,
        text
    ):

        self.messages.append(
            {
                "role":"assistant",
                "content":text
            }
        )

        self.trim()



    def trim(self):

        if len(self.messages) > MAX_MEMORY_MESSAGES:

            self.messages = [
                self.messages[0]
            ] + self.messages[
                -MAX_MEMORY_MESSAGES:
            ]



    def get(self):

        return self.messages



memory = Memory()