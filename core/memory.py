from config import MAX_MEMORY_MESSAGES

from ai.prompts import SYSTEM_PROMPT


class Memory:


    def __init__(self):

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
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