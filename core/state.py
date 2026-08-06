from enum import Enum


class AssistantState(Enum):

    IDLE = "idle"

    LISTENING = "listening"

    THINKING = "thinking"

    SPEAKING = "speaking"



current_state = AssistantState.IDLE



def set_state(state):

    global current_state

    current_state = state



def get_state():

    return current_state