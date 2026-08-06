from enum import Enum

from core.console import status as render_status


class AssistantState(Enum):

    IDLE = "idle"

    LISTENING = "listening"

    THINKING = "thinking"

    SPEAKING = "speaking"



current_state = AssistantState.IDLE



def set_state(state):

    global current_state

    current_state = state

    render_status(
        state.value
    )



def get_state():

    return current_state