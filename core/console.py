import sys
import time


# Auto-disable colors when output isn't a real terminal (e.g. piped to
# a log file) so log files don't fill up with raw escape codes.
_USE_COLOR = sys.stdout.isatty()


class Color:

    RESET = "\033[0m"
    BOLD = "\033[1m"

    GRAY = "\033[90m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


# Small glyph + color per AssistantState.value, so the current state is
# readable at a glance. Keyed by plain string (not the enum) so this
# module has zero dependency on core.state - avoids a circular import.
STATE_STYLE = {
    "idle": (Color.GRAY, "o"),
    "listening": (Color.CYAN, "*"),
    "thinking": (Color.YELLOW, "~"),
    "speaking": (Color.GREEN, "^"),
}


TAG_COLOR = {
    "WAKE": Color.MAGENTA,
    "MIC": Color.CYAN,
    "VAD": Color.CYAN,
    "STT": Color.BLUE,
    "AI": Color.GREEN,
    "USER": Color.BLUE,
    "TTS": Color.GREEN,
    "TIME": Color.GRAY,
    "SYSTEM": Color.BOLD,
    "ERROR": Color.RED,
    "CAMERA": Color.MAGENTA,
    "WARMUP": Color.GRAY,
    "CLOUD": Color.YELLOW,
}



def _style(text, color):

    if not _USE_COLOR:
        return text

    return f"{color}{text}{Color.RESET}"



def log(tag, *parts):

    color = TAG_COLOR.get(
        tag,
        Color.RESET
    )

    stamp = time.strftime("%H:%M:%S")

    prefix = _style(
        f"[{stamp}] [{tag}]",
        color
    )

    message = " ".join(
        str(p) for p in parts
    )

    print(f"{prefix} {message}")



def status(state_value):

    color, symbol = STATE_STYLE.get(
        state_value,
        (Color.RESET, "?")
    )

    line = _style(
        f"{symbol} {state_value.upper()}",
        color
    )

    print(line)



def banner(name):

    title = f"{name} AI"

    line = "=" * (len(title) + 8)

    framed = _style(line, Color.CYAN)

    styled_title = _style(title, Color.BOLD)

    print(f"\n{framed}")

    print(f"    {styled_title}")

    print(f"{framed}\n")