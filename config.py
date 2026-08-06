import os


# ==========================
# Paths
# ==========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VOICE_MODEL = os.path.join(
    BASE_DIR,
    "voices",
    "it_IT-paola-medium.onnx"
)


# ==========================
# Audio
# ==========================

SAMPLE_RATE = 16000

CHANNELS = 1

# 1280 samples = 80ms @ 16kHz, the frame size openWakeWord is tuned
# for. Silero VAD handles this length fine too, so one block size
# can serve both wakeword detection and VAD.
AUDIO_BLOCK_SIZE = 1280


# ==========================
# Wake word
# ==========================

# Search string inside openwakeword models
WAKEWORD_MODEL = "hey_jarvis"

WAKE_THRESHOLD = 0.5



# ==========================
# Voice activity detection
# ==========================

VAD_THRESHOLD = 0.25


MIN_SPEECH_DURATION = 0.3

MAX_SPEECH_DURATION = 10.0

SILENCE_DURATION = 0.8

# Seconds to wait for the user to start talking after the wake word
# before giving up and going back to idle.
COMMAND_WAIT_TIMEOUT = 15



# ==========================
# Speech recognition
# ==========================

WHISPER_MODEL = "base"

WHISPER_LANGUAGE = "it"



# ==========================
# AI models
# ==========================

BRAIN_MODEL = "qwen2.5:3b"

VISION_MODEL = "llava:7b"

# Keep a model loaded in Ollama between turns so a pause (thinking,
# listening, sitting idle) doesn't force a slow reload on the next
# request.
OLLAMA_KEEP_ALIVE = "30m"



# ==========================
# Memory
# ==========================

MAX_MEMORY_MESSAGES = 10



# ==========================
# Camera
# ==========================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640

CAMERA_HEIGHT = 480



# ==========================
# Text-to-speech
# ==========================

# Sample rate Piper streams raw PCM at. Medium-quality voices are
# typically 22050 Hz - if you swap VOICE_MODEL, check the matching
# .onnx.json ("sample_rate" field) and update this. A mismatch makes
# speech sound sped up or slowed down, not garbled, so it's easy to
# miss.
PIPER_SAMPLE_RATE = 22050



# ==========================
# Assistant
# ==========================

ASSISTANT_NAME = "NomNom"

DEVELOPER_NAME = "Niccolò Bardi"

DEVELOPER_ALIAS = "Badoverse"