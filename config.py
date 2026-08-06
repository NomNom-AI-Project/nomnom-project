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

# Better for wakeword + VAD
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
# Assistant
# ==========================

ASSISTANT_NAME = "NomNom"

DEVELOPER_NAME = "Niccolò Bardi"

DEVELOPER_ALIAS = "Badoverse"