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
# for. audio/vad.py re-buffers this into the 512-sample windows Silero
# requires, so one block size can serve both.
AUDIO_BLOCK_SIZE = 1280

# Pause after the greeting before we start listening, to let the tail
# of our own TTS drain out of the mic buffer. Was 0.8s; the buffer is
# now flushed AFTER this delay rather than before, so a shorter settle
# is enough and it cuts dead air off every single interaction.
MIC_SETTLE_DELAY = 0.3


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

# "tiny" is roughly 2x faster than "base" with some accuracy loss;
# "small" is more accurate but noticeably slower on CPU. If STT is
# your bottleneck, this is the dial to turn.
WHISPER_MODEL = "base"

WHISPER_LANGUAGE = "it"



# ==========================
# AI models
# ==========================

BRAIN_MODEL = "qwen2.5:3b"

# llava:7b is heavy on CPU and is usually the single slowest step in a
# camera question. If vision is still too slow, swapping this for a
# smaller multimodal model is by far the biggest win available -
# "moondream" (~1.8B) or "llava-phi3" (~3.8B) are drop-in alternatives
# via `ollama pull`, typically 2-4x faster at some accuracy cost.
VISION_MODEL = "llava:7b"

# Keep models loaded in Ollama between turns so a pause doesn't force
# a slow reload on the next request.
OLLAMA_KEEP_ALIVE = "30m"

# Context window. Smaller = faster prompt processing. This needs to
# comfortably fit the system prompt plus MAX_MEMORY_MESSAGES of
# history; 2048 is plenty for short voice exchanges.
NUM_CTX = 2048

# Reply length caps. Voice answers should be short anyway, and every
# extra token is extra time before the user hears the end.
BRAIN_NUM_PREDICT = 120

VISION_NUM_PREDICT = 80

# Load models at startup (in a background thread, so startup isn't
# blocked) instead of paying a cold load on first use. Vision warmup
# costs RAM for a model you may never use in a given session - set to
# False if memory is tight on your machine.
WARMUP_BRAIN = True

WARMUP_VISION = True



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

# LLaVA-style models downscale to 336x336 internally anyway, so
# sending a full 640x480 frame just costs JPEG encode time and a
# bigger base64 payload for no accuracy gain.
VISION_IMAGE_MAX_SIZE = 448

VISION_JPEG_QUALITY = 80



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