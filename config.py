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
# of the wake beep drain out of the mic buffer. The buffer is flushed
# AFTER this delay, so it only has to cover the beep's echo.
MIC_SETTLE_DELAY = 0.2


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

# How long you have to be quiet before the command counts as finished.
# This is pure dead time on every turn. 0.5s is comfortable for short
# commands; raise it if it cuts you off mid-sentence.
SILENCE_DURATION = 0.5

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

# "auto" uses an NVIDIA GPU via CUDA if one is available, else the CPU.
# On a GPU even "small" transcribes in ~0.1-0.2s.
WHISPER_DEVICE = "auto"

# int8 is the fast choice on CPU; on a GPU "float16" is usually best.
WHISPER_COMPUTE_TYPE = "int8"

# CPU threads for Whisper. ctranslate2 defaults to 4; using all your
# cores is a free speedup on most desktops.
WHISPER_CPU_THREADS = os.cpu_count() or 4

WHISPER_LANGUAGE = "it"



# ==========================
# Backend
# ==========================

# "cloud": answer with a hosted model, and fall back to the local
#          Ollama models automatically if it fails or you're offline.
# "local": Ollama only - nothing leaves the machine.
#
# Cloud mode also falls back to local when no API key is set, so it's
# safe as the default.
BACKEND = "cloud"

# Any OpenAI-compatible API works (Groq, OpenRouter, Together, OpenAI,
# a vLLM server on your LAN...). Defaults are for Groq.
CLOUD_BASE_URL = "https://api.groq.com/openai/v1"

# Name of the environment variable holding the API key. Put it in a
# .env file next to main.py (already gitignored) - see .env.example.
CLOUD_API_KEY_ENV = "NOMNOM_API_KEY"

# One model for both chat and camera questions: it accepts images and
# can have its reasoning switched off, so it starts answering at once.
# Groq retires models often - if requests start failing with a
# "model decommissioned" error, check console.groq.com/docs/deprecations
CLOUD_MODEL = "qwen/qwen3.8-27b"

CLOUD_VISION_MODEL = "qwen/qwen3.8-27b"

# Extra request fields, provider-specific. "reasoning_effort": "none"
# stops the model "thinking" before it answers - essential for voice,
# where those hidden tokens are pure waiting. Remove it for providers
# or models that don't support it.
CLOUD_EXTRA_PARAMS = {
    "reasoning_effort": "none"
}

# Seconds to wait for the connection or the next piece of the reply
# before giving up and answering locally instead.
CLOUD_TIMEOUT = 5

# After a cloud failure, use local models for this many seconds before
# trying the cloud again, so being offline doesn't cost a timeout on
# every turn.
CLOUD_RETRY_AFTER = 60



# ==========================
# Local models (Ollama)
# ==========================

# In cloud mode these are the fallback.

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
#
# In cloud mode only the local brain is warmed up (so the fallback is
# instant); the local vision model is skipped to save RAM.
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

# The voice's sample rate is read from its .onnx.json automatically.

# Short tone played after the wake word, instead of a spoken greeting.
# It's pre-rendered in memory, so it plays instantly - a spoken
# "Eccomi, dimmi." cost ~1s of synthesis + speech on every activation.
BEEP_FREQUENCY = 880

BEEP_DURATION = 0.12



# ==========================
# Assistant
# ==========================

ASSISTANT_NAME = "NomNom"

DEVELOPER_NAME = "Niccolò Bardi"

DEVELOPER_ALIAS = "Badoverse"