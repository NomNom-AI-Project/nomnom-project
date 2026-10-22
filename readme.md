# NomNom

A voice assistant in Italian. Wake word → speech recognition → LLM → spoken reply, with optional webcam vision.

By default the answers come from a hosted model (fast and much smarter), with the local Ollama models as an automatic fallback when the cloud fails or you're offline. Wake word, voice detection, speech recognition and speech synthesis always run locally. Set `BACKEND = "local"` and nothing leaves the machine at all.

Built by **Niccolò Bardi** (Badoverse).

---

## How it works

```
microphone ──► wake word ──► VAD ──► Whisper ──► intent
   (sounddevice)  (openWakeWord)  (Silero)   (faster-whisper)   │
                                                                │
                                        ┌───────────────────────┴──────┐
                                        │                              │
                                   normal question              camera question
                                        │                              │
                              cloud model                    cloud model + frame
                         (fallback: qwen2.5:3b)        (fallback: llava:7b)
                                        │                              │
                                        └───────────────┬──────────────┘
                                                        │
                                              Piper TTS (loaded once) ──► speakers
```

The assistant idles until it hears the wake word, beeps, records until you stop talking, then answers out loud. Each sentence starts playing as soon as it's complete, while the rest is still being generated; the next sentence is synthesised while the current one plays.

---

## Requirements

**System packages** (Linux) — PortAudio is used for both the microphone and the speakers:

```bash
# Fedora
sudo dnf install python3 python3-pip portaudio

# Ubuntu / Debian
sudo apt install python3 python3-pip libportaudio2
```

**Ollama** install from [ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull qwen2.5:3b
ollama pull llava:7b
```

**Piper** — the TTS engine (`piper-tts` 1.3 or newer, installed from `req.txt` and used as a Python library), plus an Italian voice. The project expects the voice at:

```
voices/it_IT-paola-medium.onnx
```

Piper voices ship as a pair: the `.onnx` file and a matching `.onnx.json`. Keep both together in `voices/`.

**Python packages**:

```bash
pip install -r req.txt
```

On newer Ubuntu/Debian you may need `pip install -r req.txt --break-system-packages`, or use a virtualenv.

**Cloud API key** (optional, but it's the default backend) — get a key from your provider (Groq by default: [console.groq.com/keys](https://console.groq.com/keys)), then:

```bash
cp .env.example .env
# edit .env and set NOMNOM_API_KEY=...
```

`.env` is gitignored. Without a key, NomNom simply runs on the local models.

---

## Running

From the project root:

```bash
python3 main.py
```

Wait for `[SYSTEM] Ready`, then say **"Hey Jarvis"**. After the beep, ask your question.

Stop it with `Ctrl+C`.

The console shows the current state as a coloured glyph:

| Glyph | State | Meaning |
|-------|-------|---------|
| `o` | IDLE | Waiting for the wake word |
| `*` | LISTENING | Recording your command |
| `~` | THINKING | Transcribing / generating |
| `^` | SPEAKING | Talking back to you |

---

## Asking about the camera

Questions containing camera-related words are routed to the vision model automatically, using the current webcam frame:

> "Cosa vedi?"
> "Descrivi cosa c'è davanti a te"
> "Di che colore è questo oggetto?"
> "Cosa sto indossando?"

Routing is keyword-based, defined in `VISION_WORDS` at the top of `ai/intent.py`. If a phrasing you use isn't recognised, add it to that list the file is meant to be edited.

---

## Configuration

Everything is in `config.py`. The values worth knowing:

### Backend

| Setting | Default | Notes |
|---|---|---|
| `BACKEND` | `"cloud"` | `"cloud"` = hosted model with local fallback. `"local"` = Ollama only, fully offline. |
| `CLOUD_BASE_URL` | Groq | Any OpenAI-compatible API (OpenRouter, Together, OpenAI, vLLM on your LAN...). |
| `CLOUD_API_KEY_ENV` | `"NOMNOM_API_KEY"` | Environment variable (or `.env` entry) holding the key. |
| `CLOUD_MODEL` / `CLOUD_VISION_MODEL` | `"qwen/qwen3.8-27b"` | Handles both text and images. |
| `CLOUD_EXTRA_PARAMS` | `{"reasoning_effort": "none"}` | Turns off the model's hidden "thinking" so it answers immediately. Remove for providers that don't support it. |
| `CLOUD_TIMEOUT` | `5` | Seconds before giving up on the cloud and answering locally. |
| `CLOUD_RETRY_AFTER` | `60` | After a cloud failure, use local models for this long before retrying. |

**How the fallback works:** if the cloud fails *before* saying anything (offline, bad key, retired model, timeout), the same question goes to the local model, so you always get an answer. If it fails midway through a reply, the part already spoken is kept rather than starting over locally. The `[CLOUD]` log lines show when and why it fell back.

### Speed

| Setting | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | `"tiny"` is ~2x faster, less accurate. `"small"` is slower, more accurate. |
| `WHISPER_DEVICE` | `"auto"` | Uses an NVIDIA GPU (CUDA) when available, otherwise the CPU. |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | Best on CPU; `"float16"` is usually best on a GPU. |
| `WHISPER_CPU_THREADS` | all cores | CPU threads for transcription. |
| `BRAIN_MODEL` | `"qwen2.5:3b"` | The conversational model. |
| `VISION_MODEL` | `"llava:7b"` | **Usually the slowest step.** `moondream` (~1.8B) or `llava-phi3` (~3.8B) are 2-4x faster. |
| `NUM_CTX` | `2048` | Context window. Smaller = faster prompt processing. |
| `BRAIN_NUM_PREDICT` | `120` | Max reply length. Shorter = sooner finished. |
| `VISION_NUM_PREDICT` | `80` | Same, for camera answers. |
| `WARMUP_VISION` | `True` | Preloads the vision model at startup. Set `False` if RAM is tight. |
| `OLLAMA_KEEP_ALIVE` | `"30m"` | How long Ollama keeps models resident between turns. |

### Listening

| Setting | Default | Notes |
|---|---|---|
| `WAKEWORD_MODEL` | `"hey_jarvis"` | Substring matched against openWakeWord's bundled models. |
| `WAKE_THRESHOLD` | `0.5` | Lower = triggers more easily (and more false positives). |
| `VAD_THRESHOLD` | `0.25` | Lower = more sensitive to quiet speech. |
| `SILENCE_DURATION` | `0.5` | Seconds of silence before your command is considered finished. |
| `MIN_SPEECH_DURATION` | `0.3` | Bursts shorter than this are discarded as noise (coughs, door slams). |
| `MAX_SPEECH_DURATION` | `10.0` | Hard cap on a single command. |
| `COMMAND_WAIT_TIMEOUT` | `15` | How long to wait for you to start talking before giving up. |
| `MIC_SETTLE_DELAY` | `0.2` | Pause after the beep so its echo isn't recorded. |
| `BEEP_FREQUENCY` / `BEEP_DURATION` | `880` / `0.12` | The wake beep. |

### Identity

`ASSISTANT_NAME`, `DEVELOPER_NAME`, `DEVELOPER_ALIAS` feed the system prompt in `ai/prompts.py`.

---

## Project structure

```
main.py                 orchestration loop
config.py               all settings
req.txt                 Python dependencies

ai/
  brain.py              LLM calls (streaming), memory wiring, model warmup
  cloud.py              OpenAI-compatible streaming client + local fallback
  intent.py             routes camera questions vs normal questions
  prompts.py            system prompt (single source of truth)

audio/
  microphone.py         input stream + frame queue
  wakeword.py           openWakeWord detection
  vad.py                Silero voice activity detection
  stt.py                faster-whisper transcription
  tts.py                Piper (loaded once) → sound device, background pipeline

vision/
  camera.py             background webcam capture thread
  analyzer.py           frame encoding + vision model

core/
  console.py            coloured, timestamped logging
  state.py              assistant state + status glyph
  memory.py             conversation history
  text.py               sentence streaming + speech sanitising
  events.py             event bus (unused, available)

voices/                 Piper voice models (not included)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'core.console'`**
Run from the project root the folder directly containing `main.py`. If your editor flags it but the program runs fine, that's the editor's import root, not a real error; open the project folder itself as the workspace.

**It hears the wake word but never hears the command**
Check `[MIC] Listening...` appears. If it does and nothing follows, lower `VAD_THRESHOLD`. Also confirm your mic is the system default `microphone.py` uses the default input device.

**Speech sounds sped up or slowed down**
The sample rate is read from the voice's `.onnx.json`, so make sure the `.onnx` and its `.onnx.json` are the matching pair.

**No sound at all**
Check `python3 -c "import sounddevice as sd; print(sd.query_devices())"` lists your output device and that it's the system default. Then verify Piper on its own: `echo "ciao" | piper --model voices/it_IT-paola-medium.onnx --output_file test.wav`.

**`OSError: PortAudio library not found`**
Install PortAudio: `sudo dnf install portaudio` (Fedora) or `sudo apt install libportaudio2` (Ubuntu/Debian).

**Wake word model not found**
The startup error lists every model openWakeWord has available. `WAKEWORD_MODEL` is matched as a substring against those paths pick a name from the list.

**Camera answers are slow**
Expected with `llava:7b` on CPU. Check the `[TIME]` lines to confirm that's the slow stage, then switch `VISION_MODEL` to a smaller model.

**It always answers with the local model**
Look at the `[CLOUD]` lines at startup and when it falls back. `no API key` means `.env` is missing or the variable name doesn't match `CLOUD_API_KEY_ENV`. `HTTP 401` is a wrong key. `HTTP 400`/`404` mentioning the model usually means the provider retired it; pick a current one for `CLOUD_MODEL` (for Groq: [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations)).

**Ollama connection errors**
Make sure the server is running: `ollama serve`. Confirm the models are pulled: `ollama list`.

---

## Performance notes

Each turn logs a timing breakdown:

```
[TIME] transcribed in 0.84s
[TIME] first audio in 0.62s
[TIME] command end -> first audio 1.46s
[TIME] total 4.30s
```

`first audio` is measured when sound actually reaches the speaker (so it includes synthesis), counted from the end of transcription. `command end -> first audio` is the full wait after your command was captured, and is the number that matters for how responsive it feels. If `transcribed` dominates, use a smaller `WHISPER_MODEL` or a GPU. If `first audio` dominates, the LLM is the bottleneck: use a smaller model. If only `total` is high, it's just a long reply; lower `BRAIN_NUM_PREDICT`.

---

## Notes

- In `"cloud"` mode, your transcribed questions and the recent conversation history (text, not audio) and, for camera questions, a single webcam frame are sent to the cloud provider. Everything else stays local. In `"local"` mode nothing leaves the machine.
- Conversation memory keeps the last `MAX_MEMORY_MESSAGES` exchanges; it resets when you restart.
- Camera questions are answered by the vision model directly rather than passed through the conversational model afterwards. That's a deliberate speed trade-off: roughly a third of the latency, at the cost of some persona consistency in camera replies.