# NomNom

A fully local, offline voice assistant in Italian. Wake word → speech recognition → LLM → spoken reply, with optional webcam vision. Nothing leaves the machine.

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
                                   qwen2.5:3b                      llava:7b
                                    (Ollama)                     (Ollama + frame)
                                        │                              │
                                        └───────────────┬──────────────┘
                                                        │
                                                     Piper TTS ──► speakers
```

The assistant idles until it hears the wake word, plays a short greeting, records until you stop talking, then answers out loud. Answers begin playing on the first complete sentence, while the rest is still being generated.

---

## Requirements

**System packages** (Linux):

```bash
sudo apt install python3 python3-pip alsa-utils   # alsa-utils provides `aplay`
```

**Ollama** install from [ollama.com](https://ollama.com), then pull the models:

```bash
ollama pull qwen2.5:3b
ollama pull llava:7b
```

**Piper** — the TTS engine, plus an Italian voice. The project expects the voice at:

```
voices/it_IT-paola-medium.onnx
```

Piper voices ship as a pair: the `.onnx` file and a matching `.onnx.json`. Keep both together in `voices/`.

**Python packages**:

```bash
pip install -r req.txt
```

On newer Ubuntu/Debian you may need `pip install -r req.txt --break-system-packages`, or use a virtualenv.

---

## Running

From the project root:

```bash
python3 main.py
```

Wait for `[SYSTEM] Ready`, then say **"Hey Jarvis"**. The assistant replies *"Eccomi, dimmi."* then ask your question.

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

### Speed

| Setting | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `"base"` | `"tiny"` is ~2x faster, less accurate. `"small"` is slower, more accurate. |
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
| `SILENCE_DURATION` | `0.8` | Seconds of silence before your command is considered finished. |
| `MIN_SPEECH_DURATION` | `0.3` | Bursts shorter than this are discarded as noise (coughs, door slams). |
| `MAX_SPEECH_DURATION` | `10.0` | Hard cap on a single command. |
| `COMMAND_WAIT_TIMEOUT` | `15` | How long to wait for you to start talking before giving up. |
| `MIC_SETTLE_DELAY` | `0.3` | Pause after the greeting so its own audio tail isn't recorded. |

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
  intent.py             routes camera questions vs normal questions
  prompts.py            system prompt (single source of truth)

audio/
  microphone.py         input stream + frame queue
  wakeword.py           openWakeWord detection
  vad.py                Silero voice activity detection
  stt.py                faster-whisper transcription
  tts.py                Piper → aplay

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
`PIPER_SAMPLE_RATE` doesn't match your voice model. Open the `.onnx.json` next to your `.onnx` voice, find `sample_rate`, and set it to match.

**No sound at all**
Verify `aplay` works standalone: `aplay /usr/share/sounds/alsa/Front_Center.wav`. Then verify Piper: `echo "ciao" | piper --model voices/it_IT-paola-medium.onnx --output_file test.wav`.

**Wake word model not found**
The startup error lists every model openWakeWord has available. `WAKEWORD_MODEL` is matched as a substring against those paths pick a name from the list.

**Camera answers are slow**
Expected with `llava:7b` on CPU. Check the `[TIME]` lines to confirm that's the slow stage, then switch `VISION_MODEL` to a smaller model.

**Ollama connection errors**
Make sure the server is running: `ollama serve`. Confirm the models are pulled: `ollama list`.

---

## Performance notes

Each turn logs a timing breakdown:

```
[TIME] transcribed in 0.84s
[TIME] first audio in 1.12s
[TIME] total 4.30s
```

`first audio` is what actually matters for how responsive it feels replies start playing on the first sentence rather than after the full answer is generated. If `total` is high but `first audio` is low, it's just a long reply; lower `BRAIN_NUM_PREDICT`. If `first audio` itself is high, the model is the bottleneck use a smaller one.

---

## Notes

- Everything runs locally. No API keys, no network calls, no telemetry.
- Conversation memory keeps the last `MAX_MEMORY_MESSAGES` exchanges; it resets when you restart.
- Camera questions are answered by the vision model directly rather than passed through the conversational model afterwards. That's a deliberate speed trade-off: roughly a third of the latency, at the cost of some persona consistency in camera replies.