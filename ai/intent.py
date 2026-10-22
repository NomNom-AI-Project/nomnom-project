import re


# Vision intent used to cost a FULL LLM round trip (a qwen2.5:3b call)
# on every single command, before any answer could even start - and on
# the vision path it also forced an extra model load/evict cycle in
# Ollama, since it ran a different model than llava right before it.
#
# For a fixed, small domain like "is the user asking about the camera?"
# a word-boundary keyword match is effectively instant, deterministic,
# and easier to tune than a 3B model's judgement. Add your own phrases
# to the list below - it's meant to be edited.

VISION_WORDS = [
    # Single words here must be unambiguous on their own. Common words
    # like "guarda", "intorno", "davanti" or "foto" used to be listed
    # alone, and sent everyday questions ("guarda, che tempo fa
    # intorno a Roma?") to the slow vision model. Those now only count
    # inside a phrase that really refers to what the camera sees.

    # the camera itself
    "telecamera", "webcam", "fotocamera", "videocamera", "inquadratura",

    # seeing
    "cosa vedi", "che vedi", "chi vedi", "che cosa vedi",
    "mi vedi", "lo vedi", "la vedi", "li vedi", "le vedi",
    "riesci a vedere", "puoi vedere", "vedi questo", "vedi questa",
    "guarda qui", "guarda questo", "guarda questa", "guardami",
    "dai un'occhiata", "dai un occhiata",

    # describing / recognising what's in view
    "descrivi cosa", "descrivimi cosa", "descrivi la scena",
    "descrivi quello che", "descrivimi quello che",
    "riconosci questo", "riconosci questa", "riconosci chi",
    "riconoscimi",

    # position relative to the camera
    "davanti a te", "di fronte a te", "intorno a te", "dietro di me",
    "cosa c'è qui", "chi c'è qui", "cosa c'è davanti", "chi c'è davanti",

    # what the user is holding / wearing / doing
    "in mano", "sto tenendo", "sto indossando", "ho addosso",
    "come sono vestito", "come sono vestita", "cosa sto facendo",
    "come mi vedi", "come ti sembro",

    # objects and colours in view
    "questo oggetto", "quest'oggetto", "quante dita",
    "di che colore è questo", "di che colore è questa",
    "che colore ha questo", "che colore ha questa",
]


# Word-boundary match, so "vedi" doesn't fire inside "prevedi" and
# "foto" doesn't fire inside "fotocopia". Phrases with apostrophes or
# spaces are escaped and matched literally.

_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(w) for w in VISION_WORDS)
    + r")(?!\w)",
    re.IGNORECASE
)



def needs_vision(
    text
):

    if not text:
        return False


    # Whisper sometimes outputs typographic apostrophes (’), which
    # would never match the plain ' in phrases like "cosa c'è".

    text = text.replace("\u2019", "'")


    return _PATTERN.search(text) is not None