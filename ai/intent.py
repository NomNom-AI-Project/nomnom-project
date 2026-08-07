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
    # seeing / looking
    "vedi", "vedere", "vedo", "visto", "vista",
    "guarda", "guardare", "guardi", "osserva", "osservare",

    # the camera itself
    "telecamera", "webcam", "fotocamera", "videocamera",

    # images
    "immagine", "immagini", "foto", "fotografia", "inquadratura",

    # description requests
    "descrivi", "descrivere", "descrizione",
    "riconosci", "riconoscere", "identifica",

    # spatial / worn / colour - typical camera questions
    "davanti", "di fronte", "intorno", "dietro di me",
    "indosso", "indossando", "sto tenendo", "in mano",
    "che colore", "di che colore",

    # common phrasings
    "cosa c'è", "che cosa c'è", "chi c'è",
    "cosa sto", "che cosa sto",
    "quante dita", "questo oggetto", "quest'oggetto",
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


    return _PATTERN.search(text) is not None