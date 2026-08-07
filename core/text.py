import re


# Sentence-ish boundary: punctuation plus any trailing quotes/brackets,
# or a hard line break. Good enough for speech segmentation - we only
# need a natural place to break audio, not linguistic precision.
_SENTENCE_END = re.compile(
    r"[.!?…]+[\s\"')\]]*|\n+"
)


# Markdown and other artifacts the model sometimes emits, which a TTS
# voice would either read out literally or stumble over.
_SPEECH_NOISE = re.compile(
    r"[*_`#>|]+"
)



def clean_for_speech(
    text
):

    text = _SPEECH_NOISE.sub(
        " ",
        text
    )

    # collapse whitespace left behind by the substitutions

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()



def stream_sentences(
    pieces,
    min_first_chars=15
):

    # Yields the FIRST complete sentence as soon as it's available, so
    # speech can start while the model is still generating. Everything
    # after that is accumulated and yielded as one final chunk.
    #
    # Why not yield every sentence separately? Each speak() call spawns
    # a fresh piper process, and that spawn gap between every sentence
    # sounds choppy. One early chunk plus one continuous remainder
    # gives fast first-audio without breaking up the body.

    buffer = ""

    first_done = False


    for piece in pieces:

        buffer += piece


        if first_done:
            continue


        for match in _SENTENCE_END.finditer(buffer):

            head = buffer[:match.end()].strip()


            # Skip boundaries that are too short to be a real sentence
            # ("Sì." / "Ok.") - speaking those alone then pausing
            # sounds worse than waiting for a bit more text.

            if len(head) < min_first_chars:
                continue


            buffer = buffer[match.end():]

            first_done = True


            yield head

            break



    tail = buffer.strip()


    if tail:

        yield tail