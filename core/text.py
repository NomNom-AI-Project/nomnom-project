import re


# Sentence boundary: end punctuation plus any closing quotes/brackets,
# FOLLOWED BY WHITESPACE - or a hard line break. Requiring the
# whitespace is what keeps "3.5" or "1.000" from being split into
# "3." + "5": while streaming we can't know if a "." at the very end
# of the buffer is a full stop or a decimal point, so we wait for the
# next character. The final sentence is flushed when the stream ends.
_SENTENCE_END = re.compile(
    r"[.!?…]+[\"'”»)\]]*\s+|\n+"
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

    # collapse whitespace left behind by the substitutions, and drop
    # the stray space before punctuation ("**Ciao**!" -> "Ciao!")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return re.sub(
        r"\s+([.,!?;:…])",
        r"\1",
        text
    ).strip()



def stream_sentences(
    pieces,
    min_chars=15
):

    # Yields each complete sentence as soon as it's available, so
    # speech can start while the model is still generating.
    #
    # This used to yield only the first sentence early and then one big
    # remainder, because every speak() call spawned a new piper process
    # and the gaps sounded choppy. TTS now keeps the voice loaded and
    # synthesises the next sentence while the current one plays, so
    # per-sentence streaming is gap-free and the body of long answers
    # starts sooner too.

    buffer = ""


    for piece in pieces:

        buffer += piece


        while True:

            cut = None

            for match in _SENTENCE_END.finditer(buffer):

                # Merge boundaries that are too short to stand alone
                # ("Sì." / "Ok.") into the next sentence - speaking
                # those alone then pausing sounds worse.

                if len(buffer[:match.end()].strip()) >= min_chars:

                    cut = match.end()

                    break


            if cut is None:

                break


            yield buffer[:cut].strip()

            buffer = buffer[cut:]



    tail = buffer.strip()


    if tail:

        yield tail
