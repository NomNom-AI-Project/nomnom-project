import http.client
import json
import os
import ssl
import threading
import time

from urllib.parse import urlsplit

from config import (
    BACKEND,
    CLOUD_BASE_URL,
    CLOUD_API_KEY_ENV,
    CLOUD_TIMEOUT,
    CLOUD_RETRY_AFTER,
    CLOUD_EXTRA_PARAMS
)

from core.console import log


# Minimal streaming client for any OpenAI-compatible chat API (Groq,
# OpenRouter, Together, OpenAI, a vLLM server on your LAN...).
# Standard library only, so there's nothing extra to install.
#
# It keeps ONE connection open and reuses it, so every turn after the
# first skips the TCP + TLS handshake (~100-300ms).


_url = urlsplit(CLOUD_BASE_URL)

_path = _url.path.rstrip("/") + "/chat/completions"

_lock = threading.Lock()

_conn = None

# After a failure, the cloud is skipped until this time instead of
# paying a timeout on every single turn while offline.
_down_until = 0.0



def api_key():

    return os.getenv(CLOUD_API_KEY_ENV, "").strip()



def enabled():

    return BACKEND == "cloud" and bool(api_key())



def available():

    return enabled() and time.time() >= _down_until



def mark_down(error):

    global _down_until

    _down_until = time.time() + CLOUD_RETRY_AFTER

    _close()

    log(
        "CLOUD",
        f"failed ({error}) - using local models "
        f"for the next {CLOUD_RETRY_AFTER}s"
    )



def _close():

    global _conn

    if _conn is not None:

        try:
            _conn.close()
        except Exception:
            pass

    _conn = None



def _connection():

    global _conn

    if _conn is None:

        if _url.scheme == "https":

            _conn = http.client.HTTPSConnection(
                _url.netloc,
                timeout=CLOUD_TIMEOUT,
                context=ssl.create_default_context()
            )

        else:

            _conn = http.client.HTTPConnection(
                _url.netloc,
                timeout=CLOUD_TIMEOUT
            )

    return _conn



def _send(body):

    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        # Some providers sit behind Cloudflare, which blocks the
        # default Python user agent.
        "User-Agent": "nomnom/1.0"
    }

    # A kept-alive connection may have been closed by the server while
    # idle, which only shows up as an error on the next request. Retry
    # once on a fresh connection before treating it as a real failure.

    for attempt in range(2):

        try:

            conn = _connection()

            conn.request("POST", _path, body=body, headers=headers)

            return conn.getresponse()

        except (http.client.HTTPException, OSError):

            _close()

            if attempt == 1:
                raise



def stream_chat(
    model,
    messages,
    max_tokens,
    temperature
):

    # Yields text pieces as they arrive. Raises on any failure so the
    # caller can fall back to the local model.

    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        **CLOUD_EXTRA_PARAMS
    }).encode("utf-8")


    with _lock:

        response = _send(body)


        if response.status != 200:

            detail = response.read(300).decode("utf-8", "replace")

            _close()

            raise RuntimeError(
                f"HTTP {response.status}: {detail.strip()}"
            )


        try:

            # Server-sent events: lines of "data: {...}", ending with
            # "data: [DONE]".

            while True:

                raw = response.readline()

                if not raw:
                    break


                line = raw.decode("utf-8").strip()

                if not line.startswith("data:"):
                    continue


                data = line[5:].strip()

                if data == "[DONE]":
                    break


                event = json.loads(data)

                if "error" in event:
                    raise RuntimeError(f"stream error: {event['error']}")


                choices = event.get("choices") or []

                if not choices:
                    continue


                piece = (choices[0].get("delta") or {}).get("content")

                if piece:
                    yield piece


            # drain the rest so the connection can be reused
            response.read()


        except BaseException:

            # Includes the consumer abandoning the generator: the
            # half-read response would poison the next request.
            _close()

            raise



def with_fallback(
    cloud_stream,
    local_stream
):

    # Takes two zero-argument functions that each return a stream of
    # text pieces. Tries the cloud first; if it fails BEFORE producing
    # any text, the same request goes to the local model instead, so
    # the user always gets an answer.
    #
    # If it fails midway, the part already spoken stays as the answer -
    # restarting locally would repeat half a reply out loud.

    if available():

        started = False

        try:

            for piece in cloud_stream():

                started = True

                yield piece

            return

        except Exception as error:

            mark_down(error)

            if started:
                return


    elif enabled():

        log("CLOUD", "skipped (recent failure) - using local model")


    yield from local_stream()



def warmup():

    # Open the TLS connection in the background at startup, so the
    # first real question doesn't pay the handshake either.

    if BACKEND != "cloud":

        log("CLOUD", "backend: local only")

        return


    if not api_key():

        log(
            "CLOUD",
            f"no API key in {CLOUD_API_KEY_ENV} - using local "
            "models only (see .env.example)"
        )

        return


    log("CLOUD", f"backend: {_url.netloc}, local fallback on")


    def _open():

        try:

            with _lock:
                _connection().connect()

            log("WARMUP", f"cloud connection to {_url.netloc} ready")

        except Exception as error:

            log("WARMUP", "cloud connection failed:", error)


    threading.Thread(target=_open, daemon=True).start()
