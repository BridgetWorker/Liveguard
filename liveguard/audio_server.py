"""Small HTTP server that go2rtc pulls audio from.

  /tts/<id>.wav      one-shot Piper renders (the WARN line)
  /live/<camera>.wav an endless, real-time-paced WAV stream for live talk-down

go2rtc plays a URL into a camera's backchannel with
  POST /api/streams?dst=<stream>&src=ffmpeg:<url>#audio=<codec>
so streaming live model audio is "serve a WAV that never ends". We pace it at
real time (20 ms frames) and fill gaps with silence, so ffmpeg never starves and
model audio that arrives faster than real time doesn't get dropped.
"""
from __future__ import annotations

import http.server
import re
import struct
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 24000          # Gemini Live output: 16-bit mono PCM @ 24 kHz
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2
TAIL_S = 0.4                 # keep "playing" this long after the buffer drains (room echo)


def wav_header(rate: int = SAMPLE_RATE, channels: int = 1, bits: int = 16) -> bytes:
    """Header for an unbounded stream (sizes set to max, which ffmpeg accepts)."""
    byte_rate = rate * channels * bits // 8
    block = channels * bits // 8
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, block, bits)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


@dataclass
class _Channel:
    buf: bytearray = field(default_factory=bytearray)
    open: bool = True
    last_audio: float = float("-inf")


class LiveRelay:
    """Thread-safe per-camera PCM buffers between the model and the HTTP stream."""

    def __init__(self, clock=time.monotonic):
        self._ch: dict[str, _Channel] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def open(self, camera: str) -> None:
        with self._lock:
            self._ch[camera] = _Channel()

    def close(self, camera: str) -> None:
        with self._lock:
            ch = self._ch.get(camera)
            if ch:
                ch.open = False

    def is_open(self, camera: str) -> bool:
        with self._lock:
            ch = self._ch.get(camera)
            return bool(ch and ch.open)

    def feed(self, camera: str, pcm: bytes) -> None:
        with self._lock:
            ch = self._ch.get(camera)
            if ch and ch.open:
                ch.buf.extend(pcm)

    def flush(self, camera: str) -> None:
        """Barge-in: the person started talking, drop what the model hadn't said yet."""
        with self._lock:
            ch = self._ch.get(camera)
            if ch:
                ch.buf.clear()

    def take(self, camera: str, n: int = FRAME_BYTES) -> bytes:
        """Next n bytes for playback, padded with silence."""
        with self._lock:
            ch = self._ch.get(camera)
            if not ch:
                return bytes(n)
            out = bytes(ch.buf[:n])
            del ch.buf[:n]
            if out:
                ch.last_audio = self._clock()
        return out + bytes(n - len(out))

    def is_playing(self, camera: str) -> bool:
        """True while model audio is (or just was) coming out of the speaker.
        Used to mute the camera mic so the model doesn't hear itself."""
        with self._lock:
            ch = self._ch.get(camera)
            if not ch:
                return False
            return bool(ch.buf) or (self._clock() - ch.last_audio) < TAIL_S


_SAFE_TTS = re.compile(r"^[a-f0-9]{8,64}\.wav$")
_LIVE = re.compile(r"^/live/([A-Za-z0-9_\-]+)\.wav$")


class AudioServer:
    def __init__(self, out_dir: str, port: int = 8099, host: str = "0.0.0.0"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.relay = LiveRelay()
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                path = urllib.parse.urlparse(self.path).path
                if path.startswith("/tts/"):
                    return server._serve_tts(self, path[5:])
                m = _LIVE.match(path)
                if m:
                    return server._serve_live(self, m.group(1))
                self.send_error(404)

        self._httpd = http.server.ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]

    def start(self) -> "AudioServer":
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="audio-http").start()
        return self

    def shutdown(self) -> None:
        self._httpd.shutdown()

    def _serve_tts(self, h, name: str) -> None:
        f = self.out_dir / name
        if not _SAFE_TTS.match(name) or not f.is_file():
            return h.send_error(404)
        data = f.read_bytes()
        h.send_response(200)
        h.send_header("Content-Type", "audio/wav")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def _serve_live(self, h, camera: str) -> None:
        if not self.relay.is_open(camera):
            return h.send_error(404)
        h.send_response(200)
        h.send_header("Content-Type", "audio/wav")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        try:
            h.wfile.write(wav_header())
            next_t = time.monotonic()
            while self.relay.is_open(camera):
                h.wfile.write(self.relay.take(camera))
                next_t += FRAME_MS / 1000
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.5:      # fell way behind (GC pause, slow client): resync
                    next_t = time.monotonic()
            h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def go2rtc_play(go2rtc_url: str, stream: str, media_url: str, codec: str, file: bool = False) -> None:
    """Ask go2rtc to play a URL into a camera's two-way audio."""
    src = f"ffmpeg:{media_url}#audio={codec}" + ("#input=file" if file else "")
    q = urllib.parse.urlencode({"dst": stream, "src": src})
    req = urllib.request.Request(f"{go2rtc_url.rstrip('/')}/api/streams?{q}", method="POST")
    urllib.request.urlopen(req, timeout=10).read()
