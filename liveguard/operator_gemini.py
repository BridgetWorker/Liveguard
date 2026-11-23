"""Live talk-down with Gemini Live.

One session per camera, opened only at the OPERATOR rung (listening time is billed),
closed when the person leaves, steps off the property, or after max_session_s.

    camera mic (go2rtc RTSP) ──ffmpeg──► 16 kHz PCM ──► Gemini Live
    Frigate latest.jpg every few s ─────────────────►      │
    camera speaker ◄── go2rtc ◄── /live/<cam>.wav ◄── 24 kHz PCM

Half-duplex: while the model's voice is coming out of the speaker, mic audio is
replaced with silence, because outdoor camera mics hear their own speaker and the
model would otherwise talk to itself.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from .audio_server import AudioServer, go2rtc_play
from .voice import operator_prompt

log = logging.getLogger("liveguard.gemini")

MIC_RATE = 16000
MIC_CHUNK = MIC_RATE * 2 // 10   # 100 ms of 16-bit mono

KICKOFF = ("A person just triggered this camera and the decision to speak has been made. "
           "Speak your first line to them now.")


async def ffmpeg_mic(rtsp_url: str) -> AsyncIterator[bytes]:
    """Camera audio as 16 kHz mono s16le, 100 ms chunks."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-vn", "-ac", "1", "-ar", str(MIC_RATE), "-f", "s16le", "-",
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        while True:
            chunk = await proc.stdout.readexactly(MIC_CHUNK)
            yield chunk
    except asyncio.IncompleteReadError:
        return
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def frigate_snapshot(frigate_url: str, camera: str) -> bytes | None:
    try:
        u = f"{frigate_url.rstrip('/')}/api/{camera}/latest.jpg?height=480&quality=70"
        with urllib.request.urlopen(u, timeout=3) as r:
            return r.read()
    except Exception:
        return None


def _default_client(api_key: str):
    from google import genai  # lazy: only needed when the operator is enabled
    return genai.Client(api_key=api_key)


def _blob(data: bytes, mime: str):
    try:
        from google.genai import types
        return types.Blob(data=data, mime_type=mime)
    except ImportError:  # tests with a fake client
        return {"data": data, "mime_type": mime}


@dataclass
class GeminiLiveOperator:
    api_key: str
    server: AudioServer
    go2rtc_url: str
    public_url: str
    rtsp_url: str = "rtsp://localhost:8554/{stream}"
    frigate_url: str | None = None
    model: str = "gemini-3.8-live"
    voice: str | None = None
    codec: str = "pcma"
    streams: dict[str, str] = field(default_factory=dict)
    image_every_s: float = 3.0
    max_session_s: float = 180.0
    # injectable for tests
    client_factory: Callable = _default_client
    mic_factory: Callable[[str], AsyncIterator[bytes]] = ffmpeg_mic
    snapshot: Callable[[str], bytes | None] | None = None
    play: Callable[[str, str], None] | None = None

    _stops: dict[str, threading.Event] = field(default_factory=dict)
    _threads: dict[str, threading.Thread] = field(default_factory=dict)

    # ------------------------------------------------------------ Operator API

    def start(self, camera: str, context: dict) -> None:
        t = self._threads.get(camera)
        if t and t.is_alive():
            return
        stop = threading.Event()
        self._stops[camera] = stop
        t = threading.Thread(target=self._run, args=(camera, context, stop), daemon=True,
                             name=f"gemini-{camera}")
        self._threads[camera] = t
        t.start()

    def update(self, camera: str, context: dict) -> None:
        pass  # the session watches the camera itself via snapshots

    def stop(self, camera: str) -> None:
        ev = self._stops.get(camera)
        if ev:
            ev.set()

    def join(self, camera: str, timeout: float | None = None) -> None:
        t = self._threads.get(camera)
        if t:
            t.join(timeout)

    # ------------------------------------------------------------ session

    def _stream(self, camera: str) -> str:
        return self.streams.get(camera, camera)

    def _config(self, context: dict) -> dict:
        cfg = {
            "response_modalities": ["AUDIO"],
            "system_instruction": operator_prompt(context),
            # audio+video sessions are cut at ~2 min without compression
            "context_window_compression": {"sliding_window": {}},
        }
        if self.voice:
            cfg["speech_config"] = {"voice_config": {"prebuilt_voice_config": {"voice_name": self.voice}}}
        return cfg

    def _run(self, camera: str, context: dict, stop: threading.Event) -> None:
        relay = self.server.relay
        relay.open(camera)
        try:
            play = self.play or (lambda stream, url: go2rtc_play(self.go2rtc_url, stream, url, self.codec))
            play(self._stream(camera), f"{self.public_url.rstrip('/')}/live/{camera}.wav")
            asyncio.run(self._session(camera, context, stop))
        except Exception:
            log.exception("gemini session on %s failed", camera)
        finally:
            relay.close(camera)
            log.info("gemini session on %s closed", camera)

    async def _session(self, camera: str, context: dict, stop: threading.Event) -> None:
        relay = self.server.relay
        client = self.client_factory(self.api_key)
        started = time.monotonic()
        async with client.aio.live.connect(model=self.model, config=self._config(context)) as session:
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": KICKOFF}]}, turn_complete=True)

            async def send_mic():
                rtsp = self.rtsp_url.format(stream=self._stream(camera), camera=camera)
                async for chunk in self.mic_factory(rtsp):
                    if relay.is_playing(camera):
                        chunk = bytes(len(chunk))  # half-duplex: don't let it hear itself
                    await session.send_realtime_input(audio=_blob(chunk, f"audio/pcm;rate={MIC_RATE}"))

            async def send_frames():
                snap = self.snapshot or (lambda cam: frigate_snapshot(self.frigate_url, cam)
                                         if self.frigate_url else None)
                while True:
                    jpg = await asyncio.to_thread(snap, camera)
                    if jpg:
                        try:
                            await session.send_realtime_input(video=_blob(jpg, "image/jpeg"))
                        except TypeError:  # older SDKs call it media=
                            await session.send_realtime_input(media=_blob(jpg, "image/jpeg"))
                    await asyncio.sleep(self.image_every_s)

            async def receive():
                while True:  # receive() yields one model turn, then ends
                    async for msg in session.receive():
                        if getattr(msg, "go_away", None):
                            log.info("gemini go_away on %s", camera)
                            return
                        sc = getattr(msg, "server_content", None)
                        if sc is not None and getattr(sc, "interrupted", False):
                            relay.flush(camera)
                        data = getattr(msg, "data", None)
                        if data:
                            relay.feed(camera, data)

            async def watchdog():
                # poll rather than block a thread on stop.wait(): asyncio.run() waits for
                # executor threads at exit, which would hold the session open
                while not stop.is_set() and time.monotonic() - started < self.max_session_s:
                    await asyncio.sleep(0.1)

            tasks = [asyncio.create_task(c()) for c in (send_mic, send_frames, receive, watchdog)]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for t in done:
                if t.exception():
                    raise t.exception()
            # let the last words play out before the stream closes
            for _ in range(20):  # at most 2 s
                if not relay.is_playing(camera):
                    break
                await asyncio.sleep(0.1)
