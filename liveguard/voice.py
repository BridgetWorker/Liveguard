"""Speaking through cameras.

Two separate jobs:
  * Speaker   one-shot lines (the WARN rung). Local Piper TTS -> go2rtc backchannel.
              Free, private, no rate limits.
  * Operator  live two-way talk-down (the OPERATOR rung). Pluggable real-time
              backend; only opened when needed because providers bill listening time.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .audio_server import go2rtc_play

if TYPE_CHECKING:
    from .audio_server import AudioServer

log = logging.getLogger("liveguard.voice")

# ---------------------------------------------------------------- wording


def short_description(desc: str | None, max_words: int = 12) -> str | None:
    """Pull a speakable 'person in a gray hoodie' fragment out of a GenAI description."""
    if not desc:
        return None
    first = re.split(r"(?<=[.!?])\s", desc.strip(), maxsplit=1)[0]
    m = re.search(r"\b(?:wearing|in an?|dressed in)\b\s+([^,.;]+)", first, re.I)
    frag = m.group(1) if m else None
    if frag:  # stop before where/what they're doing: "dark hoodie near the fence" -> "dark hoodie"
        frag = re.split(r"\s+(?:near|by|at|on|next to|standing|walking|holding|carrying|"
                        r"looking|who|that|is|and is)\b", frag, maxsplit=1, flags=re.I)[0]
    if frag:
        words = re.sub(r"^(?:an?|the)\s+", "", frag.strip(), flags=re.I).split()
        return " ".join(words[:max_words])
    return None


def warning_line(desc: str | None, camera_area: str, night: bool) -> str:
    """camera_area is spoken as written, preposition included: "on the front porch"."""
    who = short_description(desc)
    callout = f"You, in the {who}." if who else "Hello there."
    when = "at this hour" if night else "right now"
    return (f"{callout} This is home security. You're on camera {camera_area} "
            f"and this is being recorded. The homeowner isn't expecting anyone {when}. "
            f"Please leave the property.")


OPERATOR_RULES = """\
You are the security monitor for a private home, speaking through an outdoor camera speaker.
Goal: get an uninvited person to leave calmly, or learn who they are and why they're here.

Hard rules:
- Be calm, firm and brief. One or two sentences per turn.
- Mention concrete details you can see (clothing, location, what they're doing) so they
  know they're being watched live.
- NEVER claim police, guards or anyone else are on the way unless the context says so.
- NEVER threaten harm, insult, or reveal whether anyone is home.
- If asked whether you're a real person, don't claim to be human. Say you're the home's
  security monitoring and that the homeowner is being notified.
- If they say they're a delivery driver, neighbour or guest, ask them to leave a note
  or text the homeowner, and tell them the homeowner is being notified.
- Only address people on the property. If they step back onto the street or sidewalk,
  or leave, say "Thank you" and stop talking.
"""


def operator_prompt(context: dict) -> str:
    lines = [OPERATOR_RULES, "Current situation:"]
    for k in ("camera_area", "local_time", "intent", "description", "police_called"):
        if context.get(k) is not None:
            lines.append(f"- {k.replace('_', ' ')}: {context[k]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- interfaces


class Speaker(Protocol):
    def say(self, camera: str, text: str) -> None: ...


class Operator(Protocol):
    def start(self, camera: str, context: dict) -> None: ...
    def update(self, camera: str, context: dict) -> None: ...
    def stop(self, camera: str) -> None: ...


class LogSpeaker:
    def say(self, camera: str, text: str) -> None:
        log.info("[speak %s] %s", camera, text)


@dataclass
class LogOperator:
    active: set[str] = field(default_factory=set)

    def start(self, camera: str, context: dict) -> None:
        self.active.add(camera)
        log.info("[operator start %s]\n%s", camera, operator_prompt(context))

    def update(self, camera: str, context: dict) -> None:
        log.info("[operator update %s] %s", camera, context.get("description"))

    def stop(self, camera: str) -> None:
        self.active.discard(camera)
        log.info("[operator stop %s]", camera)


# ---------------------------------------------------------------- Piper + go2rtc


@dataclass
class Go2rtcSpeaker:
    """Render with Piper, serve the WAV from the AudioServer, tell go2rtc to play it.

    go2rtc fetches the file itself, so `public_url` must be reachable *from go2rtc*.
    `codec` must match the camera's backchannel: pcma for most Dahua/Amcrest/Reolink,
    opus for some others.
    """
    go2rtc_url: str
    piper_model: str
    public_url: str
    server: "AudioServer"
    codec: str = "pcma"
    piper_bin: str = "piper"
    # camera name in Frigate -> go2rtc stream name (if different)
    streams: dict[str, str] = field(default_factory=dict)
    min_gap_s: float = 8.0
    _last: dict[str, float] = field(default_factory=dict)

    def _render(self, text: str) -> str:
        name = f"{uuid.uuid4().hex}.wav"
        out = self.server.out_dir / name
        subprocess.run([self.piper_bin, "--model", self.piper_model, "--output_file", str(out)],
                       input=text.encode(), check=True, timeout=30)
        self._prune()
        return name

    def _prune(self, keep_s: float = 3600) -> None:
        cutoff = time.time() - keep_s
        for f in self.server.out_dir.glob("*.wav"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)

    def say(self, camera: str, text: str) -> None:
        now = time.time()
        if now - self._last.get(camera, 0) < self.min_gap_s:
            return  # don't talk over ourselves
        if self.server.relay.is_open(camera):
            return  # a live session owns the speaker
        self._last[camera] = now
        name = self._render(text)
        go2rtc_play(self.go2rtc_url, self.streams.get(camera, camera),
                    f"{self.public_url.rstrip('/')}/tts/{name}", self.codec, file=True)


@dataclass
class SpeakerOperator:
    """Fallback 'operator' with no real-time model: repeats escalating one-way lines.
    Not conversational, but it works fully offline and costs nothing."""
    speaker: Speaker
    every_s: float = 20.0
    _started: dict[str, float] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=dict)

    LINES = (
        "I can see you on camera. You need to leave the property now.",
        "You are still being recorded. The homeowner has been notified. Leave now.",
    )

    def start(self, camera: str, context: dict) -> None:
        self._started[camera] = self._last[camera] = time.time()
        self.speaker.say(camera, self.LINES[0])

    def update(self, camera: str, context: dict) -> None:
        if camera not in self._started:
            return
        now = time.time()
        if now - self._last[camera] >= self.every_s:
            self._last[camera] = now
            self.speaker.say(camera, self.LINES[1])

    def stop(self, camera: str) -> None:
        self._started.pop(camera, None)
        self._last.pop(camera, None)
