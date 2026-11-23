"""Test mode audio: your computer's speakers and mic stand in for the camera's.

Speaker: Piper (if installed and a model is configured) -> espeak-ng / macOS `say`
         -> log only.
Live talk-down: Gemini's voice plays through the default output; your mic is the
         camera mic. Use headphones, or the half-duplex mute will cut you off while
         it talks.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator

from .audio_server import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, LiveRelay
from .operator_gemini import MIC_CHUNK, MIC_RATE

log = logging.getLogger("liveguard.desktop")


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def file_player() -> list[str] | None:
    for cmd in (["paplay"], ["aplay", "-q"], ["afplay"],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if _have(cmd[0]):
            return cmd
    return None


def raw_player(rate: int = SAMPLE_RATE) -> list[str] | None:
    """A command that plays 16-bit mono PCM from stdin."""
    if _have("paplay"):
        return ["paplay", "--raw", "--format=s16le", f"--rate={rate}", "--channels=1"]
    if _have("aplay"):
        return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1"]
    if _have("ffplay"):
        return ["ffplay", "-nodisp", "-loglevel", "quiet", "-f", "s16le", "-ar", str(rate),
                "-ch_layout", "mono", "-i", "-"]
    return None


def default_mic_input() -> list[str]:
    if platform.system() == "Darwin":
        return ["-f", "avfoundation", "-i", ":0"]
    if platform.system() == "Windows":
        return ["-f", "dshow", "-i", "audio=default"]
    return ["-f", "pulse", "-i", "default"]   # PulseAudio and PipeWire


@dataclass
class DesktopSpeaker:
    piper_model: str | None = None
    piper_bin: str = "piper"

    def say(self, camera: str, text: str) -> None:
        log.info("[speak %s] %s", camera, text)
        player = file_player()
        try:
            if self.piper_model and _have(self.piper_bin) and player:
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    subprocess.run([self.piper_bin, "--model", self.piper_model, "--output_file", f.name],
                                   input=text.encode(), check=True, timeout=30,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(player + [f.name], check=False, timeout=60)
            elif _have("say"):
                subprocess.run(["say", text], check=False, timeout=60)
            elif _have("espeak-ng") or _have("espeak"):
                subprocess.run(["espeak-ng" if _have("espeak-ng") else "espeak", text],
                               check=False, timeout=60)
        except Exception as e:
            log.warning("desktop speech failed: %s", e)


def make_desktop_mic(mic_input: str | None = None):
    args = shlex.split(mic_input) if mic_input else default_mic_input()

    async def desktop_mic(_url: str) -> AsyncIterator[bytes]:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error", *args,
            "-ac", "1", "-ar", str(MIC_RATE), "-f", "s16le", "-",
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            while True:
                yield await proc.stdout.readexactly(MIC_CHUNK)
        except asyncio.IncompleteReadError:
            return
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    return desktop_mic


def make_desktop_play(relay: LiveRelay):
    """Replacement for go2rtc playback: pump the relay into the local speakers."""

    def play(camera: str, _url: str) -> None:
        cmd = raw_player()
        if not cmd:
            log.warning("no audio player found (install pulseaudio-utils, alsa-utils or ffmpeg)")
            return

        def pump():
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                next_t = time.monotonic()
                while relay.is_open(camera):
                    proc.stdin.write(relay.take(camera, FRAME_BYTES))
                    next_t += FRAME_MS / 1000
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -0.5:
                        next_t = time.monotonic()
            except (BrokenPipeError, ValueError):
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.terminate()

        threading.Thread(target=pump, daemon=True, name=f"desktop-play-{camera}").start()

    return play
