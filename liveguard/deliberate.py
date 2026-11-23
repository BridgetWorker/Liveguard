"""Think before speaking.

The rules engine decides when a situation *might* deserve a voice. Before Liveguard
speaks to a track for the first time, a deliberator weighs everything (who it might
be, where they are, what they're doing, the snapshot) and returns speak / don't
speak plus a reason. The reason is published, so you can always see why it talked
or stayed quiet.

Speaking to the wrong person (a neighbour, a kid getting a ball, a meter reader)
costs more than staying quiet, because you're notified either way. So every
deliberator defaults to silence when unsure.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .models import Intent

log = logging.getLogger("liveguard.deliberate")


@dataclass
class DeliberationRequest:
    track_id: str
    camera: str
    camera_area: str
    local_time: str
    night: bool
    armed: bool
    intent: str
    intent_reasons: list[str]
    trust_score: float
    trust_identity: str | None
    trust_reasons: list[str]
    zones_now: list[str]
    on_property: bool
    seconds_on_camera: float
    descriptions: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    speak: bool
    reason: str
    line: str | None = None   # optional opening line written by the model
    source: str = "rules"


class Deliberator(Protocol):
    def deliberate(self, req: DeliberationRequest) -> Verdict: ...


class RulesDeliberator:
    """No-model fallback. Speaks only for clear cases."""

    def __init__(self, max_trust: float = 0.2):
        self.max_trust = max_trust

    def deliberate(self, req: DeliberationRequest) -> Verdict:
        if not req.on_property:
            return Verdict(False, "not on our property")
        if req.trust_score > self.max_trust:
            return Verdict(False, f"might be someone we know (trust {req.trust_score:.2f})")
        if req.intent == Intent.SUSPICIOUS.value:
            return Verdict(True, "suspicious behaviour on the property: " + "; ".join(req.intent_reasons))
        if req.intent == Intent.LOITERING.value and (req.night or req.armed):
            return Verdict(True, "unknown person lingering at night / while armed")
        return Verdict(False, "not clear enough to speak; notifying only")


SYSTEM_PROMPT = """\
You are the judgment layer of a home security system. A rules engine flagged a person
and wants to speak to them through an outdoor camera speaker. Decide whether speaking
is appropriate RIGHT NOW. The homeowner is already being notified, so staying quiet is
always safe; speaking to an innocent person is embarrassing and erodes trust.

Think it through privately first:
1. Are they actually on the property, or on the street/sidewalk/neighbour's land?
2. Who could they plausibly be? Family or a friend the face/phone checks missed,
   a neighbour, a child fetching a ball, a delivery driver, a utility or meter reader,
   a lost visitor, a solicitor. Look at clothing, uniforms, vehicles and what they hold.
3. What are they doing? Normal (walking to the door, leaving a package, reading a
   meter) or concerning (checking car doors or windows, hiding, going to the back,
   hanging around with no purpose, returning repeatedly)?
4. Does time of day or the alarm state change things?

Speak only if a reasonable homeowner watching this live would want to speak to them.
When in doubt, don't.

Answer with JSON only:
{"speak": true|false, "reason": "<one sentence the homeowner will read>",
 "line": "<if speaking: calm first sentence to say, mentioning a visible detail; else empty>"}
"""


@dataclass
class OllamaDeliberator:
    """Local vision LLM via Ollama (e.g. a qwen-vl or gemma vision model).
    Attaches the Frigate snapshot if frigate_url is set."""
    url: str = "http://localhost:11434"
    model: str = "qwen2.5vl:7b"
    frigate_url: str | None = None
    timeout: float = 20.0

    def _snapshot_b64(self, track_id: str) -> str | None:
        if not self.frigate_url:
            return None
        try:
            u = f"{self.frigate_url.rstrip('/')}/api/events/{track_id}/snapshot.jpg?crop=1&quality=80"
            with urllib.request.urlopen(u, timeout=5) as r:
                return base64.b64encode(r.read()).decode()
        except Exception as e:
            log.info("no snapshot for %s: %s", track_id, e)
            return None

    def deliberate(self, req: DeliberationRequest) -> Verdict:
        user = {"role": "user", "content": "Situation:\n" + json.dumps(asdict(req), indent=1)}
        img = self._snapshot_b64(req.track_id)
        if img:
            user["images"] = [img]
        body = json.dumps({
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, user],
        }).encode()
        r = urllib.request.Request(f"{self.url.rstrip('/')}/api/chat", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=self.timeout) as resp:
            content = json.loads(resp.read())["message"]["content"]
        return parse_verdict(content, source=f"ollama:{self.model}")


def parse_verdict(content: str, source: str) -> Verdict:
    """Lenient parse; anything unclear becomes 'don't speak'."""
    try:
        start, end = content.index("{"), content.rindex("}") + 1
        data = json.loads(content[start:end])
    except ValueError:
        return Verdict(False, "model gave no usable answer; staying quiet", source=source)
    speak = data.get("speak") is True or str(data.get("speak")).lower() == "true"
    line = (data.get("line") or "").strip() or None
    reason = (data.get("reason") or "").strip() or ("model chose to speak" if speak else "model chose silence")
    return Verdict(speak, reason, line if speak else None, source=source)
