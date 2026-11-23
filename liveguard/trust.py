"""Trust scoring: how sure are we this person belongs here?

Each signal is treated as independent evidence and combined with a
noisy-OR: score = 1 - prod(1 - s_i). One strong signal is enough to
lean "family"; two medium signals that agree push it over the line.
A face match alone at night is deliberately capped, because a bad
match on a dark porch is the classic false-friend case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Observation, Presence, TrustResult


@dataclass
class TrustConfig:
    family: set[str] = field(default_factory=set)       # Frigate face names
    trusted: set[str] = field(default_factory=set)      # friends, regular sitters
    # Frigate face name -> HA person / device id (for cross-checking)
    face_to_person: dict[str, str] = field(default_factory=dict)
    face_strong: float = 0.85
    face_weak: float = 0.65
    wifi_window_s: float = 300.0
    door_window_s: float = 120.0
    night_hours: tuple[int, int] = (22, 6)
    night_face_only_cap: float = 0.6


def _is_night(hour: int, span: tuple[int, int]) -> bool:
    start, end = span
    return hour >= start or hour < end if start > end else start <= hour < end


def score(obs: Observation, presence: Presence, cfg: TrustConfig, hour: int) -> TrustResult:
    evidence: list[tuple[float, str]] = []
    identity: str | None = None
    known = cfg.family | cfg.trusted

    face_ok = False
    if obs.sub_label and obs.sub_label in known:
        identity = obs.sub_label
        if obs.sub_label_score >= cfg.face_strong:
            evidence.append((0.8, f"face match {obs.sub_label} ({obs.sub_label_score:.2f})"))
            face_ok = True
        elif obs.sub_label_score >= cfg.face_weak:
            evidence.append((0.35, f"weak face match {obs.sub_label} ({obs.sub_label_score:.2f})"))
            face_ok = True

    person = cfg.face_to_person.get(identity) if identity else None

    # Wi-Fi: a family phone just joined. Stronger if it's the same person as the face.
    for who, ts in presence.recent_wifi_joins.items():
        if obs.ts - ts <= cfg.wifi_window_s and obs.ts >= ts:
            if person and who == person:
                evidence.append((0.65, f"{who}'s phone joined Wi-Fi, matches face"))
            else:
                evidence.append((0.3, f"{who}'s phone joined Wi-Fi recently"))
                identity = identity or who

    # BLE is short-range: their phone/watch is basically at the door.
    if person and person in presence.ble_nearby:
        evidence.append((0.5, f"{person} BLE nearby"))
    elif presence.ble_nearby and not identity:
        evidence.append((0.25, "family BLE tag nearby"))

    # Someone just came *out* of the house.
    if presence.last_disarm_ts and 0 <= obs.ts - presence.last_disarm_ts <= cfg.door_window_s:
        evidence.append((0.4, "alarm disarmed just now"))
    if presence.last_interior_door_ts and 0 <= obs.ts - presence.last_interior_door_ts <= cfg.door_window_s:
        evidence.append((0.3, "door opened from inside"))

    remaining = 1.0
    for s, _ in evidence:
        remaining *= 1.0 - s
    total = 1.0 - remaining

    reasons = [r for _, r in evidence]
    only_face = face_ok and len(evidence) == 1
    if only_face and _is_night(hour, cfg.night_hours) and total > cfg.night_face_only_cap:
        total = cfg.night_face_only_cap
        reasons.append("night + face only: capped")

    if not evidence:
        reasons.append("no matching signals")
    return TrustResult(score=round(total, 3), identity=identity, reasons=reasons)
