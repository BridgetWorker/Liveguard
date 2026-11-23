"""Home Assistant over REST (stdlib only).

Presence is polled from /api/states every few seconds. Polling is deliberately
boring: no websocket library, reconnect logic or extra dependency, and a 2 s
delay is irrelevant next to how long someone takes to walk up a driveway.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field

from ..models import Presence

HOME_STATES = {"home", "on"}


@dataclass
class HAEntities:
    # HA person / device_tracker entity -> household name (matches TrustConfig.face_to_person values)
    wifi_trackers: dict[str, str] = field(default_factory=dict)
    ble_trackers: dict[str, str] = field(default_factory=dict)
    alarm_panel: str | None = None              # alarm_control_panel.*
    interior_doors: list[str] = field(default_factory=list)  # binary_sensor.* opened from inside
    doorbells: dict[str, str] = field(default_factory=dict)  # binary_sensor/event entity -> camera


@dataclass
class HAPresenceTracker:
    entities: HAEntities
    presence: Presence = field(default_factory=Presence)
    _last: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> Presence:
        """Consistent copy for the engine (polling thread mutates the live one)."""
        with self._lock:
            return copy.deepcopy(self.presence)

    def apply(self, entity_id: str, state: str, ts: float) -> list[tuple[str, str]]:
        """Fold one entity state into Presence. Returns side events like ('doorbell', camera)."""
        with self._lock:
            return self._apply(entity_id, state, ts)

    def _apply(self, entity_id: str, state: str, ts: float) -> list[tuple[str, str]]:
        prev = self._last.get(entity_id)
        self._last[entity_id] = state
        changed = prev is not None and prev != state
        events: list[tuple[str, str]] = []
        p = self.presence
        e = self.entities

        if entity_id in e.wifi_trackers:
            who = e.wifi_trackers[entity_id]
            if state in HOME_STATES:
                p.family_phones_home.add(who)
                if changed and prev not in HOME_STATES:
                    p.recent_wifi_joins[who] = ts
            else:
                p.family_phones_home.discard(who)
        if entity_id in e.ble_trackers:
            who = e.ble_trackers[entity_id]
            if state not in ("not_home", "away", "off", "unavailable", "unknown"):
                p.ble_nearby.add(who)
            else:
                p.ble_nearby.discard(who)
        if entity_id == e.alarm_panel:
            armed = state.startswith("armed") or state == "triggered"
            if changed and p.alarm_armed and not armed:
                p.last_disarm_ts = ts
            p.alarm_armed = armed
        if entity_id in e.interior_doors and changed and state == "on":
            p.last_interior_door_ts = ts
        if entity_id in e.doorbells and changed and state not in ("off", "unavailable", "unknown"):
            # binary_sensor goes on; event.* entities change state to a new timestamp
            events.append(("doorbell", e.doorbells[entity_id]))
        return events

    def watched(self) -> set[str]:
        e = self.entities
        ids = set(e.wifi_trackers) | set(e.ble_trackers) | set(e.interior_doors) | set(e.doorbells)
        if e.alarm_panel:
            ids.add(e.alarm_panel)
        return ids


class HAClient:
    def __init__(self, url: str, token: str, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.url}{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else None

    def states(self) -> list[dict]:
        return self._req("GET", "/api/states") or []

    def call(self, domain: str, service: str, data: dict) -> None:
        self._req("POST", f"/api/services/{domain}/{service}", data)


def poll_presence(client: HAClient, tracker: HAPresenceTracker, on_event, interval: float = 2.0,
                  stop: threading.Event | None = None, on_error=None) -> None:
    stop = stop or threading.Event()
    watched = tracker.watched()
    while not stop.is_set():
        try:
            now = time.time()
            for s in client.states():
                if s.get("entity_id") in watched:
                    for ev in tracker.apply(s["entity_id"], str(s.get("state")), now):
                        on_event(*ev, now)
        except Exception as e:  # HA restarting, network blip: keep going
            if on_error:
                on_error(e)
        stop.wait(interval)
