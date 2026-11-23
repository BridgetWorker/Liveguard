"""Frigate MQTT -> Observation.

Topics used:
  frigate/events                  new/update/end for tracked objects
  frigate/tracked_object_update   GenAI descriptions and face results
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..models import Observation


def _sub_label(after: dict) -> tuple[str | None, float]:
    sl = after.get("sub_label")
    if isinstance(sl, (list, tuple)) and sl:
        name = sl[0]
        score = float(sl[1]) if len(sl) > 1 and sl[1] is not None else 0.0
        return name, score
    if isinstance(sl, str):
        return sl, 1.0  # manually set sub label in Frigate UI: treat as confident
    return None, 0.0


@dataclass
class FrigateParser:
    """Stateful because descriptions/faces/packages arrive on separate messages."""
    package_ttl_s: float = 900.0
    _descriptions: dict[str, str] = field(default_factory=dict)
    _faces: dict[str, tuple[str, float]] = field(default_factory=dict)
    _packages: dict[str, float] = field(default_factory=dict)  # camera -> last seen ts

    def _package_near(self, camera: str, ts: float) -> bool:
        seen = self._packages.get(camera)
        return seen is not None and ts - seen <= self.package_ttl_s

    def on_event(self, payload: bytes | str) -> Observation | None:
        msg = json.loads(payload)
        after = msg.get("after") or {}
        label = after.get("label")
        camera = after.get("camera")
        if not camera or not label:
            return None
        ts = float(after.get("frame_time") or after.get("start_time") or time.time())
        zones = list(after.get("current_zones") or after.get("entered_zones") or [])

        if label == "package":
            if msg.get("type") != "end":
                self._packages[camera] = ts
            return None
        if label != "person":
            return None

        tid = after["id"]
        name, score = _sub_label(after)
        if name is None and tid in self._faces:
            name, score = self._faces[tid]
        ended = msg.get("type") == "end"
        obs = Observation(
            track_id=tid,
            camera=camera,
            ts=ts,
            label=label,
            zones=zones,
            sub_label=name,
            sub_label_score=score,
            stationary=bool(after.get("stationary")),
            has_package_nearby=self._package_near(camera, ts),
            description=self._descriptions.get(tid),
            ended=ended,
        )
        if ended:
            self._descriptions.pop(tid, None)
            self._faces.pop(tid, None)
        return obs

    def on_tracked_update(self, payload: bytes | str) -> str | None:
        """Store GenAI description / face result. Returns the track id it applies to."""
        msg = json.loads(payload)
        tid = msg.get("id")
        if not tid:
            return None
        kind = msg.get("type")
        if kind == "description" and msg.get("description"):
            self._descriptions[tid] = msg["description"]
            return tid
        if kind == "face" and msg.get("name"):
            self._faces[tid] = (msg["name"], float(msg.get("score") or 0.0))
            return tid
        return None


def run_mqtt(host: str, port: int, username: str | None, password: str | None,
             parser: FrigateParser, sink, topic_prefix: str = "frigate") -> None:
    """Blocking MQTT loop (paho-mqtt). `sink(kind, obj)` receives ('obs', Observation)."""
    import paho.mqtt.client as mqtt  # imported lazily so the core stays dependency-free

    def on_connect(client, _ud, _flags, _rc, _props=None):
        client.subscribe(f"{topic_prefix}/events")
        client.subscribe(f"{topic_prefix}/tracked_object_update")
        client.subscribe("liveguard/feedback")

    def on_message(_client, _ud, msg):
        try:
            if msg.topic.endswith("/events"):
                obs = parser.on_event(msg.payload)
                if obs:
                    sink("obs", obs)
            elif msg.topic.endswith("/tracked_object_update"):
                parser.on_tracked_update(msg.payload)
            elif msg.topic == "liveguard/feedback":
                sink("feedback", json.loads(msg.payload))
        except Exception as e:  # never let one bad message kill the loop
            sink("error", e)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="liveguard")
    if username:
        client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=30)
    sink("mqtt_client", client)
    client.loop_forever(retry_first_connection=True)
