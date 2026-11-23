"""Test mode: scripted scenes that feed fake Frigate observations into the engine.

Camera and zone names are taken from your config, so a scene plays out against
your real zones, policy and deliberator. It just isn't coming from a camera.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .intent import ZoneConfig
from .models import Observation

_ids = itertools.count(1)


@dataclass
class Scene:
    id: str
    name: str
    blurb: str
    camera_hint: str            # "front" | "back"
    night: bool | None          # force night / day while it runs; None = real clock
    steps: list[tuple[float, dict]]  # (seconds from start, observation fields)
    armed: bool | None = None
    wifi_join: str | None = None


def _pick(names: list[str], hint: str) -> str:
    for n in names:
        if hint in n:
            return n
    if not names:
        return f"{hint}_cam"
    return names[0] if hint == "front" else names[-1]


def _first(s: set[str], fallback: str) -> str:
    return sorted(s)[0] if s else fallback


def build_scenes(zones: ZoneConfig, family: list[str]) -> list[Scene]:
    prop = [_first(zones.on_property, "")] if zones.on_property else []
    door = prop + [_first(zones.door, "porch")]
    back = prop + [_first(zones.sensitive, "backyard")]
    drive = prop + ["driveway"]
    street = [_first(zones.off_property, "sidewalk")]
    someone = family[0] if family else "family"

    def walk(zs_by_time, desc=None, **extra):
        return [(t, {"zones": zs, "description": desc, **extra}) for t, zs in zs_by_time]

    return [
        Scene("prowler", "Prowler in the back yard",
              "Unknown person crouching by the fence at 2 a.m. Should think, then speak.",
              "back", night=True,
              steps=walk([(t, back) for t in range(0, 44, 2)],
                         "A man wearing a dark hoodie, crouching by the back fence and peering at the windows.")),
        Scene("delivery", "Package delivery",
              "Driver walks up, leaves a box, leaves. Should stay quiet.",
              "front", night=False,
              steps=walk([(0, drive), (6, door), (12, door), (18, drive)],
                         "A delivery driver wearing a brown uniform carrying a box.")
              + [(14, {"zones": door, "has_package_nearby": True,
                       "description": "A delivery driver wearing a brown uniform setting a box down."})]),
        Scene("family", "Family coming home late",
              f"{someone.title()} walks up at night. Face match plus phone on Wi-Fi. Should stay silent.",
              "front", night=True, wifi_join=someone,
              steps=walk([(t, door if t > 4 else drive) for t in range(0, 20, 2)],
                         "A woman wearing a gray coat walking to the door.",
                         sub_label=someone, sub_label_score=0.82)),
        Scene("porch-sitter", "Someone waiting on the porch",
              "Stranger sits on the steps in daylight for over a minute.",
              "front", night=False,
              steps=walk([(t, door) for t in range(0, 80, 4)],
                         "A woman wearing a green jacket sitting on the front steps.", stationary=True)),
        Scene("sidewalk", "Watching from the sidewalk",
              "Someone stares at the house from off the property, alarm armed. Should never speak.",
              "front", night=True, armed=True,
              steps=walk([(t, street) for t in range(0, 60, 4)],
                         "A man standing on the sidewalk looking into the windows.", stationary=True)),
    ]


@dataclass
class SimRunner:
    """Plays scenes on a background thread, pushing ('obs', Observation) into `sink`."""
    sink: Callable[[str, object], None]
    cameras: list[str]
    set_night: Callable[[bool | None], None]
    set_armed: Callable[[bool], None]
    wifi_join: Callable[[str], None]
    _stop: dict[str, threading.Event] = field(default_factory=dict)

    def play(self, scene: Scene) -> str:
        tid = f"sim-{scene.id}-{next(_ids)}"
        cam = _pick(self.cameras, scene.camera_hint)
        stop = threading.Event()
        self._stop[tid] = stop
        threading.Thread(target=self._run, args=(scene, tid, cam, stop), daemon=True).start()
        return tid

    def _run(self, scene: Scene, tid: str, cam: str, stop: threading.Event) -> None:
        if scene.night is not None:
            self.set_night(scene.night)
        if scene.armed is not None:
            self.set_armed(scene.armed)
        if scene.wifi_join:
            self.wifi_join(scene.wifi_join)
        start = time.time()
        last = None
        for at, fields in sorted(scene.steps, key=lambda s: s[0]):
            if stop.wait(max(0.0, start + at - time.time())):
                break
            last = fields
            self.sink("obs", Observation(track_id=tid, camera=cam, ts=time.time(), **fields))
        if not stop.is_set():
            stop.wait(2.0)  # linger a moment so the end shows up as its own step
        self.sink("obs", Observation(track_id=tid, camera=cam, ts=time.time(), ended=True,
                                     **(last or {})))
        self._stop.pop(tid, None)

    def stop_all(self) -> None:
        for ev in list(self._stop.values()):
            ev.set()
