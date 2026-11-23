"""Intent: why is this person here? Judged from behaviour over the life of a track."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Intent, Observation

DELIVERY_WORDS = ("usps", "fedex", "amazon", "dhl", "delivery", "uniform", "courier", "mail carrier")
SUSPICIOUS_WORDS = (
    "car door", "door handle", "trying handle", "trying the door", "into the window",
    "into the windows", "at the window", "peering", "looking into", "crouch", "mask",
    "flashlight", "climbing", "over the fence", "hopping", "pry", "crowbar", "hiding",
)


@dataclass
class ZoneConfig:
    """Frigate zone names, grouped by meaning.

    on_property   zones that are *ours*. If set, someone only counts as on the property
                  while standing in one of these. Recommended: draw one big "property"
                  zone along the lot line plus the detail zones below.
    off_property  street, sidewalk, the neighbour's yard. Never spoken to from here.
    """
    door: set[str] = field(default_factory=lambda: {"porch", "front_door"})
    off_property: set[str] = field(default_factory=lambda: {"street", "sidewalk", "neighbor_yard"})
    sensitive: set[str] = field(default_factory=lambda: {"side_yard", "backyard", "driveway_cars", "gate"})
    on_property: set[str] = field(default_factory=set)

    def is_on_property(self, zones: set[str] | list[str]) -> bool:
        zs = set(zones)
        if self.on_property:
            return bool(zs & self.on_property)
        # No explicit property zone: any zone that isn't off-property counts as ours.
        return bool(zs - self.off_property)


@dataclass
class IntentConfig:
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    delivery_max_s: float = 120.0
    loiter_s: float = 90.0
    loiter_stationary_s: float = 45.0
    revisit_window_s: float = 600.0
    revisit_count: int = 3


@dataclass
class TrackState:
    track_id: str
    camera: str
    first_ts: float
    last_ts: float
    zones_seen: set[str] = field(default_factory=set)
    stationary_since: float | None = None
    package_seen: bool = False
    doorbell: bool = False
    descriptions: list[str] = field(default_factory=list)
    ended: bool = False
    on_property_now: bool = False
    ever_on_property: bool = False

    @property
    def dwell(self) -> float:
        return self.last_ts - self.first_ts

    def text(self) -> str:
        return " ".join(self.descriptions).lower()


class IntentTracker:
    def __init__(self, cfg: IntentConfig):
        self.cfg = cfg
        self.tracks: dict[str, TrackState] = {}
        # camera -> [(start_ts, track_id)] of tracks not yet cleared as trusted
        self._recent: dict[str, list[tuple[float, str]]] = {}

    def update(self, obs: Observation, *, doorbell: bool = False) -> TrackState:
        t = self.tracks.get(obs.track_id)
        if t is None:
            t = TrackState(obs.track_id, obs.camera, obs.ts, obs.ts)
            self.tracks[obs.track_id] = t
            starts = self._recent.setdefault(obs.camera, [])
            starts.append((obs.ts, obs.track_id))
            cutoff = obs.ts - self.cfg.revisit_window_s
            self._recent[obs.camera] = [(s, tid) for s, tid in starts if s >= cutoff]
        t.last_ts = max(t.last_ts, obs.ts)
        t.zones_seen.update(obs.zones)
        t.on_property_now = self.cfg.zones.is_on_property(obs.zones)
        t.ever_on_property |= t.on_property_now
        if obs.stationary:
            t.stationary_since = t.stationary_since or obs.ts
        else:
            t.stationary_since = None
        t.package_seen |= obs.has_package_nearby
        t.doorbell |= doorbell
        if obs.description and obs.description not in t.descriptions:
            t.descriptions.append(obs.description)
        t.ended |= obs.ended
        return t

    def mark_trusted(self, track_id: str) -> None:
        """Family/trusted tracks shouldn't count toward 'repeated visits'."""
        for cam, starts in self._recent.items():
            self._recent[cam] = [(s, tid) for s, tid in starts if tid != track_id]

    def forget(self, track_id: str) -> None:
        self.tracks.pop(track_id, None)

    def classify(self, t: TrackState, *, night: bool) -> tuple[Intent, list[str]]:
        z = self.cfg.zones
        text = t.text()
        reasons: list[str] = []

        if any(w in text for w in SUSPICIOUS_WORDS):
            return Intent.SUSPICIOUS, ["description mentions suspicious behaviour"]
        if night and t.zones_seen & z.sensitive:
            return Intent.SUSPICIOUS, [f"in {sorted(t.zones_seen & z.sensitive)} at night"]

        looks_delivery = any(w in text for w in DELIVERY_WORDS)
        at_door = bool(t.zones_seen & z.door)
        if at_door and (t.package_seen or looks_delivery) and t.dwell <= self.cfg.delivery_max_s:
            reasons.append("package/uniform at door, short visit")
            return Intent.DELIVERY, reasons

        if not t.ever_on_property:
            return Intent.PASSERBY, ["never stepped onto the property"]

        still = t.stationary_since is not None and (t.last_ts - t.stationary_since) >= self.cfg.loiter_stationary_s
        if still:
            return Intent.LOITERING, [f"stationary {t.last_ts - t.stationary_since:.0f}s"]
        if t.dwell >= self.cfg.loiter_s and not t.doorbell:
            return Intent.LOITERING, [f"on property {t.dwell:.0f}s without ringing"]
        if len(self._recent.get(t.camera, [])) >= self.cfg.revisit_count:
            return Intent.LOITERING, ["repeated visits to this camera"]

        if at_door and t.doorbell:
            return Intent.VISITOR, ["rang the doorbell"]
        return Intent.UNKNOWN, ["approaching, not enough behaviour yet"]
