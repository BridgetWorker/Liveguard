"""Core data types. Pure Python, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Intent(str, Enum):
    UNKNOWN = "unknown"
    DELIVERY = "delivery"
    VISITOR = "visitor"
    PASSERBY = "passerby"
    LOITERING = "loitering"
    SUSPICIOUS = "suspicious"


class Level(IntEnum):
    """Response ladder. Higher = more assertive. Never skip straight to the top."""
    SILENT = 0
    NOTIFY = 1
    LIGHTS = 2
    WARN = 3          # one-shot spoken warning with description
    OPERATOR = 4      # live conversational talk-down
    SIREN = 5
    ESCALATE_HUMAN = 6  # page the homeowner to take over live


@dataclass
class Observation:
    """One Frigate update for a tracked object, flattened."""
    track_id: str
    camera: str
    ts: float
    label: str = "person"
    zones: list[str] = field(default_factory=list)
    sub_label: str | None = None       # Frigate face recognition result
    sub_label_score: float = 0.0
    stationary: bool = False
    has_package_nearby: bool = False
    description: str | None = None     # GenAI / VLM description, if any
    ended: bool = False


@dataclass
class Presence:
    """Snapshot of household presence signals from Home Assistant."""
    family_phones_home: set[str] = field(default_factory=set)
    # person -> unix ts of last time their phone joined the home network
    recent_wifi_joins: dict[str, float] = field(default_factory=dict)
    ble_nearby: set[str] = field(default_factory=set)
    alarm_armed: bool = False
    last_disarm_ts: float | None = None
    last_interior_door_ts: float | None = None


@dataclass
class TrustResult:
    score: float               # 0.0 = stranger, 1.0 = certainly family
    identity: str | None
    reasons: list[str]

    @property
    def is_family(self) -> bool:
        return self.score >= 0.75


@dataclass
class Decision:
    track_id: str
    camera: str
    level: Level
    intent: Intent
    trust: TrustResult
    reasons: list[str]
