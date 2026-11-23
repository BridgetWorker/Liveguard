"""Config loading: YAML file -> dataclasses."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .adapters.homeassistant import HAEntities
from .engine import EngineConfig
from .feedback import FeedbackConfig
from .intent import IntentConfig, ZoneConfig
from .models import Level
from .policy import PolicyConfig
from .trust import TrustConfig


@dataclass
class CameraConfig:
    area: str = "in front of the house"   # spoken: "you're on camera <area>"
    lights: list[str] = field(default_factory=list)   # light/switch entities
    sirens: list[str] = field(default_factory=list)


@dataclass
class Config:
    engine: EngineConfig
    feedback: FeedbackConfig
    ha_url: str
    ha_token: str
    ha_entities: HAEntities
    notify_service: str                   # e.g. notify.mobile_app_scarletts_phone
    mqtt: dict
    cameras: dict[str, CameraConfig]
    voice: dict
    deliberation: dict = field(default_factory=dict)
    operator: dict = field(default_factory=dict)
    webui: dict = field(default_factory=dict)
    desktop: dict = field(default_factory=dict)       # test mode audio overrides
    frigate_url: str | None = None        # snapshots for the deliberator


def _load_raw(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml  # lazy: only needed when loading YAML
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _env(v):
    """Allow `!env NAME`-style values written as "${NAME}"."""
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return os.environ.get(v[2:-1], "")
    return v


def load(path: str) -> Config:
    raw = _load_raw(path)
    h = raw.get("household", {})
    z = raw.get("zones", {})
    p = raw.get("policy", {})
    ha = raw.get("home_assistant", {})
    ent = ha.get("entities", {})

    trust = TrustConfig(
        family=set(h.get("family", [])),
        trusted=set(h.get("trusted", [])),
        face_to_person=dict(h.get("face_to_person", {})),
        night_hours=tuple(h.get("night_hours", (22, 6))),
    )
    intent = IntentConfig(
        zones=ZoneConfig(
            door=set(z.get("door", ZoneConfig().door)),
            off_property=set(z.get("off_property", z.get("street", ZoneConfig().off_property))),
            on_property=set(z.get("on_property", [])),
            sensitive=set(z.get("sensitive", ZoneConfig().sensitive)),
        ),
        **{k: raw["intent"][k] for k in raw.get("intent", {})},
    )
    levels = {lv.name.lower(): lv for lv in Level}
    if "max_level" in p and str(p["max_level"]).lower() not in levels:
        raise ValueError(f"policy.max_level must be one of: {', '.join(levels)}")
    unknown = set(p) - set(PolicyConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown policy setting(s): {', '.join(sorted(unknown))}")
    policy = PolicyConfig(
        **{k: v for k, v in p.items() if k != "max_level"},
        **({"max_level": levels[str(p["max_level"]).lower()]} if "max_level" in p else {}),
    )
    cameras = {name: CameraConfig(**c) for name, c in raw.get("cameras", {}).items()}
    dl = {k: _env(v) for k, v in raw.get("deliberation", {}).items()}
    return Config(
        engine=EngineConfig(timezone=raw.get("timezone", "America/Chicago"),
                            trust=trust, intent=intent, policy=policy,
                            deliberation_timeout_s=float(dl.get("timeout_s", 10)),
                            on_timeout=dl.get("on_timeout", "hold"),
                            camera_areas={n: c.area for n, c in cameras.items()}),
        feedback=FeedbackConfig(**raw.get("feedback", {})),
        ha_url=_env(ha.get("url", "http://homeassistant.local:8123")),
        ha_token=_env(ha.get("token", "${HA_TOKEN}")),
        ha_entities=HAEntities(
            wifi_trackers=ent.get("wifi_trackers", {}),
            ble_trackers=ent.get("ble_trackers", {}),
            alarm_panel=ent.get("alarm_panel"),
            interior_doors=ent.get("interior_doors", []),
            doorbells=ent.get("doorbells", {}),
        ),
        notify_service=ha.get("notify_service", "notify.notify"),
        mqtt={k: _env(v) for k, v in raw.get("mqtt", {}).items()},
        cameras=cameras,
        voice={k: _env(v) for k, v in raw.get("voice", {}).items()},
        deliberation=dl,
        operator={k: _env(v) for k, v in raw.get("operator", {}).items()},
        webui={k: _env(v) for k, v in (raw.get("webui") or {}).items()},
        desktop=dict(raw.get("desktop") or {}),
        frigate_url=raw.get("frigate_url"),
    )
