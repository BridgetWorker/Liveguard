"""Learning from homeowner feedback.

Every notification has "This was fine" / "This was sketchy" buttons. Answers land
here, keyed by camera and hour-of-day. When a camera/hour has a strong history of
"fine" for people we couldn't identify (the neighbour's evening walk, the 3 pm
school pickup), Liveguard stops notifying about *unknown, non-suspicious* people
there. It never quiets loitering or suspicious behaviour: learning can only remove
nagging, never remove protection.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeedbackConfig:
    path: str = "/data/feedback.json"
    min_samples: int = 5
    quiet_ratio: float = 0.85
    hour_slop: int = 1  # also count neighbouring hours


@dataclass
class FeedbackStore:
    cfg: FeedbackConfig = field(default_factory=FeedbackConfig)
    # "camera|hour" -> [fine, sketchy]
    counts: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: FeedbackConfig) -> "FeedbackStore":
        store = cls(cfg)
        p = Path(cfg.path)
        if p.exists():
            try:
                store.counts = {k: list(v) for k, v in json.loads(p.read_text()).get("counts", {}).items()}
            except (ValueError, OSError):
                store.counts = {}
        return store

    def save(self) -> None:
        p = Path(self.cfg.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".feedback-")
        with os.fdopen(fd, "w") as f:
            json.dump({"counts": self.counts}, f)
        os.replace(tmp, p)

    def record(self, camera: str, hour: int, fine: bool) -> None:
        c = self.counts.setdefault(f"{camera}|{hour % 24}", [0, 0])
        c[0 if fine else 1] += 1

    def _window(self, camera: str, hour: int) -> tuple[int, int]:
        fine = sketchy = 0
        for d in range(-self.cfg.hour_slop, self.cfg.hour_slop + 1):
            f, s = self.counts.get(f"{camera}|{(hour + d) % 24}", (0, 0))
            fine += f
            sketchy += s
        return fine, sketchy

    def is_quiet(self, camera: str, hour: int) -> bool:
        fine, sketchy = self._window(camera, hour)
        total = fine + sketchy
        return total >= self.cfg.min_samples and fine / total >= self.cfg.quiet_ratio
