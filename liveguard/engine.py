"""The brain. Pure logic: observations + presence in, actions out. No network I/O,
so the whole decision path is unit-testable.

Per observation:
  1. trust   who is this?            (face, Wi-Fi, BLE, alarm/door activity)
  2. intent  why are they here?      (zones, dwell, packages, descriptions)
  3. ceiling what are we allowed to do?
       - off our property          -> notify at most, never lights or voice
       - first time we'd speak     -> ask the deliberator, hold at lights meanwhile
       - deliberator said no       -> notify at most (re-ask if behaviour gets worse)
  4. policy  climb the ladder one rung at a time, up to the ceiling
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from . import trust as trust_mod
from .deliberate import DeliberationRequest, RulesDeliberator, Verdict
from .feedback import FeedbackStore
from .intent import IntentConfig, IntentTracker
from .models import Decision, Intent, Level, Observation, Presence
from .policy import Policy, PolicyConfig
from .trust import TrustConfig


@dataclass
class Action:
    kind: str  # notify|lights|warn|operator_start|operator_stop|siren|page|publish|deliberate
    camera: str
    track_id: str
    decision: Decision | None = None
    text: str | None = None
    request: DeliberationRequest | None = None


@dataclass
class EngineConfig:
    timezone: str = "America/Chicago"
    trust: TrustConfig = field(default_factory=TrustConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    deliberation_timeout_s: float = 10.0
    on_timeout: str = "hold"      # hold: stay at lights/notify | rules: use the rules deliberator
    stale_track_s: float = 90.0   # drop tracks Frigate never sent an "end" for
    camera_areas: dict[str, str] = field(default_factory=dict)


LEVEL_ACTION = {
    Level.NOTIFY: "notify",
    Level.LIGHTS: "lights",
    Level.WARN: "warn",
    Level.OPERATOR: "operator_start",
    Level.SIREN: "siren",
    Level.ESCALATE_HUMAN: "page",
}

SEVERITY = {Intent.LOITERING: 1, Intent.SUSPICIOUS: 2}


@dataclass
class _Gate:
    status: str              # pending | approved | denied
    asked_ts: float
    intent: Intent
    verdict: Verdict | None = None


class Engine:
    def __init__(self, cfg: EngineConfig, feedback: FeedbackStore | None = None):
        self.cfg = cfg
        self.tz = ZoneInfo(cfg.timezone)
        self.tracker = IntentTracker(cfg.intent)
        self.policy = Policy(cfg.policy)
        self.feedback = feedback
        self.rules = RulesDeliberator(max_trust=cfg.policy.partial_trust / 2)
        self._acted: dict[str, Level] = {}
        self._doorbell_ts: dict[str, float] = {}
        self._gates: dict[str, _Gate] = {}
        self._last_obs: dict[str, Observation] = {}

    # ------------------------------------------------------------ inputs

    def doorbell(self, camera: str, ts: float) -> None:
        self._doorbell_ts[camera] = ts

    def verdict(self, track_id: str, v: Verdict) -> None:
        g = self._gates.get(track_id)
        if g and g.status == "pending":
            g.status = "approved" if v.speak else "denied"
            g.verdict = v

    def local_hour(self, ts: float) -> int:
        return datetime.fromtimestamp(ts, self.tz).hour

    def tick(self, now: float, presence: Presence) -> list[Action]:
        """Re-evaluate live tracks without new Frigate messages (a stationary loiterer
        sends few updates; deliberation verdicts and timeouts need a nudge)."""
        out: list[Action] = []
        for tid, last in list(self._last_obs.items()):
            stale = now - last.ts > self.cfg.stale_track_s
            out += self.handle(dataclasses.replace(last, ts=now, ended=stale), presence, _tick=True)
        return out

    # ------------------------------------------------------------ core

    def handle(self, obs: Observation, presence: Presence, _tick: bool = False) -> list[Action]:
        hour = self.local_hour(obs.ts)
        night = trust_mod._is_night(hour, self.cfg.trust.night_hours)
        rang = obs.ts - self._doorbell_ts.get(obs.camera, -1e18) <= 120

        state = self.tracker.update(obs, doorbell=rang)
        if not obs.ended and not _tick:
            self._last_obs[obs.track_id] = obs   # ticks reuse this, so keep real data only
        tr = trust_mod.score(obs, presence, self.cfg.trust, hour)
        if tr.is_family:
            self.tracker.mark_trusted(obs.track_id)
        intent, intent_reasons = self.tracker.classify(state, night=night)
        armed = presence.alarm_armed

        actions: list[Action] = []
        want, _ = self.policy.target(intent, tr, night=night, armed=armed)
        ceiling, ceiling_reason = None, None

        if not state.on_property_now:
            ceiling, ceiling_reason = Level.NOTIFY, "off our property: no lights or voice"
        elif want >= Level.WARN:
            g = self._gates.get(obs.track_id)
            worse = g is not None and SEVERITY.get(intent, 0) > SEVERITY.get(g.intent, 0)
            if g is None or (g.status == "denied" and worse):
                g = _Gate("pending", obs.ts, intent)
                self._gates[obs.track_id] = g
                req = self._request(obs, state, tr, intent, intent_reasons, night, armed)
                actions.append(Action("deliberate", obs.camera, obs.track_id, request=req))
            if g.status == "pending" and obs.ts - g.asked_ts >= self.cfg.deliberation_timeout_s:
                if self.cfg.on_timeout == "rules":
                    req = self._request(obs, state, tr, intent, intent_reasons, night, armed)
                    v = self.rules.deliberate(req)
                    v.reason = f"deliberator timed out; rules: {v.reason}"
                else:
                    v = Verdict(False, "deliberator timed out; holding", source="timeout")
                self.verdict(obs.track_id, v)
            if g.status == "pending":
                ceiling, ceiling_reason = Level.LIGHTS, "thinking before speaking"
            elif g.status == "denied":
                ceiling = Level.NOTIFY
                ceiling_reason = f"decided not to speak: {g.verdict.reason if g.verdict else ''}"

        decision = self.policy.decide(
            obs.track_id, obs.camera, intent, tr, now=obs.ts, night=night, armed=armed,
            ceiling=ceiling, ceiling_reason=ceiling_reason,
        )
        decision.reasons = intent_reasons + decision.reasons
        g = self._gates.get(obs.track_id)
        if g and g.status == "approved" and g.verdict and decision.level >= Level.WARN:
            decision.reasons.append(f"chose to speak: {g.verdict.reason}")

        # Learned quiet: don't nag about plain unknown people at a familiar camera/hour.
        if (self.feedback and decision.level == Level.NOTIFY and intent is Intent.UNKNOWN
                and self.feedback.is_quiet(obs.camera, hour)):
            decision.level = Level.SILENT
            decision.reasons.append("learned: usually fine here at this hour")

        actions.insert(0, Action("publish", obs.camera, obs.track_id, decision))
        prev = self._acted.get(obs.track_id, Level.SILENT)

        if decision.level > prev:
            line = g.verdict.line if g and g.verdict else None
            for lvl in range(prev + 1, decision.level + 1):
                kind = LEVEL_ACTION.get(Level(lvl))
                if kind:
                    actions.append(Action(kind, obs.camera, obs.track_id, decision,
                                          text=line if kind == "warn" else None))
            self._acted[obs.track_id] = decision.level
        elif decision.level < prev:
            if prev >= Level.OPERATOR > decision.level:
                actions.append(Action("operator_stop", obs.camera, obs.track_id, decision))
            self._acted[obs.track_id] = decision.level

        if obs.ended:
            if self._acted.get(obs.track_id, Level.SILENT) >= Level.OPERATOR:
                actions.append(Action("operator_stop", obs.camera, obs.track_id, decision))
            self.policy.end(obs.track_id)
            self.tracker.forget(obs.track_id)
            self._acted.pop(obs.track_id, None)
            self._gates.pop(obs.track_id, None)
            self._last_obs.pop(obs.track_id, None)
        return actions

    def _request(self, obs, state, tr, intent, intent_reasons, night, armed) -> DeliberationRequest:
        return DeliberationRequest(
            track_id=obs.track_id,
            camera=obs.camera,
            camera_area=self.cfg.camera_areas.get(obs.camera, obs.camera.replace("_", " ")),
            local_time=datetime.fromtimestamp(obs.ts, self.tz).strftime("%a %-I:%M %p"),
            night=night,
            armed=armed,
            intent=intent.value,
            intent_reasons=intent_reasons,
            trust_score=tr.score,
            trust_identity=tr.identity,
            trust_reasons=tr.reasons,
            zones_now=list(obs.zones),
            on_property=state.on_property_now,
            seconds_on_camera=round(state.dwell, 1),
            descriptions=list(state.descriptions),
        )

    def level_of(self, track_id: str) -> Level:
        return self._acted.get(track_id, Level.SILENT)

    def describe(self, track_id: str) -> str:
        t = self.tracker.tracks.get(track_id)
        return t.descriptions[-1] if t and t.descriptions else ""
