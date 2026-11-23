"""Response ladder. Turns (trust, intent, context) into an action level, climbing
one rung at a time so a misread never jumps straight to a siren."""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Decision, Intent, Level, TrustResult


@dataclass
class PolicyConfig:
    step_s: float = 15.0              # min seconds between rungs
    operator_max_s: float = 60.0      # talk-down this long without leaving -> page a human
    partial_trust: float = 0.4        # "probably family": never go past NOTIFY
    notify_deliveries: bool = True
    siren_requires_armed: bool = True
    max_level: Level = Level.ESCALATE_HUMAN


@dataclass
class _TrackLadder:
    level: Level = Level.SILENT
    changed_ts: float = 0.0
    operator_since: float | None = None
    paged: bool = False


@dataclass
class Policy:
    cfg: PolicyConfig = field(default_factory=PolicyConfig)
    _state: dict[str, _TrackLadder] = field(default_factory=dict)

    def target(self, intent: Intent, trust: TrustResult, *, night: bool, armed: bool) -> tuple[Level, str]:
        if trust.is_family:
            return Level.SILENT, f"trusted ({trust.identity or 'presence'})"
        if intent is Intent.PASSERBY:
            return Level.SILENT, "passerby"
        if intent is Intent.DELIVERY:
            return (Level.NOTIFY if self.cfg.notify_deliveries else Level.SILENT), "delivery"
        if trust.score >= self.cfg.partial_trust:
            return Level.NOTIFY, "partial trust: ask the homeowner, don't confront"
        if intent in (Intent.UNKNOWN, Intent.VISITOR):
            return Level.NOTIFY, intent.value
        if intent is Intent.LOITERING:
            return (Level.OPERATOR if night or armed else Level.WARN), "loitering"
        if intent is Intent.SUSPICIOUS:
            if armed or (night and not self.cfg.siren_requires_armed):
                return Level.SIREN, "suspicious while armed"
            return Level.OPERATOR, "suspicious"
        return Level.NOTIFY, "fallback"

    def decide(self, track_id: str, camera: str, intent: Intent, trust: TrustResult,
               *, now: float, night: bool, armed: bool,
               ceiling: Level | None = None, ceiling_reason: str | None = None) -> Decision:
        st = self._state.setdefault(track_id, _TrackLadder(changed_ts=now - self.cfg.step_s))
        want, why = self.target(intent, trust, night=night, armed=armed)
        if self.cfg.siren_requires_armed and not armed:
            want = min(want, Level.OPERATOR)
        want = min(want, self.cfg.max_level)
        reasons = [why]
        if ceiling is not None and want > ceiling:
            want = ceiling
            if ceiling_reason:
                reasons.append(ceiling_reason)

        # Once a human has been paged, stay there while the threat persists.
        if st.paged and want >= Level.OPERATOR:
            want = Level.ESCALATE_HUMAN

        if want <= st.level:
            # De-escalate immediately (e.g. face resolves to family mid-track).
            if want < st.level:
                st.level, st.changed_ts = want, now
                reasons.append("de-escalated")
        elif (now - st.changed_ts >= self.cfg.step_s
              or (intent is Intent.SUSPICIOUS and st.level < Level.WARN)):
            nxt = Level(st.level + 1)
            # Clearly suspicious: skip the quiet rungs and speak right away.
            if intent is Intent.SUSPICIOUS and nxt < Level.WARN:
                nxt = Level.WARN
            st.level, st.changed_ts = min(nxt, want), now
            reasons.append(f"escalated to {st.level.name}")

        if st.level >= Level.OPERATOR:
            st.operator_since = st.operator_since or now
            if (now - st.operator_since >= self.cfg.operator_max_s
                    and self.cfg.max_level >= Level.ESCALATE_HUMAN):
                st.level = Level.ESCALATE_HUMAN
                st.paged = True
                reasons.append("talk-down not working, paging homeowner")
        else:
            st.operator_since = None

        return Decision(track_id, camera, st.level, intent, trust, reasons)

    def end(self, track_id: str) -> None:
        self._state.pop(track_id, None)
