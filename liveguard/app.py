"""Wiring: Frigate MQTT + Home Assistant presence -> Engine -> actions."""
from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import config as config_mod
from .adapters.frigate import FrigateParser, run_mqtt
from .adapters.homeassistant import HAClient, HAPresenceTracker, poll_presence
from .deliberate import OllamaDeliberator, RulesDeliberator, Verdict
from .engine import Action, Engine
from .feedback import FeedbackStore
from .models import Level
from .trust import _is_night
from .voice import Go2rtcSpeaker, LogOperator, LogSpeaker, SpeakerOperator, warning_line

log = logging.getLogger("liveguard")


def parse_feedback_action(action: str) -> tuple[str, int, bool] | None:
    """'LIVEGUARD_FINE|front_door|19' -> ('front_door', 19, True)"""
    parts = action.split("|")
    if len(parts) != 3 or parts[0] not in ("LIVEGUARD_FINE", "LIVEGUARD_SKETCHY"):
        return None
    try:
        return parts[1], int(parts[2]), parts[0] == "LIVEGUARD_FINE"
    except ValueError:
        return None


class App:
    def __init__(self, cfg: config_mod.Config):
        self.cfg = cfg
        self.feedback = FeedbackStore.load(cfg.feedback)
        self.engine = Engine(cfg.engine, self.feedback)
        self.ha = HAClient(cfg.ha_url, cfg.ha_token)
        self.presence = HAPresenceTracker(cfg.ha_entities)
        self.q: queue.Queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="act")
        self.mqtt_client = None

        v = cfg.voice
        if v.get("go2rtc_url") and v.get("piper_model") and v.get("public_url"):
            sp = Go2rtcSpeaker(
                go2rtc_url=v["go2rtc_url"], piper_model=v["piper_model"], public_url=v["public_url"],
                out_dir=v.get("out_dir", "/data/tts"), serve_port=int(v.get("serve_port", 8099)),
                codec=v.get("codec", "pcma"), streams=v.get("streams", {}),
            )
            sp.start_server()
            self.speaker = sp
        else:
            log.warning("voice not configured: spoken lines will only be logged")
            self.speaker = LogSpeaker()
        dl = cfg.deliberation
        if dl.get("backend", "rules") == "ollama":
            self.deliberator = OllamaDeliberator(
                url=dl.get("url", "http://localhost:11434"), model=dl.get("model", "qwen2.5vl:7b"),
                frigate_url=cfg.frigate_url, timeout=float(dl.get("timeout_s", 10)))
        else:
            self.deliberator = RulesDeliberator()
        # Real-time operator backends plug in here; offline fallback for now.
        self.operator = SpeakerOperator(self.speaker) if v.get("operator") != "log" else LogOperator()

    # ---------------------------------------------------------------- inputs

    def sink(self, kind: str, obj) -> None:
        if kind == "mqtt_client":
            self.mqtt_client = obj
        else:
            self.q.put((kind, obj))

    def on_ha_event(self, kind: str, camera: str, ts: float) -> None:
        if kind == "doorbell":
            self.q.put(("doorbell", (camera, ts)))

    # ---------------------------------------------------------------- actions

    def _area(self, camera: str) -> str:
        c = self.cfg.cameras.get(camera)
        return c.area if c else f"at the {camera.replace('_', ' ')}"

    def _notify(self, a: Action, *, critical: bool = False) -> None:
        d = a.decision
        hour = self.engine.local_hour(time.time())
        title = "Liveguard: talk-down not working" if critical else f"Liveguard: {d.intent.value} at {self._area(a.camera)}"
        data = {
            "image": f"/api/frigate/notifications/{a.track_id}/snapshot.jpg",
            "tag": f"liveguard-{a.track_id}",
            "actions": [
                {"action": f"LIVEGUARD_FINE|{a.camera}|{hour}", "title": "This was fine"},
                {"action": f"LIVEGUARD_SKETCHY|{a.camera}|{hour}", "title": "Sketchy"},
            ],
        }
        if critical:
            data |= {"push": {"sound": {"name": "default", "critical": 1, "volume": 1.0}},
                     "ttl": 0, "priority": "high", "channel": "alarm_stream"}
        domain, service = self.cfg.notify_service.split(".", 1)
        self.ha.call(domain, service, {"title": title, "message": "; ".join(d.reasons), "data": data})

    def _context(self, a: Action) -> dict:
        return {
            "camera_area": self._area(a.camera),
            "local_time": datetime.now(self.engine.tz).strftime("%-I:%M %p"),
            "intent": a.decision.intent.value if a.decision else None,
            "description": self.engine.describe(a.track_id) or None,
            "police_called": False,
        }

    def execute(self, a: Action) -> None:
        cam = self.cfg.cameras.get(a.camera)
        try:
            if a.kind == "publish" and self.mqtt_client:
                d = a.decision
                self.mqtt_client.publish("liveguard/decision", json.dumps({
                    "camera": a.camera, "track_id": a.track_id, "level": d.level.name,
                    "intent": d.intent.value, "trust": d.trust.score, "identity": d.trust.identity,
                    "reasons": d.reasons + d.trust.reasons,
                }))
            elif a.kind == "notify":
                self._notify(a)
            elif a.kind == "lights" and cam and cam.lights:
                self.ha.call("homeassistant", "turn_on", {"entity_id": cam.lights})
            elif a.kind == "warn":
                night = _is_night(self.engine.local_hour(time.time()), self.cfg.engine.trust.night_hours)
                line = a.text or warning_line(self.engine.describe(a.track_id), self._area(a.camera), night)
                self.speaker.say(a.camera, line)
            elif a.kind == "deliberate":
                try:
                    v = self.deliberator.deliberate(a.request)
                except Exception as e:
                    log.warning("deliberator failed: %s", e)
                    v = Verdict(False, f"deliberator error ({type(e).__name__}); staying quiet", source="error")
                log.info("%s %s deliberation: speak=%s (%s) %s", a.camera, a.track_id, v.speak, v.source, v.reason)
                self.q.put(("verdict", (a.track_id, v)))
            elif a.kind == "operator_start":
                self.operator.start(a.camera, self._context(a))
            elif a.kind == "operator_stop":
                self.operator.stop(a.camera)
            elif a.kind == "siren" and cam and cam.sirens:
                self.ha.call("homeassistant", "turn_on", {"entity_id": cam.sirens})
            elif a.kind == "page":
                self._notify(a, critical=True)
        except Exception:
            log.exception("action %s on %s failed", a.kind, a.camera)

    # ---------------------------------------------------------------- loop

    def dispatch(self, actions: list[Action]) -> None:
        for a in actions:
            if a.kind not in ("publish",) and a.decision:
                log.info("%s %s -> %s (%s)", a.camera, a.track_id, a.kind, "; ".join(a.decision.reasons))
            self.pool.submit(self.execute, a)

    def run(self) -> None:
        m = self.cfg.mqtt
        threading.Thread(target=run_mqtt, daemon=True, name="mqtt", args=(
            m.get("host", "localhost"), int(m.get("port", 1883)), m.get("username"), m.get("password"),
            FrigateParser(), self.sink)).start()

        threading.Thread(target=poll_presence, daemon=True, name="ha", args=(
            self.ha, self.presence, self.on_ha_event),
            kwargs={"on_error": lambda e: log.warning("HA poll failed: %s", e)}).start()

        log.info("liveguard running")
        last_tick = 0.0
        while True:
            try:
                kind, obj = self.q.get(timeout=1.0)
            except queue.Empty:
                kind, obj = "tick", None
            now = time.time()
            if kind == "tick" or now - last_tick >= 2.0:
                last_tick = now
                self.dispatch(self.engine.tick(now, self.presence.snapshot()))
            if kind == "obs":
                self.dispatch(self.engine.handle(obj, self.presence.snapshot()))
                if self.engine.level_of(obj.track_id) >= Level.OPERATOR:
                    self.pool.submit(self.operator.update, obj.camera, {"description": obj.description})
            elif kind == "verdict":
                tid, v = obj
                self.engine.verdict(tid, v)
                self.dispatch(self.engine.tick(now, self.presence.snapshot()))
            elif kind == "doorbell":
                self.engine.doorbell(*obj)
            elif kind == "feedback":
                parsed = parse_feedback_action(str(obj.get("action", "")))
                if parsed:
                    self.feedback.record(*parsed)
                    self.feedback.save()
            elif kind == "error":
                log.warning("bad MQTT message: %s", obj)


def main() -> None:
    ap = argparse.ArgumentParser(prog="liveguard")
    ap.add_argument("-c", "--config", default="/config/liveguard.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    App(config_mod.load(args.config)).run()


if __name__ == "__main__":
    main()
