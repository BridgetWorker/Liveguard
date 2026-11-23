"""Wiring: Frigate MQTT + Home Assistant presence -> Engine -> actions.

Test mode (--test): no cameras, MQTT or Home Assistant needed. Scenes from the web
UI feed fake observations, Home Assistant calls are only logged, and voice goes
through this computer's speakers and mic.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import config as config_mod
from .adapters.frigate import FrigateParser, run_mqtt
from .adapters.homeassistant import HAClient, HAPresenceTracker, poll_presence
from .audio_server import AudioServer
from .deliberate import OllamaDeliberator, RulesDeliberator, Verdict
from .engine import Action, Engine
from .feedback import FeedbackStore
from .models import Level, Observation
from .operator_gemini import GeminiLiveOperator
from .sim import SimRunner, build_scenes
from .trust import _is_night
from .voice import Go2rtcSpeaker, LogOperator, LogSpeaker, SpeakerOperator, warning_line

log = logging.getLogger("liveguard")

DEMO_CONFIG = Path(__file__).with_name("demo.yaml")


def parse_feedback_action(action: str) -> tuple[str, int, bool] | None:
    """'LIVEGUARD_FINE|front_door|19' -> ('front_door', 19, True)"""
    parts = action.split("|")
    if len(parts) != 3 or parts[0] not in ("LIVEGUARD_FINE", "LIVEGUARD_SKETCHY"):
        return None
    try:
        return parts[1], int(parts[2]), parts[0] == "LIVEGUARD_FINE"
    except ValueError:
        return None


class LoggedHA:
    """Test mode stand-in for Home Assistant: records calls instead of making them."""

    def __init__(self, record):
        self.record = record

    def call(self, domain: str, service: str, data: dict) -> None:
        target = data.get("entity_id") or data.get("title") or ""
        if isinstance(target, list):
            target = ", ".join(target)
        self.record("ha", text=f"{domain}.{service}: {target}")

    def states(self) -> list:
        return []


class EventLog:
    """Recent happenings for the web UI. Thread-safe ring buffer with sequence numbers."""

    def __init__(self, size: int = 400):
        self._buf: deque = deque(maxlen=size)
        self._seq = 0
        self._lock = threading.Lock()

    def add(self, kind: str, **fields) -> None:
        with self._lock:
            self._seq += 1
            self._buf.append({"seq": self._seq, "ts": time.time(), "kind": kind, **fields})

    def since(self, seq: int) -> tuple[list[dict], int]:
        with self._lock:
            return [e for e in self._buf if e["seq"] > seq], self._seq


class App:
    def __init__(self, cfg: config_mod.Config, *, test_mode: bool = False, config_path: str | None = None):
        self.cfg = cfg
        self.config_path = config_path
        self.test_mode = test_mode
        self.events = EventLog()
        self.tracks: dict[str, dict] = {}      # live view of active tracks
        self._tracks_lock = threading.Lock()
        self.feedback = FeedbackStore.load(cfg.feedback)
        self.engine = Engine(cfg.engine, self.feedback)
        self.ha = LoggedHA(self.events.add) if test_mode else HAClient(cfg.ha_url, cfg.ha_token)
        self.presence = HAPresenceTracker(cfg.ha_entities)
        self.q: queue.Queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="act")
        self.mqtt_client = None

        self._build_voice()
        dl = cfg.deliberation
        if dl.get("backend", "rules") == "ollama":
            self.deliberator = OllamaDeliberator(
                url=dl.get("url", "http://localhost:11434"), model=dl.get("model", "qwen2.5vl:7b"),
                frigate_url=None if test_mode else cfg.frigate_url, timeout=float(dl.get("timeout_s", 10)))
        else:
            self.deliberator = RulesDeliberator()

        self.scenes = build_scenes(cfg.engine.intent.zones, sorted(cfg.engine.trust.family))
        self.sim = SimRunner(
            sink=self.sink, cameras=list(cfg.cameras) or ["front_door", "backyard"],
            set_night=self.set_night, set_armed=self.set_armed, wifi_join=self.wifi_join,
        )

    # ---------------------------------------------------------------- setup

    def _build_voice(self) -> None:
        cfg, v, op = self.cfg, self.cfg.voice, self.cfg.operator
        streams = v.get("streams", {}) or {}
        backend = op.get("backend", "speaker")
        self.audio = None

        if self.test_mode:
            from .desktop import DesktopSpeaker, make_desktop_mic, make_desktop_play
            self.speaker = DesktopSpeaker(piper_model=v.get("piper_model"))
            if backend == "gemini" and op.get("api_key"):
                self.audio = AudioServer(v.get("out_dir", "/tmp/liveguard-tts"), port=0, host="127.0.0.1")
                self.operator = GeminiLiveOperator(
                    api_key=op["api_key"], server=self.audio, go2rtc_url="", public_url="",
                    model=op.get("model", "gemini-3.8-live"), voice=op.get("voice"),
                    max_session_s=float(op.get("max_session_s", 180)),
                    mic_factory=make_desktop_mic(cfg.desktop.get("mic_input")),
                    play=make_desktop_play(self.audio.relay), snapshot=lambda cam: None,
                )
            else:
                self.operator = SpeakerOperator(self.speaker)
            return

        if v.get("go2rtc_url") and v.get("public_url"):
            self.audio = AudioServer(v.get("out_dir", "/data/tts"), int(v.get("serve_port", 8099))).start()
        if self.audio and v.get("piper_model"):
            self.speaker = Go2rtcSpeaker(
                go2rtc_url=v["go2rtc_url"], piper_model=v["piper_model"], public_url=v["public_url"],
                server=self.audio, codec=v.get("codec", "pcma"), streams=streams,
            )
        else:
            log.warning("voice not configured: spoken lines will only be logged")
            self.speaker = LogSpeaker()

        if backend == "gemini" and self.audio and op.get("api_key"):
            self.operator = GeminiLiveOperator(
                api_key=op["api_key"], server=self.audio, go2rtc_url=v["go2rtc_url"],
                public_url=v["public_url"], rtsp_url=op.get("rtsp_url", "rtsp://localhost:8554/{stream}"),
                frigate_url=cfg.frigate_url, model=op.get("model", "gemini-3.8-live"),
                voice=op.get("voice"), codec=v.get("codec", "pcma"), streams=streams,
                max_session_s=float(op.get("max_session_s", 180)),
            )
        elif backend == "log":
            self.operator = LogOperator()
        else:
            if backend == "gemini":
                log.warning("gemini operator needs voice.go2rtc_url, voice.public_url and operator.api_key; "
                            "falling back to one-way lines")
            self.operator = SpeakerOperator(self.speaker)

    # ---------------------------------------------------------------- inputs

    def sink(self, kind: str, obj) -> None:
        if kind == "mqtt_client":
            self.mqtt_client = obj
        else:
            self.q.put((kind, obj))

    def on_ha_event(self, kind: str, camera: str, ts: float) -> None:
        if kind == "doorbell":
            self.q.put(("doorbell", (camera, ts)))

    # test-mode controls (also used by scenes)
    def set_night(self, night: bool | None) -> None:
        self.engine.night_override = night

    def set_armed(self, armed: bool) -> None:
        with self.presence._lock:
            p = self.presence.presence
            if p.alarm_armed and not armed:
                p.last_disarm_ts = time.time()
            p.alarm_armed = armed

    def wifi_join(self, who: str) -> None:
        with self.presence._lock:
            self.presence.presence.family_phones_home.add(who)
            self.presence.presence.recent_wifi_joins[who] = time.time()
        self.events.add("presence", text=f"{who}'s phone joined Wi-Fi")

    # ---------------------------------------------------------------- actions

    def _area(self, camera: str) -> str:
        c = self.cfg.cameras.get(camera)
        return c.area if c else f"at the {camera.replace('_', ' ')}"

    def _night_now(self) -> bool:
        if self.engine.night_override is not None:
            return self.engine.night_override
        return _is_night(self.engine.local_hour(time.time()), self.cfg.engine.trust.night_hours)

    def _notify(self, a: Action, *, critical: bool = False) -> None:
        d = a.decision
        hour = self.engine.local_hour(time.time())
        title = ("Liveguard: talk-down not working" if critical
                 else f"Liveguard: {d.intent.value} {self._area(a.camera)}")
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
            if a.kind == "publish":
                if self.mqtt_client:
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
                line = a.text or warning_line(self.engine.describe(a.track_id), self._area(a.camera),
                                              self._night_now())
                self.events.add("speak", camera=a.camera, track=a.track_id, text=line)
                self.speaker.say(a.camera, line)
            elif a.kind == "deliberate":
                try:
                    v = self.deliberator.deliberate(a.request)
                except Exception as e:
                    log.warning("deliberator failed: %s", e)
                    v = Verdict(False, f"deliberator error ({type(e).__name__}); staying quiet", source="error")
                log.info("%s %s deliberation: speak=%s (%s) %s", a.camera, a.track_id, v.speak, v.source, v.reason)
                self.events.add("verdict", camera=a.camera, track=a.track_id, speak=v.speak,
                                text=v.reason, source=v.source)
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

    def _observe_decision(self, a: Action) -> None:
        d = a.decision
        with self._tracks_lock:
            prev = self.tracks.get(a.track_id)
            self.tracks[a.track_id] = {
                "track": a.track_id, "camera": a.camera, "level": d.level.name, "rung": int(d.level),
                "intent": d.intent.value, "trust": d.trust.score, "identity": d.trust.identity,
                "reasons": d.reasons, "trust_reasons": d.trust.reasons, "updated": time.time(),
            }
        if prev is None:
            self.events.add("seen", camera=a.camera, track=a.track_id, intent=d.intent.value,
                            text=f"Person seen {self._area(a.camera)}")

    def dispatch(self, actions: list[Action]) -> None:
        for a in actions:
            if a.kind == "publish":
                self._observe_decision(a)
            else:
                d = a.decision
                if d:
                    log.info("%s %s -> %s (%s)", a.camera, a.track_id, a.kind, "; ".join(d.reasons))
                self.events.add("action", action=a.kind, camera=a.camera, track=a.track_id,
                                level=d.level.name if d else None,
                                intent=d.intent.value if d else None,
                                reasons=d.reasons if d else [])
            self.pool.submit(self.execute, a)

    def _forget_ended(self, obs: Observation) -> None:
        if obs.ended:
            with self._tracks_lock:
                self.tracks.pop(obs.track_id, None)
            self.events.add("ended", camera=obs.camera, track=obs.track_id, text="Left the camera")

    def process(self, kind: str, obj, now: float | None = None) -> None:
        now = now or time.time()
        if kind == "tick":
            self.dispatch(self.engine.tick(now, self.presence.snapshot()))
            with self._tracks_lock:
                live = set(self.engine._last_obs)
                for tid in [t for t in self.tracks if t not in live]:
                    self.tracks.pop(tid, None)
        elif kind == "obs":
            self.dispatch(self.engine.handle(obj, self.presence.snapshot()))
            self._forget_ended(obj)
            if self.engine.level_of(obj.track_id) >= Level.OPERATOR:
                self.pool.submit(self.operator.update, obj.camera, {"description": obj.description})
        elif kind == "verdict":
            tid, v = obj
            self.engine.verdict(tid, v)
            self.dispatch(self.engine.tick(now, self.presence.snapshot()))
        elif kind == "doorbell":
            self.engine.doorbell(*obj)
            self.events.add("presence", camera=obj[0], text="Doorbell pressed")
        elif kind == "feedback":
            parsed = parse_feedback_action(str(obj.get("action", "")))
            if parsed:
                self.feedback.record(*parsed)
                self.feedback.save()
                self.events.add("feedback", camera=parsed[0],
                                text="Marked fine" if parsed[2] else "Marked sketchy")
        elif kind == "error":
            log.warning("bad MQTT message: %s", obj)

    def run(self) -> None:
        if not self.test_mode:
            m = self.cfg.mqtt
            threading.Thread(target=run_mqtt, daemon=True, name="mqtt", args=(
                m.get("host", "localhost"), int(m.get("port", 1883)), m.get("username"), m.get("password"),
                FrigateParser(), self.sink)).start()
            threading.Thread(target=poll_presence, daemon=True, name="ha", args=(
                self.ha, self.presence, self.on_ha_event),
                kwargs={"on_error": lambda e: log.warning("HA poll failed: %s", e)}).start()

        log.info("liveguard running%s", " in test mode" if self.test_mode else "")
        last_tick = 0.0
        while True:
            try:
                kind, obj = self.q.get(timeout=1.0)
            except queue.Empty:
                kind, obj = "tick", None
            now = time.time()
            if kind != "tick" and now - last_tick >= 2.0:
                self.process("tick", None, now)
            if kind == "tick":
                last_tick = now
            self.process(kind, obj, now)

    def restart(self) -> None:
        """Re-exec in place so a saved config takes effect (Docker or desktop)."""
        log.info("restarting to apply new config")
        os.execv(sys.executable, [sys.executable, "-m", "liveguard.app", *sys.argv[1:]])


def main() -> None:
    ap = argparse.ArgumentParser(prog="liveguard")
    ap.add_argument("-c", "--config", default="/config/liveguard.yaml")
    ap.add_argument("--test", action="store_true",
                    help="test mode: simulated scenes, desktop speakers/mic, no Frigate or HA needed")
    ap.add_argument("--webui-host", default=None)
    ap.add_argument("--webui-port", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    path = Path(args.config)
    if args.test and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DEMO_CONFIG, path)
        log.info("no config at %s: wrote a demo config you can edit in the web UI", path)

    cfg = config_mod.load(str(path))
    app = App(cfg, test_mode=args.test, config_path=str(path))

    from .webui import WebUI
    host = args.webui_host or cfg.webui.get("host") or ("127.0.0.1" if args.test else "0.0.0.0")
    port = args.webui_port or int(cfg.webui.get("port", 8080))
    ui = WebUI(app, host=host, port=port, password=cfg.webui.get("password")).start()
    log.info("web UI on http://%s:%d", "localhost" if host in ("0.0.0.0", "127.0.0.1") else host, ui.port)
    if host == "0.0.0.0" and not cfg.webui.get("password"):
        log.warning("web UI is open to your network with no password; set webui.password")
    app.run()


if __name__ == "__main__":
    main()
