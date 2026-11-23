"""Web UI: live view, test bench and settings. Stdlib HTTP server, one static page."""
from __future__ import annotations

import base64
import copy
import hmac
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config as config_mod
from .models import Observation

log = logging.getLogger("liveguard.webui")

INDEX = Path(__file__).with_name("web") / "index.html"
MASK = "********"
SECRET_KEYS = {"token", "password", "api_key"}


# ------------------------------------------------------------------ config helpers

def mask_secrets(d):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k in SECRET_KEYS and isinstance(v, str) and v and not v.startswith("${"):
                out[k] = MASK
            else:
                out[k] = mask_secrets(v)
        return out
    if isinstance(d, list):
        return [mask_secrets(x) for x in d]
    return d


def unmask_secrets(new, old):
    """Put real secrets back wherever the UI sent the mask unchanged."""
    if isinstance(new, dict):
        old = old if isinstance(old, dict) else {}
        return {k: (old.get(k) if v == MASK and k in SECRET_KEYS else unmask_secrets(v, old.get(k)))
                for k, v in new.items()}
    if isinstance(new, list):
        return [unmask_secrets(x, None) for x in new]
    return new


def read_raw(path: str) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text()) or {}


def save_config(path: str, raw: dict) -> None:
    """Validate by loading, keep a .bak, then replace atomically."""
    import yaml
    text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    p = Path(path)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".liveguard-", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    try:
        config_mod.load(tmp)  # raises on anything the app couldn't start with
    except Exception:
        os.unlink(tmp)
        raise
    if p.exists():
        shutil.copy(p, p.with_suffix(p.suffix + ".bak"))
    os.replace(tmp, p)


# ------------------------------------------------------------------ server

class WebUI:
    def __init__(self, app, host: str = "127.0.0.1", port: int = 8080, password: str | None = None):
        self.app = app
        self.password = password or None
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _authed(self) -> bool:
                if not ui.password:
                    return True
                h = self.headers.get("Authorization", "")
                if h.startswith("Basic "):
                    try:
                        _, _, pw = base64.b64decode(h[6:]).decode().partition(":")
                        if hmac.compare_digest(pw, ui.password):
                            return True
                    except Exception:
                        pass
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="Liveguard"')
                self.end_headers()
                return False

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            def do_GET(self):
                if not self._authed():
                    return
                url = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(url.query)
                if url.path in ("/", "/index.html"):
                    body = INDEX.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                if url.path == "/api/state":
                    return self._json(ui.state())
                if url.path == "/api/events":
                    events, last = ui.app.events.since(int(q.get("since", ["0"])[0]))
                    return self._json({"events": events, "last": last})
                if url.path == "/api/config":
                    return self._json(ui.get_config())
                self.send_error(404)

            def do_POST(self):
                if not self._authed():
                    return
                path = urllib.parse.urlparse(self.path).path
                try:
                    body = self._body()
                except ValueError:
                    return self._json({"ok": False, "error": "Request body isn't valid JSON."}, 400)
                handler = ui.routes.get(path)
                if not handler:
                    return self.send_error(404)
                try:
                    return self._json(handler(body))
                except UserError as e:
                    return self._json({"ok": False, "error": str(e)}, 400)
                except Exception as e:
                    log.exception("web UI %s failed", path)
                    return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

            do_PUT = do_POST

        self.routes = {
            "/api/config": self.put_config,
            "/api/restart": self.restart,
            "/api/sim/scene": self.play_scene,
            "/api/sim/observe": self.observe,
            "/api/sim/presence": self.set_presence,
            "/api/sim/stop": self.stop_sim,
            "/api/feedback": self.feedback,
        }
        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]

    def start(self) -> "WebUI":
        threading.Thread(target=self._httpd.serve_forever, daemon=True, name="webui").start()
        return self

    def shutdown(self) -> None:
        self._httpd.shutdown()

    # ------------------------------------------------------------ reads

    def state(self) -> dict:
        a = self.app
        p = a.presence.snapshot()
        with a._tracks_lock:
            tracks = sorted(a.tracks.values(), key=lambda t: -t["rung"])
        z = a.cfg.engine.intent.zones
        return {
            "test_mode": a.test_mode,
            "night_override": a.engine.night_override,
            "night": a._night_now(),
            "armed": p.alarm_armed,
            "phones_home": sorted(p.family_phones_home),
            "tracks": tracks,
            "cameras": [{"id": c, "area": cc.area} for c, cc in a.cfg.cameras.items()],
            "zones": {"on_property": sorted(z.on_property), "off_property": sorted(z.off_property),
                      "door": sorted(z.door), "sensitive": sorted(z.sensitive)},
            "family": sorted(a.cfg.engine.trust.family | a.cfg.engine.trust.trusted),
            "scenes": [{"id": s.id, "name": s.name, "blurb": s.blurb} for s in a.scenes],
            "operator": type(a.operator).__name__,
            "deliberator": type(a.deliberator).__name__,
            "config_path": a.config_path,
            "now": time.time(),
        }

    def get_config(self) -> dict:
        import yaml
        raw = read_raw(self.app.config_path)
        masked = mask_secrets(raw)
        return {"config": masked, "yaml": yaml.safe_dump(masked, sort_keys=False, allow_unicode=True),
                "path": self.app.config_path}

    # ------------------------------------------------------------ writes

    def put_config(self, body: dict) -> dict:
        import yaml
        old = read_raw(self.app.config_path)
        if "yaml" in body:
            try:
                new = yaml.safe_load(body["yaml"]) or {}
            except yaml.YAMLError as e:
                raise UserError(f"That YAML doesn't parse: {e}")
            if not isinstance(new, dict):
                raise UserError("The config must be a YAML mapping (key: value pairs) at the top level.")
        else:
            new = body.get("config")
            if not isinstance(new, dict):
                raise UserError("Nothing to save.")
        new = unmask_secrets(new, old)
        try:
            save_config(self.app.config_path, new)
        except UserError:
            raise
        except Exception as e:
            raise UserError(f"Not saved. Liveguard couldn't start with this config: {e}")
        self.app.events.add("config", text="Settings saved")
        return {"ok": True, "restart_required": True}

    def restart(self, _body: dict) -> dict:
        threading.Timer(0.4, self.app.restart).start()
        return {"ok": True}

    def _require_test(self):
        if not self.app.test_mode:
            raise UserError("The test bench only runs in test mode. Start Liveguard with --test.")

    def play_scene(self, body: dict) -> dict:
        self._require_test()
        scene = next((s for s in self.app.scenes if s.id == body.get("id")), None)
        if not scene:
            raise UserError("Unknown scene.")
        self.app.events.add("scene", text=f"Started scene: {scene.name}")
        tid = self.app.sim.play(scene)
        return {"ok": True, "track": tid}

    def observe(self, body: dict) -> dict:
        self._require_test()
        cam = body.get("camera") or next(iter(self.app.cfg.cameras), "front_door")
        zones = [z for z in body.get("zones", []) if isinstance(z, str)]
        obs = Observation(
            track_id=body.get("track") or "manual-1", camera=cam, ts=time.time(), zones=zones,
            description=(body.get("description") or None), stationary=bool(body.get("stationary")),
            sub_label=(body.get("sub_label") or None), sub_label_score=float(body.get("sub_label_score") or 0),
            has_package_nearby=bool(body.get("package")), ended=bool(body.get("ended")),
        )
        self.app.sink("obs", obs)
        return {"ok": True, "track": obs.track_id}

    def set_presence(self, body: dict) -> dict:
        self._require_test()
        a = self.app
        if "night" in body:
            a.set_night(body["night"] if body["night"] in (True, False) else None)
        if "armed" in body:
            a.set_armed(bool(body["armed"]))
            a.events.add("presence", text="Alarm armed" if body["armed"] else "Alarm disarmed")
        if body.get("wifi_join"):
            a.wifi_join(str(body["wifi_join"]))
        if body.get("doorbell"):
            a.sink("doorbell", (str(body["doorbell"]), time.time()))
        return {"ok": True}

    def stop_sim(self, _body: dict) -> dict:
        self._require_test()
        self.app.sim.stop_all()
        return {"ok": True}

    def feedback(self, body: dict) -> dict:
        cam, fine = body.get("camera"), body.get("fine")
        if not cam or fine not in (True, False):
            raise UserError("Feedback needs a camera and fine: true/false.")
        hour = self.app.engine.local_hour(time.time())
        tag = "LIVEGUARD_FINE" if fine else "LIVEGUARD_SKETCHY"
        self.app.sink("feedback", {"action": f"{tag}|{cam}|{hour}"})
        return {"ok": True}


class UserError(Exception):
    pass
