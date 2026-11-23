import base64
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from liveguard import config as config_mod
from liveguard.app import DEMO_CONFIG, App, EventLog
from liveguard.deliberate import RulesDeliberator
from liveguard.engine import Engine
from liveguard.models import Observation, Presence
from liveguard.webui import MASK, WebUI, mask_secrets, unmask_secrets


@pytest.fixture
def demo_cfg_path(tmp_path, monkeypatch):
    p = tmp_path / "liveguard.yaml"
    shutil.copy(DEMO_CONFIG, p)
    text = p.read_text().replace("/data/feedback.json", str(tmp_path / "fb.json"))
    p.write_text(text)
    return p


# ------------------------------------------------------------------ scenes, run deterministically

def run_scene(scene_id, cfg_path):
    cfg = config_mod.load(str(cfg_path))
    cfg.feedback.path = str(cfg_path.parent / "fb.json")
    app = App(cfg, test_mode=True, config_path=str(cfg_path))
    scene = next(s for s in app.scenes if s.id == scene_id)
    eng: Engine = app.engine
    eng.night_override = scene.night
    p = Presence(alarm_armed=bool(scene.armed))
    t0 = 1_800_000_000.0
    if scene.wifi_join:
        p.recent_wifi_joins[scene.wifi_join] = t0
    rules = RulesDeliberator()
    cam = "backyard" if scene.camera_hint == "back" else "front_door"
    kinds = []
    for at, fields in sorted(scene.steps, key=lambda s: s[0]):
        for a in eng.handle(Observation(track_id="s", camera=cam, ts=t0 + at, **fields), p):
            kinds.append(a.kind)
            if a.kind == "deliberate":
                eng.verdict("s", rules.deliberate(a.request))
    return [k for k in kinds if k != "publish"]


def test_scene_prowler_thinks_then_speaks(demo_cfg_path):
    k = run_scene("prowler", demo_cfg_path)
    assert k.index("deliberate") < k.index("warn")


def test_scene_delivery_never_speaks(demo_cfg_path):
    k = run_scene("delivery", demo_cfg_path)
    assert "warn" not in k and "lights" not in k


def test_scene_family_is_silent(demo_cfg_path):
    assert run_scene("family", demo_cfg_path) == []


def test_scene_porch_sitter_in_daylight_is_not_confronted(demo_cfg_path):
    k = run_scene("porch-sitter", demo_cfg_path)
    assert "deliberate" in k and "warn" not in k


def test_scene_sidewalk_only_notifies(demo_cfg_path):
    assert set(run_scene("sidewalk", demo_cfg_path)) <= {"notify"}


# ------------------------------------------------------------------ secrets

def test_mask_and_unmask_round_trip():
    raw = {"home_assistant": {"token": "abc", "url": "x"}, "mqtt": {"password": "${MQTT_PASSWORD}"},
           "operator": {"api_key": ""}}
    m = mask_secrets(raw)
    assert m["home_assistant"]["token"] == MASK
    assert m["mqtt"]["password"] == "${MQTT_PASSWORD}"   # env refs aren't secret
    assert m["operator"]["api_key"] == ""
    m["home_assistant"]["url"] = "y"
    back = unmask_secrets(m, raw)
    assert back["home_assistant"] == {"token": "abc", "url": "y"}


def test_event_log_since():
    log = EventLog(size=3)
    for i in range(5):
        log.add("x", n=i)
    events, last = log.since(3)
    assert last == 5 and [e["n"] for e in events] == [3, 4]


# ------------------------------------------------------------------ HTTP

@pytest.fixture
def ui(demo_cfg_path):
    cfg = config_mod.load(str(demo_cfg_path))
    cfg.feedback.path = str(demo_cfg_path.parent / "fb.json")
    app = App(cfg, test_mode=True, config_path=str(demo_cfg_path))
    server = WebUI(app, host="127.0.0.1", port=0).start()
    yield app, server
    server.shutdown()


def call(server, path, body=None, method=None, auth=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{server.port}{path}", data=data,
                                 method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    if auth:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"u:{auth}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_index_and_state(ui):
    app, s = ui
    code, body = call(s, "/")
    assert code == 200 and b"Liveguard" in body
    code, body = call(s, "/api/state")
    st = json.loads(body)
    assert st["test_mode"] and len(st["scenes"]) == 5 and st["cameras"][0]["id"] == "front_door"


def test_config_save_and_validation(ui, demo_cfg_path):
    app, s = ui
    cfg = json.loads(call(s, "/api/config")[1])["config"]
    cfg["policy"]["max_level"] = "warn"
    code, body = call(s, "/api/config", {"config": cfg}, "PUT")
    assert code == 200 and json.loads(body)["restart_required"]
    assert "max_level: warn" in demo_cfg_path.read_text()
    assert Path(str(demo_cfg_path) + ".bak").exists()

    code, body = call(s, "/api/config", {"yaml": "policy:\n  max_level: nuke\n"}, "PUT")
    assert code == 400 and "must be one of" in json.loads(body)["error"]
    assert "max_level: warn" in demo_cfg_path.read_text()   # bad save didn't touch the file


def test_sim_endpoints_drive_the_app(ui):
    app, s = ui
    assert call(s, "/api/sim/presence", {"night": True, "armed": True})[0] == 200
    assert app.engine.night_override is True and app.presence.snapshot().alarm_armed
    call(s, "/api/sim/observe", {"camera": "backyard", "zones": ["property", "backyard"],
                                  "description": "A man peering into the windows."})
    kind, obs = app.q.get(timeout=2)
    assert kind == "obs" and obs.camera == "backyard" and obs.zones == ["property", "backyard"]
    app.process(kind, obs)
    assert any(t["camera"] == "backyard" for t in app.tracks.values())


def test_feedback_endpoint(ui):
    app, s = ui
    assert call(s, "/api/feedback", {"camera": "front_door", "fine": True})[0] == 200
    kind, obj = app.q.get(timeout=2)
    app.process(kind, obj)
    assert sum(c[0] for c in app.feedback.counts.values()) == 1


def test_bench_is_refused_outside_test_mode(ui):
    app, s = ui
    app.test_mode = False
    code, body = call(s, "/api/sim/scene", {"id": "prowler"})
    assert code == 400 and "--test" in json.loads(body)["error"]


def test_password_required_when_set(demo_cfg_path):
    cfg = config_mod.load(str(demo_cfg_path))
    cfg.feedback.path = str(demo_cfg_path.parent / "fb.json")
    app = App(cfg, test_mode=True, config_path=str(demo_cfg_path))
    s = WebUI(app, host="127.0.0.1", port=0, password="pw").start()
    try:
        assert call(s, "/api/state")[0] == 401
        assert call(s, "/api/state", auth="wrong")[0] == 401
        assert call(s, "/api/state", auth="pw")[0] == 200
    finally:
        s.shutdown()


def test_test_mode_records_home_assistant_calls(demo_cfg_path):
    cfg = config_mod.load(str(demo_cfg_path))
    cfg.feedback.path = str(demo_cfg_path.parent / "fb.json")
    app = App(cfg, test_mode=True, config_path=str(demo_cfg_path))
    app.ha.call("light", "turn_on", {"entity_id": ["light.porch"]})
    events, _ = app.events.since(0)
    assert events[-1]["kind"] == "ha" and "light.porch" in events[-1]["text"]
