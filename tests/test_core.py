import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from liveguard.adapters.frigate import FrigateParser
from liveguard.adapters.homeassistant import HAEntities, HAPresenceTracker
from liveguard.app import parse_feedback_action
from liveguard.deliberate import RulesDeliberator, Verdict, parse_verdict
from liveguard.engine import Engine, EngineConfig
from liveguard.feedback import FeedbackConfig, FeedbackStore
from liveguard.intent import IntentConfig, IntentTracker
from liveguard.models import Intent, Level, Observation, Presence, TrustResult
from liveguard.policy import Policy, PolicyConfig
from liveguard.trust import TrustConfig, score
from liveguard.voice import short_description, warning_line

TZ = ZoneInfo("America/Chicago")


def at(hour, minute=0):
    return datetime(2026, 11, 22, hour, minute, tzinfo=TZ).timestamp()


TRUST = TrustConfig(family={"scarlett", "mom"}, face_to_person={"scarlett": "scarlett", "mom": "mom"})


def obs(tid="t1", cam="front_door", ts=None, **kw):
    return Observation(track_id=tid, camera=cam, ts=ts if ts is not None else at(14), **kw)


# ------------------------------------------------------------------ trust

def test_strong_face_in_daytime_is_family():
    r = score(obs(sub_label="scarlett", sub_label_score=0.92), Presence(), TRUST, hour=14)
    assert r.is_family and r.identity == "scarlett"


def test_face_alone_at_night_is_capped():
    r = score(obs(sub_label="mom", sub_label_score=0.95), Presence(), TRUST, hour=2)
    assert not r.is_family
    assert "capped" in r.reasons[-1]


def test_face_plus_wifi_at_night_is_family():
    ts = at(2)
    p = Presence(recent_wifi_joins={"mom": ts - 30})
    r = score(obs(ts=ts, sub_label="mom", sub_label_score=0.7), p, TRUST, hour=2)
    assert r.is_family


def test_stranger_scores_zero():
    r = score(obs(), Presence(), TRUST, hour=14)
    assert r.score == 0 and not r.is_family


def test_old_wifi_join_is_ignored():
    ts = at(14)
    r = score(obs(ts=ts), Presence(recent_wifi_joins={"mom": ts - 3600}), TRUST, hour=14)
    assert r.score == 0


# ------------------------------------------------------------------ intent

def run_track(steps, night=False, doorbell_at=None):
    tr = IntentTracker(IntentConfig())
    state = None
    for i, o in enumerate(steps):
        state = tr.update(o, doorbell=doorbell_at is not None and i >= doorbell_at)
    return tr.classify(state, night=night)[0]


def test_delivery():
    t0 = at(14)
    steps = [obs(ts=t0, zones=["driveway"]),
             obs(ts=t0 + 20, zones=["porch"], has_package_nearby=True)]
    assert run_track(steps) is Intent.DELIVERY


def test_delivery_by_description():
    t0 = at(14)
    steps = [obs(ts=t0, zones=["porch"], description="A person in a USPS uniform holding a box.")]
    assert run_track(steps) is Intent.DELIVERY


def test_passerby():
    assert run_track([obs(zones=["sidewalk"]), obs(ts=at(14) + 5, zones=["street"])]) is Intent.PASSERBY


def test_loitering_by_dwell():
    t0 = at(14)
    steps = [obs(ts=t0, zones=["driveway"]), obs(ts=t0 + 120, zones=["driveway"])]
    assert run_track(steps) is Intent.LOITERING


def test_loitering_by_sitting_still():
    t0 = at(14)
    steps = [obs(ts=t0, zones=["porch"], stationary=True), obs(ts=t0 + 50, zones=["porch"], stationary=True)]
    assert run_track(steps) is Intent.LOITERING


def test_backyard_at_night_is_suspicious():
    assert run_track([obs(ts=at(2), zones=["backyard"])], night=True) is Intent.SUSPICIOUS


def test_trying_car_doors_is_suspicious_in_daytime():
    steps = [obs(zones=["driveway"], description="A man pulling on the car door handle of a sedan.")]
    assert run_track(steps) is Intent.SUSPICIOUS


def test_visitor_rings():
    t0 = at(14)
    steps = [obs(ts=t0, zones=["porch"]), obs(ts=t0 + 10, zones=["porch"])]
    assert run_track(steps, doorbell_at=1) is Intent.VISITOR


def test_repeat_visits_but_family_doesnt_count():
    tr = IntentTracker(IntentConfig())
    t0 = at(14)
    for i in range(3):
        tr.update(obs(tid=f"fam{i}", ts=t0 + i * 60, zones=["porch"]))
        tr.mark_trusted(f"fam{i}")
    s = tr.update(obs(tid="x", ts=t0 + 200, zones=["porch"]))
    assert tr.classify(s, night=False)[0] is Intent.UNKNOWN


# ------------------------------------------------------------------ policy

STRANGER = TrustResult(0.0, None, [])
FAMILY = TrustResult(0.9, "mom", [])


def test_policy_climbs_one_rung_per_step_then_pages():
    p = Policy(PolicyConfig(step_s=10, operator_max_s=30))
    t = at(2)
    levels = [p.decide("a", "cam", Intent.LOITERING, STRANGER, now=t + i * 10, night=True, armed=False).level
              for i in range(8)]
    assert levels[:4] == [Level.NOTIFY, Level.LIGHTS, Level.WARN, Level.OPERATOR]
    assert Level.ESCALATE_HUMAN in levels
    assert levels[-1] is Level.ESCALATE_HUMAN  # sticky once paged


def test_policy_suspicious_speaks_immediately():
    p = Policy()
    d = p.decide("a", "cam", Intent.SUSPICIOUS, STRANGER, now=at(2), night=True, armed=False)
    assert d.level is Level.WARN


def test_policy_no_siren_unless_armed():
    p = Policy(PolicyConfig(step_s=0, operator_max_s=999))
    t = at(2)
    for i in range(6):
        d = p.decide("a", "cam", Intent.SUSPICIOUS, STRANGER, now=t + i, night=True, armed=False)
    assert d.level is Level.OPERATOR
    p2 = Policy(PolicyConfig(step_s=0, operator_max_s=999))
    for i in range(6):
        d = p2.decide("a", "cam", Intent.SUSPICIOUS, STRANGER, now=t + i, night=True, armed=True)
    assert d.level is Level.SIREN


def test_policy_deescalates_when_family_recognised():
    p = Policy(PolicyConfig(step_s=0))
    t = at(2)
    for i in range(3):
        p.decide("a", "cam", Intent.LOITERING, STRANGER, now=t + i, night=True, armed=False)
    d = p.decide("a", "cam", Intent.LOITERING, FAMILY, now=t + 5, night=True, armed=False)
    assert d.level is Level.SILENT


def test_partial_trust_never_confronts():
    p = Policy(PolicyConfig(step_s=0))
    half = TrustResult(0.5, "mom", [])
    for i in range(6):
        d = p.decide("a", "cam", Intent.LOITERING, half, now=at(2) + i, night=True, armed=True)
    assert d.level is Level.NOTIFY


# ------------------------------------------------------------------ engine end-to-end

def kinds(actions):
    return [a.kind for a in actions if a.kind != "publish"]


def test_engine_family_walking_up_is_silent():
    e = Engine(EngineConfig(trust=TRUST))
    acts = e.handle(obs(sub_label="scarlett", sub_label_score=0.9, zones=["porch"]), Presence())
    assert kinds(acts) == []


def test_engine_delivery_quiet_when_configured():
    cfg = EngineConfig(trust=TRUST, policy=PolicyConfig(notify_deliveries=False))
    e = Engine(cfg)
    acts = e.handle(obs(zones=["porch"], has_package_nearby=True), Presence())
    assert kinds(acts) == []


def test_engine_night_prowler_thinks_then_speaks():
    e = Engine(EngineConfig(trust=TRUST, policy=PolicyConfig(step_s=10, operator_max_s=30)))
    t = at(2)
    first = e.handle(obs(ts=t, zones=["backyard"]), Presence())
    assert kinds(first) == ["deliberate", "notify", "lights"]   # holds before speaking
    req = next(a.request for a in first if a.kind == "deliberate")
    assert req.on_property and req.intent == "suspicious"
    e.verdict("t1", Verdict(True, "backyard at 2am", line="You by the fence, in the dark jacket."))
    second = e.handle(obs(ts=t + 1, zones=["backyard"]), Presence())
    warn = [a for a in second if a.kind == "warn"]
    assert kinds(second) == ["warn"] and warn[0].text.startswith("You by the fence")
    assert kinds(e.handle(obs(ts=t + 12, zones=["backyard"]), Presence())) == ["operator_start"]
    end = kinds(e.handle(obs(ts=t + 15, zones=["backyard"], ended=True), Presence()))
    assert end == ["operator_stop"]


def test_engine_deliberator_says_no():
    e = Engine(EngineConfig(trust=TRUST, policy=PolicyConfig(step_s=0)))
    t = at(2)
    e.handle(obs(ts=t, zones=["driveway"], stationary=True), Presence())
    assert "deliberate" in kinds(e.handle(obs(ts=t + 50, zones=["driveway"], stationary=True), Presence()))
    e.verdict("t1", Verdict(False, "looks like the neighbour's kid getting a ball"))
    later = []
    for i in range(1, 6):
        later += kinds(e.handle(obs(ts=t + 50 + i, zones=["driveway"], stationary=True), Presence()))
    assert "warn" not in later and "operator_start" not in later and "deliberate" not in later


def test_engine_reasks_when_behaviour_gets_worse():
    e = Engine(EngineConfig(trust=TRUST, policy=PolicyConfig(step_s=0)))
    t = at(2)
    e.handle(obs(ts=t, zones=["porch"], stationary=True), Presence())
    acts = e.handle(obs(ts=t + 50, zones=["porch"], stationary=True), Presence())
    assert "deliberate" in kinds(acts)  # loitering -> ask
    e.verdict("t1", Verdict(False, "probably waiting for a ride"))
    acts = e.handle(obs(ts=t + 55, zones=["porch"], stationary=True,
                        description="Person peering into the front window."), Presence())
    assert "deliberate" in kinds(acts)  # now suspicious -> ask again


def test_engine_timeout_holds_by_default():
    e = Engine(EngineConfig(trust=TRUST, deliberation_timeout_s=5, policy=PolicyConfig(step_s=0)))
    t = at(2)
    e.handle(obs(ts=t, zones=["backyard"]), Presence())
    acts = kinds(e.tick(t + 6, Presence()))
    assert "warn" not in acts


def test_engine_timeout_can_fall_back_to_rules():
    e = Engine(EngineConfig(trust=TRUST, deliberation_timeout_s=5, on_timeout="rules",
                            policy=PolicyConfig(step_s=0)))
    t = at(2)
    e.handle(obs(ts=t, zones=["backyard"]), Presence())
    assert "warn" in kinds(e.tick(t + 6, Presence()))


def test_engine_never_speaks_off_property():
    e = Engine(EngineConfig(trust=TRUST, policy=PolicyConfig(step_s=0)))
    t = at(2)
    acts = []
    for i in range(6):
        acts += kinds(e.handle(obs(ts=t + i * 20, zones=["sidewalk"], stationary=True,
                                   description="Person looking into the windows from the sidewalk"),
                               Presence(alarm_armed=True)))
    assert set(acts) <= {"notify"}


def test_engine_stops_talking_when_they_step_off():
    e = Engine(EngineConfig(trust=TRUST, policy=PolicyConfig(step_s=0, operator_max_s=999)))
    t = at(2)
    e.handle(obs(ts=t, zones=["backyard"]), Presence())
    e.verdict("t1", Verdict(True, "prowler"))
    for i in range(1, 4):
        e.handle(obs(ts=t + i, zones=["backyard"]), Presence())
    assert e.level_of("t1") >= Level.OPERATOR
    acts = kinds(e.handle(obs(ts=t + 10, zones=["sidewalk"]), Presence()))
    assert acts == ["operator_stop"]


def test_explicit_property_zone():
    from liveguard.intent import ZoneConfig
    z = ZoneConfig(on_property={"property"})
    assert z.is_on_property(["property", "driveway"])
    assert not z.is_on_property(["driveway"])  # outside the drawn lot line


def test_parse_verdict_is_lenient_and_safe():
    v = parse_verdict('thinking... {"speak": true, "reason": "prowler", "line": "Hey you."}', "t")
    assert v.speak and v.line == "Hey you."
    assert not parse_verdict("no json here", "t").speak
    assert parse_verdict('{"speak": false, "reason": "kid", "line": "ignored"}', "t").line is None


def test_rules_deliberator():
    from liveguard.deliberate import DeliberationRequest
    base = dict(track_id="t", camera="c", camera_area="yard", local_time="2am", night=True, armed=False,
                intent="suspicious", intent_reasons=["x"], trust_score=0.0, trust_identity=None,
                trust_reasons=[], zones_now=["backyard"], on_property=True, seconds_on_camera=5)
    r = RulesDeliberator()
    assert r.deliberate(DeliberationRequest(**base)).speak
    assert not r.deliberate(DeliberationRequest(**base | {"on_property": False})).speak
    assert not r.deliberate(DeliberationRequest(**base | {"trust_score": 0.3})).speak


def test_engine_learned_quiet(tmp_path):
    fb = FeedbackStore(FeedbackConfig(path=str(tmp_path / "f.json"), min_samples=3))
    for _ in range(4):
        fb.record("front_door", 18, True)
    e = Engine(EngineConfig(trust=TRUST), fb)
    acts = e.handle(obs(ts=at(18), zones=["driveway"]), Presence())
    assert kinds(acts) == []
    # ...but learning never quiets suspicious behaviour
    acts = e.handle(obs(tid="t2", ts=at(18), zones=["driveway"],
                        description="Person trying handle on the car door"), Presence())
    assert "deliberate" in kinds(acts) and "notify" in kinds(acts)


def test_feedback_roundtrip(tmp_path):
    cfg = FeedbackConfig(path=str(tmp_path / "f.json"), min_samples=2)
    fb = FeedbackStore(cfg)
    fb.record("cam", 3, True)
    fb.record("cam", 3, True)
    fb.save()
    assert FeedbackStore.load(cfg).is_quiet("cam", 4)  # hour slop
    fb.record("cam", 3, False)
    assert not fb.is_quiet("cam", 3)


# ------------------------------------------------------------------ adapters

def ev(kind, **after):
    base = {"id": "1700.1-abc", "camera": "front_door", "label": "person", "frame_time": at(14),
            "current_zones": ["porch"], "stationary": False, "sub_label": None}
    base.update(after)
    return json.dumps({"type": kind, "before": {}, "after": base})


def test_frigate_parser_basic_and_face_list():
    p = FrigateParser()
    o = p.on_event(ev("update", sub_label=["mom", 0.91]))
    assert o.sub_label == "mom" and o.sub_label_score == pytest.approx(0.91) and o.zones == ["porch"]


def test_frigate_parser_merges_description_face_and_package():
    p = FrigateParser()
    p.on_event(ev("new", id="pkg1", label="package"))
    p.on_tracked_update(json.dumps({"type": "description", "id": "1700.1-abc", "description": "A man in a red jacket."}))
    p.on_tracked_update(json.dumps({"type": "face", "id": "1700.1-abc", "name": "scarlett", "score": 0.8}))
    o = p.on_event(ev("update"))
    assert o.description == "A man in a red jacket."
    assert o.sub_label == "scarlett"
    assert o.has_package_nearby
    end = p.on_event(ev("end"))
    assert end.ended
    assert p.on_event(ev("update")).description is None  # cleared after end


def test_frigate_parser_ignores_cars():
    assert FrigateParser().on_event(ev("new", label="car")) is None


def test_ha_presence():
    t = HAPresenceTracker(HAEntities(
        wifi_trackers={"device_tracker.mom_phone": "mom"},
        alarm_panel="alarm_control_panel.home",
        doorbells={"binary_sensor.doorbell": "front_door"},
    ))
    t.apply("device_tracker.mom_phone", "not_home", 1)
    t.apply("device_tracker.mom_phone", "home", 100)
    t.apply("alarm_control_panel.home", "armed_away", 1)
    t.apply("alarm_control_panel.home", "disarmed", 200)
    t.apply("binary_sensor.doorbell", "off", 1)
    events = t.apply("binary_sensor.doorbell", "on", 300)
    snap = t.snapshot()
    assert snap.recent_wifi_joins["mom"] == 100 and "mom" in snap.family_phones_home
    assert snap.last_disarm_ts == 200 and not snap.alarm_armed
    assert events == [("doorbell", "front_door")]


def test_ha_first_poll_is_not_a_join():
    t = HAPresenceTracker(HAEntities(wifi_trackers={"device_tracker.p": "mom"}))
    t.apply("device_tracker.p", "home", 5)
    assert "mom" not in t.presence.recent_wifi_joins


# ------------------------------------------------------------------ voice / misc

def test_short_description():
    assert short_description("A man wearing a gray hoodie and jeans, carrying a bag.") == "gray hoodie and jeans"
    assert short_description("Someone near the gate.") is None


def test_warning_line_mentions_clothing():
    line = warning_line("A woman wearing a blue raincoat.", "by the side gate", night=True)
    assert "blue raincoat" in line and "on camera by the side gate" in line and "police" not in line.lower()
    assert short_description("A man wearing a dark hoodie near the fence.") == "dark hoodie"


def test_innocent_descriptions_are_not_suspicious():
    for d in ("A man in a dark hoodie standing near the fence.", "A woman walking past the front window.",
              "A child picking up a ball by the gate."):
        assert run_track([obs(zones=["driveway"], description=d)]) is not Intent.SUSPICIOUS


def test_parse_feedback_action():
    assert parse_feedback_action("LIVEGUARD_FINE|front_door|19") == ("front_door", 19, True)
    assert parse_feedback_action("LIVEGUARD_SKETCHY|cam|3") == ("cam", 3, False)
    assert parse_feedback_action("OTHER|x|1") is None


def test_example_config_loads(monkeypatch):
    from pathlib import Path
    from liveguard import config
    monkeypatch.setenv("HA_TOKEN", "abc")
    cfg = config.load(str(Path(__file__).parent.parent / "config.example.yaml"))
    assert cfg.ha_token == "abc"
    assert cfg.engine.intent.zones.on_property == {"property"}
    assert cfg.engine.policy.max_level is Level.ESCALATE_HUMAN
    assert cfg.engine.camera_areas["backyard"] == "in the back yard"
    assert cfg.deliberation["backend"] == "ollama"
