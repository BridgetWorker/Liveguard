import asyncio
import struct
import threading
import time
import urllib.request
from types import SimpleNamespace

from liveguard.audio_server import FRAME_BYTES, AudioServer, LiveRelay, wav_header
from liveguard.operator_gemini import GeminiLiveOperator


# ------------------------------------------------------------------ relay / server

def test_relay_pads_flushes_and_tracks_playback():
    clock = [0.0]
    r = LiveRelay(clock=lambda: clock[0])
    r.open("cam")
    assert r.take("cam", 4) == b"\0\0\0\0" and not r.is_playing("cam")
    r.feed("cam", b"\1\1\1\1\1\1")
    assert r.take("cam", 4) == b"\1\1\1\1" and r.is_playing("cam")
    r.flush("cam")
    assert r.take("cam", 4) == b"\0\0\0\0"
    assert r.is_playing("cam")        # echo tail
    clock[0] += 1.0
    assert not r.is_playing("cam")
    r.close("cam")
    r.feed("cam", b"\2\2")
    assert r.take("cam", 2) == b"\0\0"  # closed channels drop audio


def test_wav_header_is_24k_mono_16bit():
    h = wav_header()
    assert h[:4] == b"RIFF" and h[8:12] == b"WAVE" and len(h) == 44
    _, fmt, ch, rate, _, _, bits = struct.unpack("<IHHIIHH", h[16:36])
    assert (fmt, ch, rate, bits) == (1, 1, 24000, 16)


def test_server_streams_live_audio_in_real_time(tmp_path):
    srv = AudioServer(str(tmp_path), port=0, host="127.0.0.1").start()
    try:
        srv.relay.open("porch")
        srv.relay.feed("porch", b"\x07" * FRAME_BYTES)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/live/porch.wav", timeout=5)
        t0 = time.monotonic()
        head = resp.read(44)
        first = resp.read(FRAME_BYTES)
        rest = resp.read(FRAME_BYTES * 10)
        elapsed = time.monotonic() - t0
        srv.relay.close("porch")
        assert head == wav_header()
        assert first == b"\x07" * FRAME_BYTES
        assert rest == bytes(FRAME_BYTES * 10)   # silence once the buffer is empty
        assert elapsed >= 0.15                   # paced (11 frames x 20 ms), not dumped
    finally:
        srv.shutdown()


def test_server_rejects_unknown_paths(tmp_path):
    srv = AudioServer(str(tmp_path), port=0, host="127.0.0.1").start()
    try:
        for path in ("/live/nope.wav", "/tts/../../etc/passwd", "/tts/zz.wav"):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=3)
                raise AssertionError(path)
            except urllib.error.HTTPError as e:
                assert e.code == 404
    finally:
        srv.shutdown()


# ------------------------------------------------------------------ operator with a fake Gemini

class FakeSession:
    def __init__(self, script):
        self.script = script            # list of turns; each turn is a list of messages
        self.sent_audio = []
        self.sent_frames = 0
        self.kickoff = None
        self.config = None

    async def send_client_content(self, turns, turn_complete):
        self.kickoff = (turns, turn_complete)

    async def send_realtime_input(self, audio=None, video=None, media=None):
        if audio is not None:
            self.sent_audio.append(audio["data"] if isinstance(audio, dict) else audio.data)
        if video is not None or media is not None:
            self.sent_frames += 1

    async def receive(self):
        if self.script:
            for msg in self.script.pop(0):
                await asyncio.sleep(0.01)
                yield msg
        else:
            await asyncio.sleep(3600)
            yield None


class FakeClient:
    def __init__(self, session):
        s = session

        class _Conn:
            async def __aenter__(self_inner):
                return s

            async def __aexit__(self_inner, *a):
                return False

        class _Live:
            def connect(self_inner, model, config):
                s.config = (model, config)
                return _Conn()

        self.aio = SimpleNamespace(live=_Live())


def msg(data=None, interrupted=False, go_away=None):
    return SimpleNamespace(data=data, go_away=go_away,
                           server_content=SimpleNamespace(interrupted=interrupted, turn_complete=False))


def make_op(tmp_path, session, mic_chunks=50):
    srv = AudioServer(str(tmp_path), port=0, host="127.0.0.1")
    played = []

    async def mic(_url):
        for _ in range(mic_chunks):
            await asyncio.sleep(0.01)
            yield b"\x05" * 3200
        await asyncio.sleep(3600)

    op = GeminiLiveOperator(
        api_key="k", server=srv, go2rtc_url="http://g", public_url="http://lg:8099",
        client_factory=lambda key: FakeClient(session), mic_factory=mic,
        snapshot=lambda cam: b"\xff\xd8jpeg", play=lambda stream, url: played.append((stream, url)),
        image_every_s=0.05, max_session_s=5, streams={"backyard": "backyard_cam"},
    )
    return op, srv, played


def test_operator_session_end_to_end(tmp_path):
    model_audio = b"\x09" * 4800
    session = FakeSession([[msg(data=model_audio)]])
    op, srv, played = make_op(tmp_path, session)
    op.start("backyard", {"camera_area": "in the back yard", "description": "man in a red jacket"})
    time.sleep(0.3)
    assert srv.relay.is_open("backyard")
    op.stop("backyard")
    op.join("backyard", 5)

    assert played == [("backyard_cam", "http://lg:8099/live/backyard.wav")]
    model, cfg = session.config
    assert model == "gemini-3.8-live" and cfg["response_modalities"] == ["AUDIO"]
    assert "red jacket" in cfg["system_instruction"] and "NEVER claim police" in cfg["system_instruction"]
    assert "sliding_window" in cfg["context_window_compression"]
    assert session.kickoff[1] is True
    assert session.sent_frames >= 2
    assert session.sent_audio  # mic forwarded
    assert not srv.relay.is_open("backyard")


def test_operator_mutes_mic_while_model_speaks(tmp_path):
    session = FakeSession([[msg(data=b"\x09" * 48000)]])   # 1 s of model speech
    op, srv, _ = make_op(tmp_path, session)
    op.start("cam", {})
    time.sleep(0.4)
    op.stop("cam")
    op.join("cam", 5)
    assert any(c == bytes(3200) for c in session.sent_audio), "mic should be zeroed during playback"


def test_operator_flushes_on_barge_in(tmp_path):
    session = FakeSession([[msg(data=b"\x09" * 48000), msg(interrupted=True)]])
    op, srv, _ = make_op(tmp_path, session)
    seen = []
    orig_flush = srv.relay.flush
    srv.relay.flush = lambda cam: (seen.append(cam), orig_flush(cam))
    op.start("cam", {})
    time.sleep(0.3)
    op.stop("cam")
    op.join("cam", 5)
    assert seen == ["cam"]


def test_operator_go_away_ends_session(tmp_path):
    session = FakeSession([[msg(go_away=SimpleNamespace(time_left="10s"))]])
    op, srv, _ = make_op(tmp_path, session)
    op.start("cam", {})
    op.join("cam", 5)
    assert not op._threads["cam"].is_alive()


def test_operator_respects_max_session(tmp_path):
    session = FakeSession([])
    op, srv, _ = make_op(tmp_path, session)
    op.max_session_s = 0.3
    t0 = time.monotonic()
    op.start("cam", {})
    op.join("cam", 5)
    assert time.monotonic() - t0 < 3


def test_second_start_is_ignored_while_running(tmp_path):
    session = FakeSession([])
    op, srv, played = make_op(tmp_path, session)
    op.start("cam", {})
    op.start("cam", {})
    time.sleep(0.1)
    op.stop("cam")
    op.join("cam", 5)
    assert len(played) == 1
