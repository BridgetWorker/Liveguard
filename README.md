# Liveguard

An AI guard for [Frigate](https://frigate.video). It works out who belongs at your house, keeps quiet for family and deliveries, and thinks before it speaks. When someone really shouldn't be there, it talks to them through the camera speaker like a live monitoring operator.

> **Status: early.** The decision engine and the Gemini Live operator are covered by tests, using a fake Gemini session for the operator. The Frigate, Home Assistant, Piper, go2rtc and Gemini connections haven't been run against real hardware or the live API yet.

## How it decides

```
Frigate (MQTT) ─┐
                ├─► trust ─► intent ─► ceiling ─► ladder ─► actions
Home Assistant ─┘   who?     why?      allowed?   one rung   notify / lights /
(presence)                                        at a time  speak / siren / page you
```

**1. Trust: who is this?** Several independent signals, combined:

| Signal | Source |
|---|---|
| Face match | Frigate face recognition (`sub_label`) |
| Phone just joined Wi-Fi | HA device tracker (router, UniFi, companion app) |
| Phone/watch at the door | BLE (ESPresense, Bermuda) |
| Someone just came out of the house | Alarm disarmed, interior door opened |

A face match alone at night is capped. A bad match on a dark porch is the classic way to end up trusting a stranger, so at night the face needs a second signal (phone, BLE) that agrees.

**2. Intent: why are they here?** Judged from behavior over the whole track, not one frame:

- **Delivery:** reaches the door with a package or uniform and leaves quickly. *Silent* (optional notification).
- **Passerby:** never steps onto the property. *Silent.*
- **Visitor:** rings the bell. *Notify.*
- **Loitering:** stays past a time limit, sits still, or keeps coming back.
- **Suspicious:** goes into the backyard or side yard at night, tries car doors, peers into windows.

**3. Ceiling: what are we allowed to do?**

- **Property line.** Anyone outside your property zones gets *notify at most*: no lights, no voice. If someone walks back onto the sidewalk mid-talk-down, it stops talking.
- **Think before speaking.** The first time a track would get a voice, Liveguard asks a deliberator first. That's a local vision model (Ollama) given the snapshot and all the evidence, or a rules fallback. It answers *speak / don't speak* with a reason you can read, and can write the opening line. Lights can come on while it thinks. If it says no, that track stays at notify unless the behavior gets worse (loitering → suspicious), which triggers a new decision. If no answer comes back in time, it **holds** by default instead of speaking.

**4. Ladder.** Silent → notify → lights → spoken warning → live talk-down → siren → page you. It climbs one rung at a time so a misread can never jump straight to a siren. The siren needs the alarm to be armed. If the talk-down runs for 60 seconds and they're still there, you get a critical alert.

**5. Learning.** Every notification has *This was fine* and *Sketchy* buttons. When a camera has a strong history of "fine" at a given hour (the neighbor's evening walk, school pickup), Liveguard stops notifying about plain unknown people there. Learning only ever removes nagging. It never quiets loitering or suspicious behavior.

Every decision and its reasons are published to `liveguard/decision` on MQTT, so you can always see why it spoke or stayed quiet.

## Live talk-down (Gemini Live)

At the operator rung, Liveguard opens a Gemini Live session for that camera:

```
camera mic ─(go2rtc RTSP → ffmpeg)─► 16 kHz PCM ──► Gemini Live
Frigate latest.jpg every 3 s ─────────────────────►     │
camera speaker ◄─ go2rtc ◄─ /live/<cam>.wav ◄─ 24 kHz PCM
```

- **Opened only when needed.** Gemini bills listening time, so there's no always-on session. A typical 1–2 minute talk-down costs a few cents.
- **Half-duplex.** While the model's voice is playing, the camera mic is muted toward Gemini, because outdoor cameras hear their own speaker.
- **Barge-in.** If the person talks over it, Gemini signals an interruption and the unplayed audio is dropped.
- **Ends when:** they leave, they step off the property, it pages you, or `max_session_s` (default 3 min) runs out.
- **Use the paid tier.** On the free tier, Google may use the audio and images to improve its products.

Set `operator.backend: speaker` to skip Gemini entirely. It then repeats one-way lines, fully offline.

## Web UI

Open `http://<host>:8080` to use it. It has three tabs:

- **Live**: who's on camera, how far up the ladder each person is, and why. Below that, a logbook of everything it noticed, decided and said, with *This was fine* / *Sketchy* buttons on notifications.
- **Test bench**: only works in test mode (see below).
- **Settings**: every option as a form, or the raw YAML. Saving checks that Liveguard can actually start with the new config before writing it, and keeps the previous version as `.bak`. Restart from the page to apply. Secrets are never sent back to the browser.

Set `webui.password` before exposing the UI on your network. Your browser will ask for it, and any username works.

## Test mode

```sh
pip install -e .
liveguard --test -c ./liveguard.yaml     # writes a demo config if the file doesn't exist
# open http://localhost:8080
```

No cameras, MQTT or Home Assistant needed. Your real zones, rules and deliberator run against scripted scenes:

| Scene | What should happen |
|---|---|
| Prowler in the back yard | Thinks, decides to speak, warns, then starts a talk-down |
| Package delivery | Stays quiet (at most a notification) |
| Family coming home late | Silent: face match plus phone on Wi-Fi |
| Someone waiting on the porch | Thinks, decides not to confront in daylight |
| Watching from the sidewalk | Notification only, never speaks (off the property) |

You can also move a person yourself (camera, zones, description, face match), and change the conditions: day or night, alarm armed, a phone joining Wi-Fi, the doorbell.

In test mode, Home Assistant calls only show up in the logbook; nothing is sent. Spoken warnings play on your computer's speakers (Piper if configured, otherwise `say` or `espeak-ng`). With `operator.backend: gemini`, the live talk-down uses your mic and speakers in place of the camera's, so you can argue with it yourself. Use headphones: the mic is muted while it talks.

## Setup

1. **Frigate**: enable face recognition and GenAI descriptions (for "you in the gray hoodie"). Draw zones:
   - `property`: one big zone along your lot line (**recommended**). Nobody outside it is ever spoken to.
   - `street`, `sidewalk`, `neighbor_yard`: off-property areas.
   - `porch`, `backyard`, `side_yard`, `gate`, `driveway_cars`: detail zones.
2. **Home Assistant**: create a long-lived token. Add [`docs/home-assistant-feedback.yaml`](docs/home-assistant-feedback.yaml) as an automation so the notification buttons reach Liveguard.
3. **Voice**: download a [Piper](https://github.com/rhasspy/piper) voice into `./models`. Your camera needs two-way audio through go2rtc. Set `codec` to match it (`pcma` for most Dahua/Amcrest/Reolink). For live talk-down, add a paid-tier `GEMINI_API_KEY` and point `operator.rtsp_url` at go2rtc's RTSP restream.
4. **Deliberator (optional but recommended)**: run Ollama with a vision model, e.g. `ollama pull qwen2.5vl:7b`.
5. Copy `config.example.yaml` to `config/liveguard.yaml` (or edit it later in the web UI), then:

```sh
cp docker-compose.example.yaml docker-compose.yaml
HA_TOKEN=... MQTT_PASSWORD=... GEMINI_API_KEY=... LIVEGUARD_PASSWORD=... docker compose up -d
```

While you're testing, set `policy.max_level: notify` (or `warn`) so it can't go further than that.

## Development

```sh
pip install -e ".[dev]"
pytest
```

The core (`trust`, `intent`, `policy`, `engine`, `deliberate`) is pure Python with no I/O, and each module is tested on its own. `tests/test_core.py` walks through full scenarios: family arriving at night, a delivery, a prowler (think → speak → talk-down → stop), a neighbor's kid the deliberator chooses to leave alone, and someone staring from the sidewalk who is never spoken to.

## Ground rules baked in

- It never claims police are on the way unless that's true, never threatens, and never says whether anyone is home.
- If asked directly whether it's a person, it doesn't claim to be human. It says it's the home's security monitoring and that the homeowner is being notified.
- When it's unsure, it notifies you instead of confronting anyone.
- Recording two-way audio is regulated differently from state to state. Check yours.

## Roadmap

- [x] Gemini Live operator
- [ ] OpenAI Realtime / local speech-to-speech operator backends
- [ ] Add confirmed visitors to Frigate's face library from the notification
- [ ] Vehicle / plate trust (Frigate LPR)
- [x] Web UI for settings, a live view and a test bench
- [ ] HA dashboard card for the decision log
