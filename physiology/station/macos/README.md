# Exam-station band relay — macOS setup

The relay holds the Bluetooth link to the heart-rate band and forwards beats to
whichever physical session is running. It has to run on **the machine the band
is near**, because Bluetooth is a local radio — nothing about it travels with
the browser.

## Which machine runs what

The band talks to the Mac. The Mac talks to Django. Those are separate links,
and Django does **not** have to be on the Mac.

| Setup | `BACKEND` in `run_station_relay.sh` |
|---|---|
| Django on this Mac | `http://127.0.0.1:8000/api` |
| Django on another laptop | `http://<that-laptop-ip>:8000/api` |

If Django is elsewhere, start it so it accepts connections from the network —
`runserver 0.0.0.0:8000`, not the default `127.0.0.1` — and check the firewall
allows port 8000. `ALLOWED_HOSTS` is already `['*']`.

Both machines must be on the same network, and the kiosk browser must reach the
same Django the relay does. If the browser is on the Mac and Django is on the
Windows laptop, the frontend's API base has to point there too.

## Install (once)

```bash
cd <repo>/physiology/station/macos
bash install_station_relay.sh
```

No `sudo`. A LaunchAgent belongs to the logged-in user, and macOS binds
Bluetooth to a user session — a root daemon can scan but reliably fails to
*connect* to a GATT device.

The script creates a virtualenv, installs `bleak` and `requests`, runs the
relay once in the foreground, then installs the launchd agent.

### The foreground step is not optional

macOS asks for Bluetooth permission the first time a process scans, and it only
asks a process that has a terminal. A background agent that never received the
grant reports `waiting for the band to appear` forever, with no error and no
prompt — the single most likely way to lose a demo.

When the prompt appears, click **Allow**. If it never appears, open
**System Settings → Privacy & Security → Bluetooth** and enable **Terminal**
(and **Python**, if listed), then re-run the script.

## Check it

```bash
tail -f ~/Library/Logs/VivaSense/relay.log
```

Healthy output:

```
[relay] starting 2026-09-02 09:14:02
INFO station relay starting; band "VivaSense-HR"
INFO band connected (88:13:BF:62:43:FE)
INFO now feeding "Pavith Individual Session 03" [demo_in_progress] - band on Pavith
INFO posted 8 beat sample(s) to Pavith Individual Session 03
```

## If it will not connect

| Log line | Cause |
|---|---|
| `waiting for the band to appear` | Band unpowered, out of range, **or Bluetooth permission was never granted** — or something else already holds the connection (only one at a time; close nRF Connect and any other relay) |
| `station token rejected (403)` | `STATION_TOKEN` here does not match `EXAM_STATION_TOKEN` in Django's `.env` |
| `no session has claimed this band` | Normal before the kiosk panel binds the wearer. Resolves itself once the demo phase opens |
| `samples refused: band not assigned yet` | Group session — pick the wearer in the kiosk panel |
| Nothing at all | Agent not running: `launchctl print gui/$UID/tech.vivasense.stationrelay` |

**Do not pair the band in System Settings → Bluetooth.** BLE GATT devices are
not paired like headphones; the relay connects directly, and a macOS pairing can
interfere.

## Control

```bash
launchctl bootout   gui/$UID/tech.vivasense.stationrelay   # stop
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/tech.vivasense.stationrelay.plist   # start
launchctl kickstart -k gui/$UID/tech.vivasense.stationrelay   # restart
```
