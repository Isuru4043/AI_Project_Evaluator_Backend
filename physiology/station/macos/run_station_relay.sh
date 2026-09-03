#!/bin/bash
# =============================================================================
# VivaSense exam-station band relay — macOS launcher.
#
# Holds the Bluetooth link to the heart-rate band and feeds whichever physical
# session is running. Started at login by a launchd agent; see
# install_station_relay.sh. Nobody types anything on the day.
#
# Edit the three settings below for this station, then run
# install_station_relay.sh ONCE.
# =============================================================================

# --- The API root. NOT a session URL: the relay finds the session itself.
#     This MUST point at the same backend the kiosk browser is talking to.
#     Getting it wrong is silent from the band's side: the OLED shows a pulse,
#     the relay stays connected, and nothing ever reaches the system.
#       kiosk on https://www.vivasense.tech -> https://api.vivasense.tech/api
#       Django running on THIS Mac          -> http://127.0.0.1:8000/api
#       Django on another machine           -> http://<that-machine-ip>:8000/api
BACKEND="https://api.vivasense.tech/api"

# --- Must match EXAM_STATION_TOKEN in the backend's environment.
STATION_TOKEN="2208720c-f09f-4fa7-8070-7663ea807d605e3d55e9-7658-4285-81ba-c12a6bacd46c"

# --- Advertised name of the band. Leave as-is unless it was renamed.
BAND_NAME="VivaSense-HR"

# -----------------------------------------------------------------------------
# Resolve the backend repo root: this file lives at
#   <repo>/physiology/station/macos/run_station_relay.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
cd "$REPO" || exit 1

# launchd gives an agent almost no PATH, so the interpreter is addressed by
# absolute path rather than by name.
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$REPO/venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "[relay] no virtualenv found at $REPO/.venv or $REPO/venv"
  echo "[relay] run install_station_relay.sh first"
  exit 1
fi

if [ "$STATION_TOKEN" = "CHANGE_ME" ] || [ -z "$STATION_TOKEN" ]; then
  echo "[relay] STATION_TOKEN is not set. Edit run_station_relay.sh."
  exit 1
fi

# The relay reconnects internally, but if the process itself dies - a Bluetooth
# stack reset, say - bring it straight back. launchd's KeepAlive covers a crash
# of this script; this loop covers a clean exit of the Python process.
while true; do
  echo "[relay] starting $(date '+%Y-%m-%d %H:%M:%S')"
  "$PY" -m physiology.station_sidecar \
      --backend "$BACKEND" \
      --token   "$STATION_TOKEN" \
      --device  "$BAND_NAME"
  echo "[relay] exited; restarting in 10s"
  sleep 10
done
