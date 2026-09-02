#!/bin/bash
# =============================================================================
# Install the band relay as a macOS launchd agent.
#
#   bash install_station_relay.sh
#
# Run ONCE on the exam-station Mac. No sudo: a LaunchAgent belongs to the
# logged-in user, and Bluetooth on macOS is bound to a user session anyway - a
# root/system daemon can scan but reliably fails to CONNECT to a GATT device.
#
# It also runs the relay once in the foreground first. That is deliberate:
# macOS only shows the Bluetooth permission prompt to a process with a
# terminal, and a background agent that was never granted it fails silently
# forever. Grant it when asked, and the same interpreter keeps the grant.
# =============================================================================

set -u

LABEL="tech.vivasense.stationrelay"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUNNER="$HERE/run_station_relay.sh"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/Library/Logs/VivaSense"

echo "repo: $REPO"

# ---------------------------------------------------------------- virtualenv
PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$REPO/venv/bin/python"
fi
if [ ! -x "$PY" ]; then
  echo "==> creating virtualenv at $REPO/.venv"
  python3 -m venv "$REPO/.venv" || { echo "python3 -m venv failed. Install Python 3 first."; exit 1; }
  PY="$REPO/.venv/bin/python"
fi

echo "==> installing relay dependencies"
# Only what the relay needs. The full Django requirements are not required
# just to forward beats, and installing them here would drag in the whole CV
# toolchain on a machine that may only be acting as the station.
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet bleak requests || { echo "pip install failed"; exit 1; }
"$PY" -c "import bleak, requests" || { echo "bleak/requests not importable"; exit 1; }
echo "    ok"

chmod +x "$RUNNER"
mkdir -p "$LOGDIR" "$HOME/Library/LaunchAgents"

# ------------------------------------------------------- Bluetooth permission
cat <<'MSG'

==> Bluetooth permission check

macOS asks for Bluetooth access the first time a process scans, and it only
asks a process that has a terminal. A launchd agent that never got the grant
just reports "waiting for the band to appear" forever.

Starting the relay in the foreground for 25 seconds.
  - If macOS asks to allow Bluetooth, click Allow.
  - Watch for "band connected".
  - It stops on its own.

MSG
read -r -p "Press Return to start the check... " _ || true

( "$RUNNER" & echo $! > /tmp/vivasense_relay_check.pid ) 2>/dev/null
sleep 25
if [ -f /tmp/vivasense_relay_check.pid ]; then
  pkill -P "$(cat /tmp/vivasense_relay_check.pid)" 2>/dev/null
  kill "$(cat /tmp/vivasense_relay_check.pid)" 2>/dev/null
  rm -f /tmp/vivasense_relay_check.pid
fi
pkill -f 'physiology.station_sidecar' 2>/dev/null
echo ""
echo "    foreground check finished."
echo "    If you never saw 'band connected', open"
echo "      System Settings > Privacy & Security > Bluetooth"
echo "    and enable Terminal (and Python, if listed), then re-run this script."
echo ""

# ------------------------------------------------------------------ the agent
echo "==> installing launchd agent"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
rm -f "$PLIST"

cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$RUNNER</string>
  </array>

  <!-- Start at login and keep it up. KeepAlive covers a crash; the runner's
       own loop covers a clean exit of the Python process. -->
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>10</integer>

  <key>WorkingDirectory</key>
  <string>$REPO</string>
  <key>StandardOutPath</key>
  <string>$LOGDIR/relay.log</string>
  <key>StandardErrorPath</key>
  <string>$LOGDIR/relay.log</string>
</dict>
</plist>
PLISTEOF

launchctl bootstrap "gui/$UID" "$PLIST" || {
  echo "launchctl bootstrap failed"; exit 1; }
launchctl kickstart -k "gui/$UID/$LABEL" 2>/dev/null || true

sleep 6
echo ""
if pgrep -f 'physiology.station_sidecar' >/dev/null; then
  echo "  relay: RUNNING"
else
  echo "  relay: not up yet - check the log below"
fi
echo ""
echo "Done. The relay starts at every login."
echo ""
echo "  log      tail -f $LOGDIR/relay.log"
echo "  status   launchctl print gui/$UID/$LABEL | head -20"
echo "  stop     launchctl bootout gui/$UID/$LABEL"
echo "  start    launchctl bootstrap gui/$UID $PLIST"
echo ""
