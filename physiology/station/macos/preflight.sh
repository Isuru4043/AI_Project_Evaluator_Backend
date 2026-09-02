#!/bin/bash
# =============================================================================
# Demo-day preflight for the exam-station Mac.
#
#   bash preflight.sh
#
# Checks every link in the chain and names the exact fix for whichever one is
# broken. Run it the day before, and again ten minutes before the demo.
#
# The chain:  band --BLE--> this Mac --HTTP--> Django --> kiosk browser
# A break anywhere shows up in the panel as "no data", so this reports each
# link separately instead of leaving you to guess which one failed.
# =============================================================================

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUNNER="$HERE/run_station_relay.sh"
LABEL="tech.vivasense.stationrelay"

pass=0; fail=0
ok()   { echo "  [ OK ] $1"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $1"; echo "         -> $2"; fail=$((fail+1)); }
note() { echo "         $1"; }

echo ""
echo "=============================================="
echo " VivaSense station preflight"
echo "=============================================="
echo " repo: $REPO"
echo ""

# ---------------------------------------------------------------- 1. python
echo "1. Python + relay dependencies"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="$REPO/venv/bin/python"
if [ ! -x "$PY" ]; then
  bad "no virtualenv" "run: bash install_station_relay.sh"
else
  ok "virtualenv at $PY"
  if "$PY" -c "import bleak, requests" 2>/dev/null; then
    ok "bleak + requests importable"
  else
    bad "bleak/requests missing" "run: $PY -m pip install bleak requests"
  fi
fi
echo ""

# --------------------------------------------------------------- 2. settings
echo "2. Station configuration"
BACKEND=$(grep -E '^BACKEND=' "$RUNNER" | head -1 | cut -d'"' -f2)
TOKEN=$(grep -E '^STATION_TOKEN=' "$RUNNER" | head -1 | cut -d'"' -f2)
BAND=$(grep -E '^BAND_NAME=' "$RUNNER" | head -1 | cut -d'"' -f2)
[ -n "$BACKEND" ] && ok "backend: $BACKEND" || bad "BACKEND not set" "edit run_station_relay.sh"
if [ -z "$TOKEN" ] || [ "$TOKEN" = "CHANGE_ME" ]; then
  bad "STATION_TOKEN not set" "edit run_station_relay.sh"
else
  ok "token present (${TOKEN:0:8}...)"
fi
ok "band name: $BAND"
if echo "$BACKEND" | grep -q '127.0.0.1\|localhost'; then
  note "BACKEND points at THIS Mac. If Django runs on another"
  note "machine, change it to that machine's IP."
fi
echo ""

# ---------------------------------------------------------------- 3. backend
echo "3. Backend reachable"
CODE=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$BACKEND/physio/station/active/" 2>/dev/null)
case "$CODE" in
  000) bad "cannot reach $BACKEND" "is Django running? if it is on another machine it must use runserver 0.0.0.0:8000, and the firewall must allow 8000" ;;
  401|403) ok "reachable (auth required, as expected)" ;;
  200) ok "reachable" ;;
  *) bad "unexpected HTTP $CODE" "check the URL path ends in /api" ;;
esac

if [ "$CODE" != "000" ] && [ -n "$TOKEN" ]; then
  AUTH=$(curl -s -m 8 -H "X-Station-Token: $TOKEN" "$BACKEND/physio/station/active/?device=$BAND" 2>/dev/null)
  ACODE=$(curl -s -o /dev/null -m 8 -w '%{http_code}' -H "X-Station-Token: $TOKEN" "$BACKEND/physio/station/active/?device=$BAND" 2>/dev/null)
  if [ "$ACODE" = "200" ]; then
    ok "station token accepted"
    if echo "$AUTH" | grep -q '"session_id": *null'; then
      note "no physical session running right now (normal before the demo)"
    else
      note "a session is live and visible to the relay"
    fi
  else
    bad "token rejected (HTTP $ACODE)" "STATION_TOKEN here must equal EXAM_STATION_TOKEN in Django's .env"
  fi
fi
echo ""

# -------------------------------------------------------------- 4. bluetooth
echo "4. Bluetooth + band"
if [ -x "$PY" ] && "$PY" -c "import bleak" 2>/dev/null; then
  RESULT=$("$PY" - "$BAND" <<'PYEOF' 2>&1
import asyncio, sys
from bleak import BleakScanner
name = sys.argv[1].lower()
async def main():
    try:
        devs = await BleakScanner.discover(timeout=10.0)
    except Exception as exc:
        print("SCANFAIL", exc); return
    hit = [d for d in devs if d.name and name in d.name.lower()]
    if hit:
        print("FOUND", hit[0].address, hit[0].name)
    else:
        print("NOTFOUND", len(devs))
asyncio.run(main())
PYEOF
)
  case "$RESULT" in
    FOUND*) ok "band advertising: $(echo "$RESULT" | cut -d' ' -f2-)" ;;
    SCANFAIL*)
      bad "Bluetooth scan failed" "System Settings > Privacy & Security > Bluetooth: enable Terminal (and Python)"
      note "$RESULT" ;;
    NOTFOUND*)
      SEEN=$(echo "$RESULT" | cut -d' ' -f2)
      if [ "$SEEN" = "0" ]; then
        bad "no BLE devices seen at all" "Bluetooth off, or permission not granted (System Settings > Privacy & Security > Bluetooth)"
      else
        bad "band not advertising (saw $SEEN other devices)" "band unpowered/out of range, OR already connected to something - only one connection at a time, so stop any other relay or nRF Connect"
      fi ;;
  esac
else
  note "skipped - bleak not installed"
fi
echo ""

# ------------------------------------------------------------------ 5. agent
echo "5. Relay agent"
if launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1; then
  ok "launchd agent installed"
  if pgrep -f 'physiology.station_sidecar' >/dev/null; then
    ok "relay process running"
  else
    bad "agent installed but relay not running" "launchctl kickstart -k gui/$UID/$LABEL ; then tail ~/Library/Logs/VivaSense/relay.log"
  fi
else
  bad "launchd agent not installed" "run: bash install_station_relay.sh"
fi
echo ""

echo "=============================================="
echo " $pass passed, $fail failed"
if [ "$fail" -eq 0 ]; then
  echo " Ready. Power the band, start a session, clip"
  echo " the finger on during the DEMO phase."
else
  echo " Fix the items above and run this again."
fi
echo "=============================================="
echo ""
exit 0
