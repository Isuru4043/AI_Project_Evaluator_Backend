"""Exam-station band relay: a service, not a command.

Runs continuously on the exam-room PC. It holds the Bluetooth link to the
heart-rate band and posts beats to whichever physical session is currently
running, asking the platform which one that is.

    python -m physiology.station_sidecar --backend https://host/api --token XXX

WHY IT DISCOVERS THE SESSION INSTEAD OF BEING TOLD
    An earlier version took a session id in its URL, which meant somebody had
    to start it by hand once per viva. In practice that gets done late or not
    at all, and the calm baseline can only be captured while beats are already
    arriving - so a late start silently costs the whole arousal comparison.
    Asking "which session is live?" lets one process start at boot and follow
    sessions as they come and go, with nobody at a keyboard.

WHAT IT SURVIVES
    band not switched on yet     keeps scanning until it appears
    band out of range / reboots  reconnects on its own
    no session running           stays connected, discards beats
    session ends, next begins    retargets without a restart
    backend down                 drops the batch, keeps going
    laptop reboot                Windows Task Scheduler restarts it (see
                                 install_station_service.bat)

Nothing here can fail an exam: every network path is best-effort, and losing
the band costs a supplementary signal, never the viva.

Requires: pip install bleak requests
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger('physio.relay')

HR_MEASUREMENT_UUID = '00002a37-0000-1000-8000-00805f9b34fb'
BATTERY_LEVEL_UUID = '00002a19-0000-1000-8000-00805f9b34fb'

FLUSH_SECONDS = 5.0
SESSION_POLL_SECONDS = 10.0
RECONNECT_DELAY_S = 5.0
SCAN_TIMEOUT_S = 15.0
MAX_BATCH = 200
# Beats collected while no session is running are dropped rather than queued:
# they belong to nobody, and replaying them into the next session would put
# one student's pulse on another's report.
MAX_BUFFER = 600


def parse_hrm(data: bytes) -> dict:
    """Decode a Heart Rate Measurement characteristic value.

    Layout (Bluetooth SIG):
        byte 0      flags
                      bit 0  value is uint16 rather than uint8
                      bit 1  sensor contact detected
                      bit 2  sensor contact supported
                      bit 3  energy expended field present
                      bit 4  RR-interval field present
        byte 1..     heart rate (1 or 2 bytes)
        [2 bytes]    energy expended, if present
        rest         RR-intervals, uint16 each, in 1/1024 s

    RR-intervals are the reason this parser exists: they are the inter-beat
    gaps, and HRV cannot be recovered from the averaged rate alone.
    """
    if not data:
        return {'bpm': None, 'ibi_ms': [], 'contact': True}

    flags = data[0]
    wide = bool(flags & 0x01)
    contact_supported = bool(flags & 0x04)
    contact_detected = bool(flags & 0x02)
    energy_present = bool(flags & 0x08)
    rr_present = bool(flags & 0x10)

    index = 1
    if wide:
        bpm = int.from_bytes(data[index:index + 2], 'little')
        index += 2
    else:
        bpm = data[index]
        index += 1

    if energy_present:
        index += 2

    ibi_ms: list[float] = []
    if rr_present:
        while index + 1 < len(data):
            rr = int.from_bytes(data[index:index + 2], 'little')
            index += 2
            # RR is in 1/1024 s units, not milliseconds.
            ibi_ms.append(round(rr * 1000.0 / 1024.0, 1))

    return {
        'bpm': bpm or None,
        'ibi_ms': ibi_ms,
        # When the band does not implement contact reporting, assume contact:
        # refusing every sample would be worse than trusting the clip.
        'contact': contact_detected if contact_supported else True,
    }


class Relay:
    """Holds the band link and forwards beats to the live session."""

    def __init__(self, api_base: str, token: str, device_name: str,
                 address: str | None):
        self.api_base = api_base.rstrip('/')
        self.token = token
        self.device_name = device_name
        self.address = address

        self.session_id: str | None = None
        self.session_label: str = ''
        self.device_bound = False
        self.battery_pct: int | None = None

        self._pending: list[dict] = []
        self.posted = 0

    # -- platform ---------------------------------------------------------

    @property
    def _headers(self) -> dict:
        return {
            'Content-Type': 'application/json',
            'X-Station-Token': self.token,
        }

    def poll_session(self) -> None:
        """Ask the platform which session, if any, is running."""
        import requests

        try:
            response = requests.get(
                f'{self.api_base}/physio/station/active/',
                # The band's own name is how the platform knows which session
                # claimed it. Without this the server can only guess, and with
                # two vivas running at once it guesses wrong.
                params={'device': self.device_name},
                headers={'X-Station-Token': self.token},
                timeout=10,
            )
        except Exception as exc:
            logger.debug('session poll failed (%s)', exc)
            return

        if response.status_code != 200:
            if response.status_code in (401, 403):
                logger.warning(
                    'station token rejected (%s) - check EXAM_STATION_TOKEN',
                    response.status_code,
                )
            return

        data = (response.json() or {}).get('data') or {}
        new_id = data.get('session_id')
        self.device_bound = bool(data.get('device_bound'))

        if new_id == self.session_id:
            return

        # Switching target: anything still queued belongs to the session that
        # just ended, so it is dropped rather than misfiled into the new one.
        self._pending.clear()
        self.session_id = new_id
        if new_id:
            self.session_label = data.get('project') or new_id
            who = data.get('student_name')
            logger.info(
                'now feeding "%s" [%s]%s',
                self.session_label, data.get('phase', '?'),
                f' - band on {who}' if who else ' - no band assigned yet',
            )
        else:
            self.session_label = ''
            reason = data.get('reason') or 'no session running'
            logger.info('%s; holding the band link', reason)

    def add(self, sample: dict) -> None:
        sample['t'] = datetime.now(timezone.utc).isoformat()
        if self.battery_pct is not None:
            sample['battery_pct'] = self.battery_pct
        self._pending.append(sample)
        if len(self._pending) > MAX_BUFFER:
            del self._pending[:-MAX_BUFFER]
        if len(self._pending) >= MAX_BATCH:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        if not self.session_id:
            # Nobody to attribute these to. Discard rather than accumulate.
            self._pending.clear()
            return

        import requests

        batch, self._pending = self._pending, []
        try:
            response = requests.post(
                f'{self.api_base}/sessions/{self.session_id}/physio/samples/',
                json={'device_id': self.device_name, 'samples': batch},
                headers=self._headers,
                timeout=15,
            )
            if response.status_code == 409:
                # No wearer chosen yet. Expected early in the demo phase, and
                # it resolves itself once the kiosk panel binds the band.
                logger.debug('samples refused: band not assigned yet')
            elif response.status_code >= 400:
                logger.warning('backend rejected batch (%s): %s',
                               response.status_code, response.text[:160])
            else:
                self.posted += len(batch)
                logger.info('posted %d beat sample(s) to %s',
                            len(batch), self.session_label)
        except Exception as exc:
            # A dropped batch costs a few seconds of the arousal curve.
            # Halting the relay over it would cost the whole session.
            logger.warning('batch dropped (%s)', exc)

    # -- band -------------------------------------------------------------

    async def _find_device(self):
        from bleak import BleakScanner

        if self.address:
            return self.address
        found = await BleakScanner.find_device_by_filter(
            lambda d, adv: bool(
                d.name and self.device_name.lower() in d.name.lower()
            ),
            timeout=SCAN_TIMEOUT_S,
        )
        return found.address if found else None

    async def _session_loop(self):
        """Poll the platform on its own cadence, independent of the band."""
        while True:
            await asyncio.to_thread(self.poll_session)
            await asyncio.sleep(SESSION_POLL_SECONDS)

    async def _band_loop(self):
        """Stay connected to the band, forever, through anything."""
        from bleak import BleakClient

        announced_missing = False
        while True:
            try:
                address = await self._find_device()
                if address is None:
                    if not announced_missing:
                        logger.info('waiting for the band to appear...')
                        announced_missing = True
                    await asyncio.sleep(RECONNECT_DELAY_S)
                    continue

                announced_missing = False
                async with BleakClient(address) as client:
                    logger.info('band connected (%s)', address)
                    try:
                        raw = await client.read_gatt_char(BATTERY_LEVEL_UUID)
                        self.battery_pct = int(raw[0])
                    except Exception:
                        pass  # Battery Service is optional

                    def on_notify(_handle, data: bytearray):
                        self.add(parse_hrm(bytes(data)))

                    await client.start_notify(HR_MEASUREMENT_UUID, on_notify)
                    while client.is_connected:
                        await asyncio.sleep(FLUSH_SECONDS)
                        await asyncio.to_thread(self.flush)

                logger.info('band disconnected; reconnecting...')
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning('band link error (%s); retrying', exc)
            await asyncio.sleep(RECONNECT_DELAY_S)

    async def run(self):
        await asyncio.gather(self._session_loop(), self._band_loop())


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Relay the exam-station heart-rate band to the platform.',
    )
    parser.add_argument(
        '--backend', required=True,
        help='API root, e.g. https://api.vivasense.tech/api  '
             '(NOT a session URL - the session is discovered).',
    )
    parser.add_argument('--token', default='', help='X-Station-Token value.')
    parser.add_argument(
        '--device', default='VivaSense-HR',
        help='Advertised name to scan for (default: VivaSense-HR).',
    )
    parser.add_argument(
        '--address', default=None,
        help='Connect straight to this MAC and skip scanning.',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    if not args.token:
        logger.warning(
            'no --token given; the backend will reject every post'
        )

    relay = Relay(args.backend, args.token, args.device, args.address)
    logger.info('station relay starting; band "%s"', args.device)
    try:
        asyncio.run(relay.run())
    except KeyboardInterrupt:
        relay.flush()
        logger.info('stopped after posting %d sample(s)', relay.posted)
        sys.exit(0)


if __name__ == '__main__':
    main()
