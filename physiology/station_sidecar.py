"""Exam-station BLE sidecar: band -> platform.

Runs on the exam-room PC beside the CV sidecar, connects to the wristband over
the standard Bluetooth Heart Rate Service, and relays beats to the backend.

    python -m physiology.station_sidecar \
        --backend https://host/api/sessions/<session_id>/physio \
        --token   $EXAM_STATION_TOKEN \
        --device  VivaSense-HR

WHY A SIDECAR AND NOT THE BROWSER
    Web Bluetooth is Chromium-only, needs a user gesture per device, and drops
    the link when the tab is backgrounded - during a 20-minute viva that is a
    guaranteed data gap. A small process holds the connection instead, and
    reuses the same X-Station-Token the CV sidecar already uses.

STANDARD SERVICE, NOT A CUSTOM ONE
    Heart Rate Service 0x180D / Heart Rate Measurement 0x2A37 already carries
    everything needed, including RR-intervals (the beat-to-beat gaps HRV is
    computed from) and a sensor-contact bit. Using it means nRF Connect and
    any HR app can talk to the band, so the firmware can be debugged without
    this script existing.

Requires: pip install bleak requests
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

logger = logging.getLogger('physio.sidecar')

HR_MEASUREMENT_UUID = '00002a37-0000-1000-8000-00805f9b34fb'
BATTERY_LEVEL_UUID = '00002a19-0000-1000-8000-00805f9b34fb'

# The band notifies about once a second. Posting each one would be a request
# per second per station for no benefit; batching keeps it to one every few.
FLUSH_SECONDS = 5.0
MAX_BATCH = 200


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


class SampleBuffer:
    """Accumulates notifications and posts them in batches."""

    def __init__(self, backend: str, token: str, device_id: str):
        self.backend = backend.rstrip('/')
        self.token = token
        self.device_id = device_id
        self._pending: list[dict] = []
        self.battery_pct = None
        self.posted = 0
        self.failed = 0

    def add(self, sample: dict) -> None:
        sample['t'] = datetime.now(timezone.utc).isoformat()
        if self.battery_pct is not None:
            sample['battery_pct'] = self.battery_pct
        self._pending.append(sample)
        if len(self._pending) >= MAX_BATCH:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        import requests

        batch, self._pending = self._pending, []
        try:
            response = requests.post(
                f'{self.backend}/samples/',
                json={'device_id': self.device_id, 'samples': batch},
                headers={
                    'Content-Type': 'application/json',
                    'X-Station-Token': self.token,
                },
                timeout=15,
            )
            if response.status_code >= 400:
                self.failed += len(batch)
                logger.warning('backend rejected batch (%s): %s',
                               response.status_code, response.text[:200])
            else:
                self.posted += len(batch)
                logger.info('posted %d sample(s)', len(batch))
        except Exception as exc:
            # A dropped batch costs a few seconds of the arousal curve. Halting
            # the viva over it would be a far worse trade, so it is logged and
            # abandoned rather than retried into a growing backlog.
            self.failed += len(batch)
            logger.warning('batch dropped (%s)', exc)


async def run(device_name: str, buffer: SampleBuffer, address: str | None):
    from bleak import BleakClient, BleakScanner

    if address:
        target = address
        logger.info('connecting to %s', address)
    else:
        logger.info('scanning for a band advertising "%s"...', device_name)
        found = await BleakScanner.find_device_by_filter(
            lambda d, adv: bool(d.name and device_name.lower() in d.name.lower()),
            timeout=20.0,
        )
        if found is None:
            logger.error(
                'no band found. Check it is powered on and advertising, or '
                'pass --address with its MAC.'
            )
            return 1
        target = found.address
        logger.info('found %s at %s', found.name, target)

    async with BleakClient(target) as client:
        logger.info('connected; subscribing to heart rate notifications')

        try:
            raw = await client.read_gatt_char(BATTERY_LEVEL_UUID)
            buffer.battery_pct = int(raw[0])
            logger.info('battery %d%%', buffer.battery_pct)
        except Exception:
            pass  # Battery Service is optional

        def on_notify(_handle, data: bytearray):
            buffer.add(parse_hrm(bytes(data)))

        await client.start_notify(HR_MEASUREMENT_UUID, on_notify)

        try:
            while client.is_connected:
                await asyncio.sleep(FLUSH_SECONDS)
                buffer.flush()
        except asyncio.CancelledError:
            pass
        finally:
            buffer.flush()
            try:
                await client.stop_notify(HR_MEASUREMENT_UUID)
            except Exception:
                pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Relay wristband heart-rate data to the platform.',
    )
    parser.add_argument(
        '--backend', required=True,
        help='Session physio endpoint, e.g. '
             'https://host/api/sessions/<session_id>/physio',
    )
    parser.add_argument('--token', default='', help='X-Station-Token value.')
    parser.add_argument(
        '--device', default='VivaSense-HR',
        help='Advertised name to scan for (default: VivaSense-HR).',
    )
    parser.add_argument(
        '--address', default=None,
        help='Connect straight to this MAC/UUID and skip scanning.',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    if not args.token:
        logger.warning(
            'no --token given; the backend will reject these posts unless '
            'EXAM_STATION_TOKEN is unset there too'
        )

    buffer = SampleBuffer(args.backend, args.token, args.device)
    try:
        code = asyncio.run(run(args.device, buffer, args.address))
    except KeyboardInterrupt:
        buffer.flush()
        code = 0
    logger.info('posted %d, failed %d', buffer.posted, buffer.failed)
    sys.exit(code)


if __name__ == '__main__':
    main()
