"""Run the CV engine over one video clip and print what it saw.

For iterating on behaviour and face recognition without creating a session,
a database row, or a viva. Point it at a clip and read the result:

    # behaviour only (gaze, presence, flags)
    python scripts/test_clip.py --video me.mp4

    # behaviour + face recognition against a reference photo
    python scripts/test_clip.py --video me.mp4 --photo me.jpg

    # two people in the clip, one photo each
    python scripts/test_clip.py --video pair.mp4 --photo alice.jpg --photo bob.jpg

WHY --photo CHANGES THE MODE: individual mode deliberately never loads the
recognition model - it assumes the largest face is the one student on the
roster. So a solo clip run without a photo exercises gaze and presence but
does NOT test face recognition. Passing a photo switches the manifest to
GROUP mode, which is the only path that actually embeds faces and matches
them against a gallery.

Needs the engine's own venv (heavy CV deps) and ffmpeg on PATH:
    .venv/Scripts/python.exe scripts/test_clip.py --video me.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Event notes are authored for the examiner UI and contain typographic
# punctuation. A Windows console defaults to cp1252 and raises on it, which
# would kill the run after the analysis has already succeeded - so print
# through UTF-8 and degrade rather than crash.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Run from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exam_cv.analyze import analyze  # noqa: E402
from exam_cv.contracts.manifest import standalone_manifest  # noqa: E402
from exam_cv.contracts.schemas import SessionMode  # noqa: E402
from exam_cv.events.store import read_events  # noqa: E402
from exam_cv.service import RunnerConfig  # noqa: E402


def build_manifest(names: list[str], group: bool, out_dir: Path) -> Path:
    mode = SessionMode.GROUP if group else SessionMode.INDIVIDUAL
    manifest = standalone_manifest(names, mode=mode)
    path = out_dir / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path, manifest


def stage_photos(photos: list[Path], manifest, out_dir: Path) -> Path | None:
    """Copy reference photos to <dir>/<student_id>.jpg, which is the layout
    the engine's gallery loader expects."""
    if not photos:
        return None
    enroll = out_dir / "enroll"
    enroll.mkdir(exist_ok=True)
    for entry, photo in zip(manifest.roster, photos):
        if not photo.exists():
            sys.exit(f"photo not found: {photo}")
        shutil.copy(photo, enroll / f"{entry.student_id}{photo.suffix.lower()}")
    return enroll


def summarize_events(events_path: Path, roster: dict) -> None:
    """Print the raw signal, not just the flags.

    A 20-second clip often will not cross the 6s threshold that earns an
    integrity flag, and "no flags" looks identical to "nothing detected". So
    report the underlying gaze samples and the longest off-screen spell too -
    that distinguishes "working, nothing severe enough" from "not working".
    """
    gaze_samples = defaultdict(lambda: {"on": 0, "off": 0})
    off_spells = defaultdict(list)   # student -> [duration_ms]
    turns = []
    flags = []
    absences = []

    # Reconstruct off-screen spells from the raw sample stream, so we can show
    # the longest one even when it never reached flag length.
    off_since: dict[str, int] = {}
    last_t: dict[str, int] = {}

    for event in read_events(events_path):
        kind = getattr(event, "type", None)
        if kind == "attribution":
            turns.append(event)
        elif kind == "integrity_flag":
            flags.append(event)
        elif kind == "behavioral":
            name = event.kind.value
            sid = event.student_id or "?"
            if name == "gaze_sample":
                on = bool(event.payload.get("on_camera"))
                gaze_samples[sid]["on" if on else "off"] += 1
                if on:
                    if sid in off_since:
                        off_spells[sid].append(event.t_ms - off_since.pop(sid))
                else:
                    off_since.setdefault(sid, event.t_ms)
                last_t[sid] = event.t_ms
            elif name == "absence":
                absences.append(event)
    # Close any spell still open at end of clip.
    for sid, start in off_since.items():
        off_spells[sid].append(last_t.get(sid, start) - start)

    print("\n-- Faces seen ----------------------------------------")
    if not gaze_samples:
        print("  none - no face was detected in any frame.")
    for sid, counts in gaze_samples.items():
        total = counts["on"] + counts["off"]
        pct = (counts["on"] / total * 100) if total else 0.0
        label = roster.get(sid, sid)
        if str(sid).startswith("unknown_track:"):
            label = f"UNRECOGNISED FACE (track {str(sid).split(':', 1)[1]})"
        print(f"  {label}")
        print(f"    gaze samples : {total}  ({pct:.0f}% on-camera)")
        spells = sorted(off_spells.get(sid, []), reverse=True)
        if spells:
            print(f"    look-aways   : {len(spells)}, longest {spells[0] / 1000:.1f}s")
            if spells[0] < RunnerConfig().gaze_flag_ms:
                print(
                    "                   (a flag needs a sustained "
                    f"{RunnerConfig().gaze_flag_ms / 1000:g}s+ look-away)"
                )
        else:
            print("    look-aways   : none")

    print("\n-- Speaking turns ----------------------------------------")
    if not turns:
        print("  none - needs voice activity AND visible lip motion.")
    for t in turns:
        who = roster.get(t.student_id, t.student_id)
        if str(t.student_id).startswith("unknown_track:"):
            who = "UNRECOGNISED FACE"
        print(
            f"  {t.t_start_ms / 1000:5.1f}s - {t.t_end_ms / 1000:5.1f}s  "
            f"{who}  (confidence {t.confidence})"
        )

    print("\n-- Integrity flags ----------------------------------------")
    if not flags:
        print("  none.")
    for f in flags:
        who = roster.get(f.student_id, f.student_id or "session")
        print(f"  [{f.video_timecode}] {f.kind.value} - {who}")
        print(f"      {f.note}")

    if absences:
        print("\n-- Absences ----------------------------------------")
        for a in absences:
            print(
                f"  {roster.get(a.student_id, a.student_id)} gone "
                f"{a.payload.get('duration_ms', 0) / 1000:.1f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one clip and print what the CV engine saw.",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--photo", type=Path, action="append", default=[],
        help="Reference face photo. One per student, in roster order. "
             "Passing any photo switches to GROUP mode so face recognition "
             "actually runs.",
    )
    parser.add_argument(
        "--name", action="append", default=[],
        help="Student display name (repeatable). Defaults to one 'Test Student'.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to keep the artifact. Default: a temp dir that is kept "
             "and printed so you can inspect the JSON.",
    )
    parser.add_argument("--target-fps", type=float, default=12.0)
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")

    names = args.name or ["Test Student"]
    if args.photo and len(args.photo) > len(names):
        # One name per photo, so the roster and the gallery line up.
        names = [f"Student {i + 1}" for i in range(len(args.photo))]

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="cv_test_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    group = bool(args.photo) or len(names) > 1
    manifest_path, manifest = build_manifest(names, group, out_dir)
    enroll_dir = stage_photos(args.photo, manifest, out_dir)

    roster = {e.student_id: e.display_name for e in manifest.roster}

    print(f"clip     : {args.video}")
    print(f"mode     : {manifest.mode.value}", end="")
    if not group:
        print("  (no face recognition - pass --photo to test it)")
    else:
        print(f"  ({len(args.photo)} reference photo(s))")
    print(f"output   : {out_dir}")
    print("\nanalyzing...", flush=True)

    summary = analyze(
        args.video,
        manifest_path,
        out_dir,
        target_fps=args.target_fps,
        enrollment_dir=enroll_dir,
    )

    events_path = out_dir / f"session_{summary.session_id}_events.jsonl"
    summarize_events(events_path, roster)

    print("\n-- Per-student summary ----------------------------------------")
    for s in summary.per_student:
        print(
            f"  {s.display_name}: {s.turn_count} turns, "
            f"{s.speaking_time_ms / 1000:.1f}s speaking, "
            f"attention {s.attention_pct}, "
            f"{s.off_screen_glance_count} look-aways, "
            f"{len(s.integrity_flags)} flags"
        )

    print(f"\nartifact : {out_dir / f'session_{summary.session_id}_summary.json'}")
    print(f"events   : {events_path}")


if __name__ == "__main__":
    main()
