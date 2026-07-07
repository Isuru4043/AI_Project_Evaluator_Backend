# exam-station-cv — CV/Behavioral Module (VivaSense)

Local exam-station service: webcam-based behavioral signals, per-student turn
attribution in group vivas, and full session A/V recording for post-hoc
lecturer review. Runs on the exam-room PC; no video leaves the machine unless
the platform later pulls the recording.

This module is built **standalone-first**. The rest of the system is treated
as not-yet-existing and integrates later through the thin contract in
`src/exam_cv/contracts/`.

## Whole-system map (thin — do not plan other modules' internals here)

1. **Platform core** — auth, projects, rubrics, groups, session scheduling (future).
2. **Report evaluator** — semantic analysis of PDF/DOCX reports.
3. **Code analyzer** — SonarCloud static analysis + LLM code questions.
4. **Viva examiner** — RAG question generation, adaptive difficulty (Bloom's), answer scoring.
5. **CV/behavioral module (THIS module)** — behavioral signals, turn attribution, session recording.
6. **BLE physiological module** — heart-rate/HRV from wrist band, baselined per session (sibling producer, not part of this module).
7. **Report generator / XAI layer** — assembles the post-session report; consumes this module's artifact verbatim.
8. **Examiner dashboard (HITL UI)** — where the examiner reviews everything (including the recording) and decides grades.

## The two invariants (enforced in code and docs)

1. **Human-in-the-loop.** The system only recommends; the examiner makes every
   final grading decision. Nothing this module emits is a grade or
   auto-triggers a penalty — integrity flags are timecoded evidence pointers
   for human review of the recording.
2. **Behavioral/physiological signals are advisory and never enter scoring.**
   Gaze, engagement, stress, HR — none of it is an input to any score
   computation or fusion. The *only* output that touches the scoring path is
   **turn attribution** (who spoke), which routes an answer's score to the
   right student without changing its value.

## The three seams this module touches

1. **Session clock/ID** (platform → CV): the platform hands over a *session
   manifest* — `session_id` (UUID), mode (individual/group), roster
   (`student_id` + display name), and a t0 clock anchor. Every event and the
   recording are stamped `(session_id, t_session_ms)`. This module never mints
   its own session identity; standalone/dev mode consumes a locally generated
   manifest of the same shape.
2. **Co-timeline'd audio + BLE streams** (siblings on the shared clock): audio
   is consumed *by* this module (VAD for active-speaker correlation, and muxed
   into the recording); BLE HR is a sibling producer stamped on the same
   session clock. This module depends on the shared timestamp convention for
   BLE — never on BLE data itself.
3. **Session-report consumer** (CV → report generator): at session end this
   module emits one versioned JSON artifact — attribution timeline,
   per-student behavioral summary, integrity flags with `t_session_ms` +
   video timecodes — plus the recording file and a
   `t_session_ms ↔ video offset` mapping so the dashboard can deep-link flag
   moments in the player. Schema is versioned (`schema_version`).

## Performance rules (bake into structure, not comments)

1. **One detector.** The MediaPipe Tasks **FaceLandmarker** is the sole
   detect+track stack (the legacy `mp.solutions.face_mesh` API was removed in
   MediaPipe 0.10.x). ArcFace never runs its own detection — it embeds
   FaceLandmarker-provided aligned crops only. `MeshPipeline` memoizes the
   per-frame result so the two rate paths never run inference twice.
2. **Embeddings never in the frame loop.** ArcFace runs only at enrollment,
   periodic re-verify (~every 10–15 s per track), and track loss/reacquire.
3. **Two-rate loop.** Attribution path at 10–15 FPS; behavioral analyzers on
   a 2–5 FPS tick.
4. **Iris comes free.** The FaceLandmarker model always outputs 478 landmarks
   incl. iris, so gaze needs no extra "refined" pass — the tick path just
   samples the same landmarks less often.
5. **Speaker-grade detail only for speaker candidates.** Non-speaking faces
   get presence + coarse gaze.

## Runtime requirements

- **ffmpeg** must be on PATH (session recording + A/V mux). The recorder
  corrects webcam fps drift at finalize via `-itsscale` so video stays in
  sync with the real-time audio.
- First run downloads two models: `face_landmarker.task` (into `models/`,
  gitignored) and InsightFace `buffalo_l` (live group mode only).
- Smoke test the live pipeline: `python scripts/smoke_live.py <out-dir>`.

## Two entry points

1. **Live** (`exam-cv --students …` / `--manifest …`): owns the webcam/mic,
   records via ffmpeg, analyzes in real time. For in-person stations where
   no browser holds the camera.
2. **Post-hoc** (`python -m exam_cv.analyze --video rec --manifest m.json
   --output-dir out`): analyzes an existing recording — the production path,
   since the browser (Agora) owns the camera during vivas. The Django
   backend's `cv_analysis` app invokes this via subprocess in THIS venv
   (`CV_ANALYSIS_PYTHON` setting) and stores the summary artifact.
   Group identity post-hoc is **seating order** (left→right = roster order);
   count mismatches resolve to unknown, never guessed.

## Layout

```
src/exam_cv/
  contracts/   # schemas (versioned), ArtifactSink protocol + FileSink
  capture/     # camera (single open, frame tee), audio + VAD, ffmpeg recorder
  faces/       # frame-rate mesh + tick-rate refined mesh, identity gallery
  speaker/     # lip-motion × VAD, turn segmentation
  behavior/    # gaze, presence, engagement (tick rate, advisory only)
  events/      # append_event() → JSONL (no DB, no bus)
  report/      # summary artifact w/ video timecodes
  service.py   # session runner / CLI entrypoint
```

Heavy deps (mediapipe, insightface, torch/silero, sounddevice, cv2) are
imported lazily so contracts and analyzers stay unit-testable on a bare
`pip install -e .` (pydantic + numpy only). Runtime extras:
`pip install -e .[station]`.
