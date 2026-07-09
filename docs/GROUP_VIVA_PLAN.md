# Group Viva Support — Plan, Status & HPC Runbook

> Portable copy of the implementation plan so any machine (esp. the department
> HPC) can pick up the work after `git clone`. Code for all 5 phases is
> **committed** (backend `main` @ 824388e, frontend `feature/group-viva-platform`
> @ b628ec0). What remains is **operational setup + the two things that could not
> be tested without live Agora keys / the HPC itself.**

## Status at a glance

| Phase | What | Code | Verified |
|-------|------|------|----------|
| A | Group auto-enroll (teammate emails) + shared PPT upload | ✅ done | ✅ serializers smoke-tested |
| B | Student demo-phase gate in LiveVivaRoom | ✅ done | ✅ build passes |
| C | Agora Cloud Recording → Azure blob | ✅ done | ❌ **never run end-to-end (needs keys)** |
| D | Group Q&A sync (`GET /viva/sessions/<id>/current/`) | ✅ done | ✅ endpoint returns 200 |
| E | HPC analysis worker (`process_cv_reports`) | ✅ done | ✅ command runs |

Migrations added: `core` 0013 (`presentation_file_url`) + Agora recording fields
on `EvaluationSession`. `python manage.py check` clean; engine tests 60/60;
frontend `next build` passes.

---

## What is LEFT to do (do these on/from the HPC)

### 1. Stand up the HPC as the CV analysis worker  ← main reason for the trip
The App Service backend has `CV_ANALYSIS_ENABLED=false`, so recordings become
`CVSessionReport` rows in **PENDING**. The HPC claims and processes them.

```bash
# On the HPC (shares the same NeonDB + Azure blob as everything else):
git clone https://github.com/Isuru4043/AI_Project_Evaluator_Backend.git
cd AI_Project_Evaluator_Backend

# backend venv
python -m venv .venv && source .venv/bin/activate    # or Windows equivalent
pip install -r requirements.txt

# engine venv (the CV toolchain lives in the exam-station-cv/ subdir)
cd exam-station-cv
python -m venv .venv-cv && source .venv-cv/bin/activate
pip install -r requirements.txt        # heavy CV deps; no GPU required
cd ..

# .env — copy from a trusted machine, then set:
#   CV_ANALYSIS_ENABLED=true
#   CV_ANALYSIS_PYTHON=<abs path to exam-station-cv/.venv-cv/bin/python>
#   (keep the same NeonDB + Azure creds as the App Service backend)

# run the worker (loops, atomic claim so multiple workers are safe):
python manage.py process_cv_reports          # add --once to drain and exit
```
~5–10 min per 20-min viva on CPU. The same command runs on your laptop for demos.

### 2. Enable Agora Cloud Recording (Phase C — the untested path)
Needs YOU to provide/enable in Agora Console, then set in `.env`:
- `AGORA_CUSTOMER_KEY`, `AGORA_CUSTOMER_SECRET` (RESTful API creds)
- Enable **Cloud Recording** on the Agora project (paid/metered add-on)
- `AGORA_CLOUD_RECORDING_ENABLED=true`
- `AGORA_RECORDING_AZURE_REGION=<Agora's numeric code for your Azure region>`
  (currently `0` — must be corrected; this is Agora's own region enum, not the
  Azure region name)

Then rehearse a short session and confirm: acquire→start→stop succeeds, an mp4
lands in Azure `videos/cloudrec/<session_id>/`, a `SessionRecording` row is
created, `enqueue_cv_analysis` fires, and (with the worker from #1 running) the
behavioral report renders. **This REST flow + the region code are the only parts
not yet exercised — expect to debug them live.**

### 3. Restart the App Service backend
To pick up the new code + 2 migrations.

### 4. Two-browser group rehearsal
Examiner starts demo → student live page shows demo banner + can screen-share →
examiner clicks Complete Demo → student Q&A starts → teammate's second browser
shows the question advancing after the first member answers.

---

## Full phase plan (reference)

### Phase A — Group enroll + PPT
- Backend `projects/views/project_views.py` `EnrollInProjectView`: accepts
  `member_emails: list[str]` with `group_number`; resolves each email →
  `StudentProfile`, rejects unknown / already-grouped members, creates
  `StudentGroup` + `GroupMember` in one transaction. Single-member path preserved.
- `core/models.py` `ProjectSubmission.presentation_file_url` (+ migration 0013).
  `SubmitProjectView` accepts optional `.ppt/.pptx` ≤50 MB via `upload_report_to_blob`.
- Frontend: dynamic teammate-email rows in `ExploreProjectsView.tsx` enroll modal;
  PPT input in `SubmissionForm.tsx` + display in `submissionPanel.tsx`.

### Phase B — Demo phase gate
- `LiveVivaRoom.tsx`: on mount fetch session status. `in_progress` + demo not
  complete → full-screen Agora room + "Demo in progress" banner, no
  `startSession`; poll every 5s. When `demo_completed_at` appears → toast → normal
  Q&A (loadFirstQuestion). Examiner side unchanged.

### Phase C — Agora Cloud Recording
- `agora_service/cloud_recording.py` mirrors `stt_manager.py` REST style:
  `acquire()`/`start()` (composite, Azure storageConfig, prefix
  `cloudrec/<session_id>/`)/`stop()`. Sources `AZURE_*` from `azure_storage`, not
  settings.
- `EvaluationSession.agora_recording_resource_id` + `agora_recording_sid` (+ migration).
- Hooks in `sessions_app/views.py`: `StartDemoView` starts recording
  (feature-flagged, non-blocking thread); `EndVivaView` stops → creates
  `SessionRecording(video_file_url=<blob url>)` → `enqueue_cv_analysis`.
- Browser-upload path stays as default/fallback.

### Phase D — Group Q&A sync
- `GET /api/viva/sessions/<id>/current/` in
  `viva_evaluator/views/session_views.py` returns latest question (+ answered
  flag), reusing the resume-serialization helper extracted from `SessionStartView`.
  No question generation on this path.
- `LiveVivaRoom.tsx`: group sessions poll `current/` every 4s; on change, swap
  question + toast "Your teammate answered". Last-write-wins; per-student
  attribution preserved by the friend's fix.

### Phase E — HPC worker
- `cv_analysis/services/runner.py` `enqueue_cv_analysis`: when
  `CV_ANALYSIS_ENABLED` off, still create/keep a PENDING `CVSessionReport`.
- `cv_analysis/management/commands/process_cv_reports.py`: loop (or `--once`),
  atomic `update(status=PROCESSING)` claim, call `run_cv_analysis(session_id)`.

## Group CV analysis
`cv_analysis/services/manifest.py` emits `mode: group` + roster; `exam_cv.analyze`
does single-camera seating-order attribution (left→right = roster order). Works
for physical-room recordings and approximately for Agora composite grids (tile
order). Per-stream remote analysis = future enhancement.

## Explicit non-goals (v1)
- AI targeting specific students by name in group Q&A ("any member answers").
- Per-student remote CV reports from individual streams.
- PPT content feeding the RAG question generator.
