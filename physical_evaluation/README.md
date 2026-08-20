# Physical evaluation backend

This app is a secure kiosk/orchestration layer around the existing
`viva_evaluator`. It does not implement a second question or scoring engine.

## Project initialization

Create a physical project through `POST /api/projects/create/`:

```json
{
  "project_name": "Final Year Project",
  "evaluation_mode": "physical",
  "physical_location": "Engineering Building - Room 42",
  "physical_panel_pin": "a-private-panel-password"
}
```

The panel password is stored only as a Django password hash. The evaluation
mode is fixed on the project; all its sessions use the same mode.

## Kiosk flow

1. An assigned examiner logs in and calls
   `POST /api/physical/projects/{project_id}/kiosk/open/` with `{"pin":"..."}`.
2. The response clears the examiner's browser auth cookies and returns a
   limited `kiosk_token`.
3. Send that token as `X-Physical-Kiosk-Token` for every remaining kiosk call.
4. List today's sessions with `GET /api/physical/kiosk/sessions/`.
5. A student selects their session and calls
   `POST /api/physical/kiosk/sessions/{session_id}/start/` after local camera,
   microphone, and screen-capture permission has been obtained.
6. If a demo is enabled, call
   `POST /api/physical/kiosk/sessions/{session_id}/demo/complete/` when it ends.
7. Start the existing evaluator with `POST /api/viva/sessions/start/`, submit
   typed or speech-transcribed answers to
   `POST /api/viva/sessions/{session_id}/answer/`, and repeat until it returns
   `session_complete: true`.
8. Upload the full local recording as multipart `video_file` (and optional
   `audio_file`) to
   `POST /api/physical/kiosk/sessions/{session_id}/complete/`.
9. Lock the panel with `POST /api/physical/kiosk/close/` and the panel PIN.

The kiosk token cannot start or answer another session and is not recognized by
normal examiner/project endpoints. `GET /api/physical/kiosk/active/` lets a
refreshed frontend resume the active demo or viva.
