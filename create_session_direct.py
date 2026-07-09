"""
Create an evaluation session DIRECTLY via the Django ORM (no API, no dev server).

Unlike create_group_session.py — which registers/enrolls over the REST API and
schedules through /sessions/schedule/manual/ — this script assumes the project
and its group (or student) already exist and just inserts one EvaluationSession
row. Handy for quickly spinning up a fresh, testable session for the new
lifecycle: scheduled -> demo_in_progress -> viva_in_progress -> completed.

Run it from the backend root with the backend venv active:

    python create_session_direct.py

Then open the printed live URL as the student to click "Start Demo" / "Start
Viva", and the examiner panel URL to Join once the student has started.
"""

import os
import sys

# Fix Windows console encoding for special characters
sys.stdout.reconfigure(encoding="utf-8")

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")
django.setup()

from datetime import timedelta

from django.utils import timezone

from core.models import (
    EvaluationSession,
    Project,
    ProjectExaminer,
    ProjectSubmission,
    StudentGroup,
    GroupMember,
    StudentProfile,
    User,
)

# ── CONFIG — edit these ──────────────────────────────────────────────────────
TARGET_PROJECT = "Library test 2"   # project_name to attach the session to
GROUP_NAME = None                    # group_name for group projects; None = first group
STUDENT_EMAIL = None                 # email for individual projects; None = first submitter
DEMO_ENABLED = True                  # examiner-set: does this session have a demo phase?
ROOM = "Lab 204"
START_DELAY_MIN = 1                  # scheduled_start = now + this many minutes
DURATION_MIN = 45                    # scheduled_end = scheduled_start + this many minutes
FRONTEND_BASE = "http://localhost:3000"

# Real passwords are hashed in the DB and can't be recovered, so every account
# printed at the end is reset to this known password.
KNOWN_PASSWORD = "Password123!"
# ─────────────────────────────────────────────────────────────────────────────


def fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


print("\n" + "=" * 60)
print("  Create evaluation session (direct ORM insert)")
print("=" * 60)

# ── 1. Find the project ──────────────────────────────────────────────────────
project = Project.objects.filter(project_name=TARGET_PROJECT).first()
if not project:
    fail(f'Project "{TARGET_PROJECT}" not found. Create/enroll it first.')

print(f"  [OK] Project: {project.project_name}")
print(f"       ID:     {project.id}")
print(f"       Group:  {project.is_group_project}")

# ── 2. Resolve the target (group or individual) + its submission ─────────────
group = None
student = None
participants = []  # list of (full_name, registration_number, User)

if project.is_group_project:
    qs = StudentGroup.objects.filter(project=project)
    group = qs.filter(group_name=GROUP_NAME).first() if GROUP_NAME else qs.first()
    if not group:
        fail("No group found for this project. Enroll a group first.")
    print(f"  [OK] Group:  {group.group_name} (ID: {group.id})")

    for m in GroupMember.objects.filter(group=group).select_related("student__user"):
        participants.append((m.student.user.full_name, m.student.registration_number, m.student.user))

    submission = ProjectSubmission.objects.filter(project=project, group=group).first()
else:
    if STUDENT_EMAIL:
        user = User.objects.filter(email=STUDENT_EMAIL).first()
        student = getattr(user, "student_profile", None) if user else None
    else:
        sub = ProjectSubmission.objects.filter(
            project=project, student__isnull=False
        ).select_related("student__user").first()
        student = sub.student if sub else None
    if not student:
        fail("No student found for this project. Enroll/submit as a student first.")
    print(f"  [OK] Student: {student.user.full_name} ({student.registration_number})")
    participants.append((student.user.full_name, student.registration_number, student.user))

    submission = ProjectSubmission.objects.filter(project=project, student=student).first()

# ── 2b. Resolve the examiner(s) assigned to this project ────────────────────
examiners = [
    pe.examiner
    for pe in ProjectExaminer.objects.filter(project=project).select_related(
        "examiner__user"
    )
]
if not examiners:
    fail("No examiner is assigned to this project. Assign one first.")

if submission:
    print(f"  [OK] Submission linked: {submission.id}")
else:
    print("  [WARN] No submission found — the viva can't generate questions until "
          "a processed submission exists for this group/student.")

# ── 3. Create the session row directly ───────────────────────────────────────
# NOTE: under the new lifecycle, status is button-driven (student clicks Start
# Demo / Start Viva), not clock-driven — scheduled_start/end are just the
# displayed window, not an auto-transition trigger.
now = timezone.now()
start = now + timedelta(minutes=START_DELAY_MIN)
end = start + timedelta(minutes=DURATION_MIN)

session = EvaluationSession.objects.create(
    project=project,
    group=group,
    student=student,
    submission=submission,
    scheduled_start=start,
    scheduled_end=end,
    location_room=ROOM,
    status="scheduled",             # student drives the transitions from here
    demo_enabled=DEMO_ENABLED,
    agora_channel_name="bruno",
)

print("\n" + "=" * 60)
print("  [DONE] Session created")
print("=" * 60)
print(f"  Session ID:   {session.id}")
print(f"  Phase:        {session.phase}  (demo_enabled={session.demo_enabled})")
print(f"  Created at:   {now:%Y-%m-%d %H:%M:%S}")
print(f"  Scheduled:    {session.scheduled_start:%Y-%m-%d %H:%M} — "
      f"{session.scheduled_end:%H:%M}  "
      f"(starts in {START_DELAY_MIN} min, lasts {DURATION_MIN} min)")
print(f"  Room:         {session.location_room}")
print(f"  Participants:")
for name, reg, _ in participants:
    print(f"    - {name} ({reg})")

print("\n  Open as STUDENT (Start Demo / Start Viva):")
print(f"    {FRONTEND_BASE}/dashboard/student/sessions/{session.id}/live")
if group:
    print("    (all group members open this same URL)")
print("\n  Open as EXAMINER (Join once the student has started):")
print(f"    {FRONTEND_BASE}/dashboard/teacher/projects/{project.id}")
print("=" * 60)

# ── 4. Reset + print full credentials for everyone in this session ──────────
# Real passwords are hashed in the DB and can't be recovered, so every
# involved account is reset to KNOWN_PASSWORD here and printed below.
for _, _, user in participants:
    user.set_password(KNOWN_PASSWORD)
    user.save()

for ep in examiners:
    ep.user.set_password(KNOWN_PASSWORD)
    ep.user.save()

print("\n" + "=" * 60)
print("  LOGIN CREDENTIALS  (passwords reset to KNOWN_PASSWORD for this run)")
print("=" * 60)
print("  Examiner(s):")
for ep in examiners:
    print(f"    - {ep.user.full_name or '(no name)'}")
    print(f"        email:    {ep.user.email}")
    print(f"        password: {KNOWN_PASSWORD}")
print("\n  Student(s):")
for name, reg, user in participants:
    print(f"    - {name or '(no name)'} ({reg})")
    print(f"        email:    {user.email}")
    print(f"        password: {KNOWN_PASSWORD}")
print("=" * 60)
