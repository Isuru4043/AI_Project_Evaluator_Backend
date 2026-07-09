"""
Create a group session for 3 students in the "Library test 2" project.

Steps:
  1. Register 3 test student accounts (skips if they already exist).
  2. Log in as an examiner (test.exam@uni.edu).
  3. Find the "Library test 2" project.
  4. Log in as student 1 and enroll the group (all 3 students).
  5. Log in as examiner and schedule a group session via manual scheduling.
"""

import os
import sys
import json

# Fix Windows console encoding for special characters
sys.stdout.reconfigure(encoding='utf-8')

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Evaluator_Backend.settings')
django.setup()

import requests

BASE = "http://127.0.0.1:8000"
PASSWORD = "Password123!"

# ── 3 test students ─────────────────────────────────────────────────────────
STUDENTS = [
    {
        "full_name": "Alice Johnson",
        "email": "alice.j@uni.edu",
        "registration_number": "REG-2026-001",
        "degree_program": "Computer Science",
        "academic_year": 3,
        "batch": "2024",
    },
    {
        "full_name": "Bob Williams",
        "email": "bob.w@uni.edu",
        "registration_number": "REG-2026-002",
        "degree_program": "Computer Science",
        "academic_year": 3,
        "batch": "2024",
    },
    {
        "full_name": "Carol Davis",
        "email": "carol.d@uni.edu",
        "registration_number": "REG-2026-003",
        "degree_program": "Computer Science",
        "academic_year": 3,
        "batch": "2024",
    },
]

# ── Examiner credentials ────────────────────────────────────────────────────
EXAMINER_EMAIL = "test.exam@uni.edu"
EXAMINER_PASSWORD = PASSWORD

# ── Target project name ─────────────────────────────────────────────────────
TARGET_PROJECT = "Library test 2"


def pretty(label, data):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(json.dumps(data, indent=2, default=str))


def api(method, url, token=None, json_data=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = getattr(requests, method.lower())(f"{BASE}{url}", headers=headers, json=json_data)
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:200]}
    return body, r.status_code


# =============================================================================
# STEP 1: Register 3 students (idempotent — will skip existing accounts)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 1: Register 3 students")
print("=" * 60)

from core.models import User, StudentProfile

for s in STUDENTS:
    existing = User.objects.filter(email=s["email"]).first()
    if existing:
        # Make sure password is set correctly
        existing.set_password(PASSWORD)
        existing.save()
        print(f"  [OK] {s['full_name']} ({s['email']}) -- already exists, password reset")
    else:
        body, code = api("POST", "/api/auth/student/register/", json_data={
            "full_name": s["full_name"],
            "email": s["email"],
            "password": PASSWORD,
            "confirm_password": PASSWORD,
            "registration_number": s["registration_number"],
            "degree_program": s["degree_program"],
            "academic_year": s["academic_year"],
            "batch": s["batch"],
        })
        if code == 201:
            print(f"  [OK] {s['full_name']} ({s['email']}) -- registered")
        else:
            print(f"  [FAIL] {s['full_name']} ({s['email']}) -- {body.get('message', body)}")

# =============================================================================
# STEP 2: Login as examiner
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 2: Login as examiner")
print("=" * 60)

# Ensure examiner password is correct
exam_user = User.objects.filter(email=EXAMINER_EMAIL).first()
if exam_user:
    exam_user.set_password(EXAMINER_PASSWORD)
    exam_user.save()

body, code = api("POST", "/api/auth/login/", json_data={
    "email": EXAMINER_EMAIL,
    "password": EXAMINER_PASSWORD,
})
if code != 200:
    print(f"  [FAIL] Examiner login failed: {body}")
    sys.exit(1)

EXAM_TOKEN = body["data"]["access"]
print(f"  [OK] Logged in as examiner ({EXAMINER_EMAIL})")

# =============================================================================
# STEP 3: Find the "Library test 2" project
# =============================================================================
print("\n" + "=" * 60)
print(f'  STEP 3: Find project "{TARGET_PROJECT}"')
print("=" * 60)

body, code = api("GET", "/api/projects/", token=EXAM_TOKEN)
if code != 200:
    print(f"  [FAIL] Failed to list projects: {body}")
    sys.exit(1)

project = None
for p in body.get("data", []):
    if p["project_name"] == TARGET_PROJECT:
        project = p
        break

if not project:
    print(f'  Project "{TARGET_PROJECT}" not found — creating it ...')
    body, code = api("POST", "/api/projects/create/", token=EXAM_TOKEN, json_data={
        "project_name": TARGET_PROJECT,
        "description": "Group project for library system evaluation",
        "is_group_project": True,
        "submission_deadline": "2026-12-31T23:59:59Z",
        "academic_year": "2026",
    })
    if code != 201:
        print(f"  [FAIL] Could not create project: {body}")
        sys.exit(1)
    project = body["data"]
    print(f"  [OK] Project created: {project['project_name']}")

    # Activate the project
    PROJECT_ID = project["id"]
    body, code = api("PATCH", f"/api/projects/{PROJECT_ID}/activate/", token=EXAM_TOKEN)
    if code == 200:
        print(f"  [OK] Project activated.")
    else:
        print(f"  [WARN] Activation response ({code}): {body.get('message', body)}")
else:
    print(f"  [OK] Found existing project: {project['project_name']}")

PROJECT_ID = project["id"]
IS_GROUP = project.get("is_group_project", False)
print(f"    ID:       {PROJECT_ID}")
print(f"    Group:    {IS_GROUP}")
print(f"    Status:   {project.get('status')}")

if not IS_GROUP:
    print(f"\n  [WARN] This project is NOT a group project — fixing ...")
    from core.models import Project as ProjectModel
    proj_obj = ProjectModel.objects.get(id=PROJECT_ID)
    proj_obj.is_group_project = True
    proj_obj.save()
    print(f"  [OK] Project updated to group project.")

# Make sure project is active
if project.get("status") != "active":
    print(f"  [WARN] Project status is '{project.get('status')}', activating...")
    from core.models import Project as ProjectModel
    proj_obj = ProjectModel.objects.get(id=PROJECT_ID)
    proj_obj.status = "active"
    proj_obj.save()
    print(f"  [OK] Project activated.")

# =============================================================================
# STEP 4: Student 1 enrolls the group (all 3 students)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 4: Enroll group of 3 students")
print("=" * 60)

# Login as student 1
body, code = api("POST", "/api/auth/login/", json_data={
    "email": STUDENTS[0]["email"],
    "password": PASSWORD,
})
if code != 200:
    print(f"  [FAIL] Student login failed: {body}")
    sys.exit(1)

STUD_TOKEN = body["data"]["access"]
print(f"  [OK] Logged in as {STUDENTS[0]['full_name']} ({STUDENTS[0]['email']})")

# Enroll with group — student 1 is the enrolling student,
# students 2 and 3 are listed as member_emails
GROUP_NAME = "Library Group Alpha"
body, code = api("POST", f"/api/projects/{PROJECT_ID}/enroll/", token=STUD_TOKEN, json_data={
    "group_number": GROUP_NAME,
    "member_emails": [STUDENTS[1]["email"], STUDENTS[2]["email"]],
})
if code == 201:
    print(f"  [OK] Group enrolled successfully!")
    print(f"    Group name: {GROUP_NAME}")
    print(f"    Members:    {', '.join(s['full_name'] for s in STUDENTS)}")
else:
    msg = body.get("message", body)
    print(f"  [WARN] Enrollment response ({code}): {msg}")
    if "already enrolled" in str(msg).lower():
        print(f"     (Group may already exist — continuing to schedule session)")
    else:
        pretty("Full response", body)

# =============================================================================
# STEP 5: Find the group ID
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 5: Find group ID")
print("=" * 60)

from core.models import StudentGroup, GroupMember

group = StudentGroup.objects.filter(project_id=PROJECT_ID).first()
if not group:
    print("  [FAIL] No group found for this project. Enrollment may have failed.")
    sys.exit(1)

GROUP_ID = str(group.id)
members = GroupMember.objects.filter(group=group).select_related("student__user")
print(f"  [OK] Group: {group.group_name} (ID: {GROUP_ID})")
print(f"    Members:")
for m in members:
    print(f"      - {m.student.user.full_name} ({m.student.registration_number})")

# =============================================================================
# STEP 6: Schedule a group session (manual)
# =============================================================================
print("\n" + "=" * 60)
print("  STEP 6: Schedule group evaluation session")
print("=" * 60)

from datetime import datetime, timedelta
from django.utils import timezone

# Schedule the session 1 hour from now, 30-minute slot
now = timezone.now()
start = now + timedelta(hours=1)
end = start + timedelta(minutes=30)

body, code = api("POST", f"/api/projects/{PROJECT_ID}/sessions/schedule/manual/",
    token=EXAM_TOKEN,
    json_data={
        "sessions": [
            {
                "group_id": GROUP_ID,
                "scheduled_start": start.isoformat(),
                "scheduled_end": end.isoformat(),
                "location_room": "Lab 204",
            }
        ]
    }
)

if code in (200, 201):
    print(f"  [OK] Group session scheduled successfully!")
    pretty("Session details", body.get("data", body))
else:
    pretty(f"Session scheduling failed ({code})", body)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("  [DONE] ALL DONE -- Summary")
print("=" * 60)
print(f"  Project:    {TARGET_PROJECT}")
print(f"  Group:      {GROUP_NAME}")
print(f"  Students:")
for s in STUDENTS:
    print(f"    - {s['full_name']} ({s['email']})")
print(f"  Session:    {start.strftime('%Y-%m-%d %H:%M')} — {end.strftime('%H:%M')}")
print(f"  Room:       Lab 204")
print(f"  Passwords:  {PASSWORD} (all accounts)")
print("=" * 60)
