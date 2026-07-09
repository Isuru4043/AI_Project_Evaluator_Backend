"""
Create an evaluation session directly via the Django ORM, automatically provisioning
the project, examiner, students, submission, and a fully initialized FAISS RAG index.

Run it from the backend root with the backend venv active:

    python create_session_direct.py
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
from django.db import transaction

from core.models import (
    EvaluationSession,
    Project,
    ProjectExaminer,
    ProjectSubmission,
    StudentGroup,
    GroupMember,
    StudentProfile,
    ExaminerProfile,
    RubricCategory,
    RubricCriteria,
    User,
)
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.services.rag.vector_store import save_index_for_submission

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_NAME = "test1"
EXAMINER_EMAIL = "examiner_test1@vivasense.tech"
STUDENT_1_EMAIL = "student1_test1@vivasense.tech"
STUDENT_2_EMAIL = "student2_test1@vivasense.tech"
KNOWN_PASSWORD = "Password123!"
FRONTEND_BASE = "http://localhost:3000"
ROOM = "Lab 204"
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_user(email, password, full_name, role):
    user = User.objects.filter(email=email).first()
    if not user:
        user = User.objects.create_user(
            email=email,
            password=password,
            full_name=full_name,
            role=role,
        )
        print(f"  [OK] Created User: {email} (role: {role})")
    else:
        user.full_name = full_name
        user.set_password(password)
        user.save()
        print(f"  [OK] Reset User: {email} (role: {role})")
    return user


with transaction.atomic():
    print("\n" + "=" * 60)
    print("  Creating / Provisioning Test Session: 'test1'")
    print("=" * 60)

    # 1. Examiner Profile
    ex_user = get_or_create_user(EXAMINER_EMAIL, KNOWN_PASSWORD, "Examiner Test1", User.Role.EXAMINER)
    examiner_profile, _ = ExaminerProfile.objects.get_or_create(
        user=ex_user,
        defaults={
            "employee_id": "EMP_T1_001",
            "department": "Computer Science",
            "designation": "Senior Lecturer",
        }
    )

    # 2. Student Profiles
    s1_user = get_or_create_user(STUDENT_1_EMAIL, KNOWN_PASSWORD, "Student1 Test1", User.Role.STUDENT)
    student1_profile, _ = StudentProfile.objects.get_or_create(
        user=s1_user,
        defaults={
            "registration_number": "REG_T1_001",
            "degree_program": "Software Engineering",
            "academic_year": 4,
            "batch": "2022",
        }
    )

    s2_user = get_or_create_user(STUDENT_2_EMAIL, KNOWN_PASSWORD, "Student2 Test1", User.Role.STUDENT)
    student2_profile, _ = StudentProfile.objects.get_or_create(
        user=s2_user,
        defaults={
            "registration_number": "REG_T1_002",
            "degree_program": "Software Engineering",
            "academic_year": 4,
            "batch": "2022",
        }
    )

    # 3. Project "test1" (Group Project)
    project, created = Project.objects.get_or_create(
        project_name=PROJECT_NAME,
        defaults={
            "description": "Adaptive group viva testing environment.",
            "is_group_project": True,
            "status": Project.Status.ACTIVE,
            "submission_deadline": timezone.now() + timedelta(days=7),
            "academic_year": "2026",
        }
    )
    if created:
        print(f"  [OK] Created Project: {PROJECT_NAME}")
    else:
        project.is_group_project = True
        project.status = Project.Status.ACTIVE
        project.save()
        print(f"  [OK] Found existing Project: {PROJECT_NAME}")

    # 4. Assign Examiner to Project
    pe, _ = ProjectExaminer.objects.get_or_create(
        project=project,
        examiner=examiner_profile,
        defaults={"role_in_project": ProjectExaminer.RoleInProject.LEAD}
    )

    # 5. Create Rubric Categories & Criteria (so the viva start call doesn't fail)
    cat, _ = RubricCategory.objects.get_or_create(
        project=project,
        category_name="Technical Implementation",
        defaults={"weight_percentage": 50.0, "description": "System architecture and coding patterns."}
    )

    crit1, _ = RubricCriteria.objects.get_or_create(
        category=cat,
        criteria_name="Secure Authentication",
        defaults={
            "max_score": 10.0,
            "weight_in_category": 50.0,
            "description": "Checks student's design of login routes, token exchange, and password storage.",
            "questions_to_ask": 2,
        }
    )

    crit2, _ = RubricCriteria.objects.get_or_create(
        category=cat,
        criteria_name="Database & Architecture",
        defaults={
            "max_score": 10.0,
            "weight_in_category": 50.0,
            "description": "Checks student's setup of database connections and schema integrity.",
            "questions_to_ask": 2,
        }
    )

    # 6. Create Student Group and members
    group, _ = StudentGroup.objects.get_or_create(
        project=project,
        group_name="Group test1"
    )
    GroupMember.objects.get_or_create(group=group, student=student1_profile)
    GroupMember.objects.get_or_create(group=group, student=student2_profile)
    print(f"  [OK] Group and members assigned: {group.group_name}")

    # 7. Project Submission
    submission, _ = ProjectSubmission.objects.get_or_create(
        project=project,
        group=group,
        defaults={
            "submitted_at": timezone.now(),
            "github_repo_url": "https://github.com/test1/test1-repo",
            "report_file_url": "https://vivasense.blob.core.windows.net/media/dummy_report.pdf",
        }
    )

    # 8. Create fully embedded FAISS Index & Report Chunks (ensures RAG retrieval works!)
    dummy_chunks = [
        {
            "text": "The system implements secure authentication using bcrypt for hashing password credentials on signup and signin. JWT access tokens are stored in HttpOnly cookies to mitigate XSS risk.",
            "source": "report",
            "metadata": {"section": "Security Architecture", "page": 5}
        },
        {
            "text": "def hash_password(password):\n    # Hashing credentials with bcrypt salt\n    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())",
            "source": "code",
            "metadata": {"file": "authentication/utils.py"}
        },
        {
            "text": "The database uses PostgreSQL with connection pooling. It runs on a serverless Neon cloud cluster utilizing SSL modes for all active transport links.",
            "source": "report",
            "metadata": {"section": "Database Architecture", "page": 8}
        },
        {
            "text": "DATABASES = {\n    'default': {\n        'ENGINE': 'django.db.backends.postgresql',\n        'NAME': 'neondb',\n        'OPTIONS': {'sslmode': 'require'}\n    }\n}",
            "source": "code",
            "metadata": {"file": "AI_Evaluator_Backend/settings.py"}
        }
    ]

    print("  [..] Initializing FAISS Index for RAG retrieval...")
    save_index_for_submission(submission, dummy_chunks)
    index_status = SubmissionIndexStatus.objects.get(submission=submission)
    index_status.status = SubmissionIndexStatus.IndexStatus.READY
    index_status.save()
    print("  [OK] FAISS Index & Chunks generated successfully.")

    # 9. Create scheduled Evaluation Session
    now = timezone.now()
    start = now + timedelta(minutes=1)
    end = start + timedelta(minutes=45)

    session = EvaluationSession.objects.create(
        project=project,
        group=group,
        submission=submission,
        scheduled_start=start,
        scheduled_end=end,
        location_room=ROOM,
        status=EvaluationSession.Status.SCHEDULED,
        demo_enabled=True,
        agora_channel_name=str(timezone.now().timestamp()),
    )

print("\n" + "=" * 60)
print("  [DONE] Test session successfully created!")
print("=" * 60)
print(f"  Session ID:   {session.id}")
print(f"  Project ID:   {project.id}")
print(f"  Phase:        {session.phase}  (demo_enabled={session.demo_enabled})")
print(f"  Scheduled:    {session.scheduled_start:%Y-%m-%d %H:%M} — {session.scheduled_end:%H:%M}")
print(f"  Room:         {session.location_room}")
print(f"  Participants:")
print(f"    - {student1_profile.user.full_name} ({student1_profile.registration_number})")
print(f"    - {student2_profile.user.full_name} ({student2_profile.registration_number})")

print("\n  Open as STUDENT (Start Demo / Start Viva):")
print(f"    {FRONTEND_BASE}/dashboard/student/sessions/{session.id}/live")
print("\n  Open as EXAMINER (Join once student has started):")
print(f"    {FRONTEND_BASE}/dashboard/teacher")
print("=" * 60)

print("\n" + "=" * 60)
print("  LOGIN CREDENTIALS (reset to password: 'Password123!')")
print("=" * 60)
print(f"  Examiner:")
print(f"    email:    {EXAMINER_EMAIL}")
print(f"    password: {KNOWN_PASSWORD}")
print(f"  Student 1:")
print(f"    email:    {STUDENT_1_EMAIL}")
print(f"    password: {KNOWN_PASSWORD}")
print(f"  Student 2:")
print(f"    email:    {STUDENT_2_EMAIL}")
print(f"    password: {KNOWN_PASSWORD}")
print("=" * 60 + "\n")
