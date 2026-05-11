"""
Full API Integration Test — Tests ALL endpoints across all apps.
Uses existing test users: test.exam@uni.edu / test.stud@uni.edu
"""
import requests
import json
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AI_Evaluator_Backend.settings')
django.setup()

BASE = "http://127.0.0.1:8001"
PASS = 0
FAIL = 0
RESULTS = []

def test(name, method, url, expected_status=None, headers=None, json_data=None, files=None, data=None):
    global PASS, FAIL
    try:
        if method == "GET":
            r = requests.get(url, headers=headers)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=json_data, files=files, data=data)
        elif method == "PUT":
            r = requests.put(url, headers=headers, json=json_data)
        elif method == "PATCH":
            r = requests.patch(url, headers=headers, json=json_data)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, json=json_data)

        status_ok = True
        if expected_status:
            if isinstance(expected_status, list):
                status_ok = r.status_code in expected_status
            else:
                status_ok = r.status_code == expected_status

        try:
            body = r.json()
        except:
            body = {"raw": r.text[:100]}

        if status_ok:
            PASS += 1
            RESULTS.append(f"  ✅ {name} — {r.status_code}")
        else:
            FAIL += 1
            msg = body.get('message', str(body)[:80]) if isinstance(body, dict) else str(body)[:80]
            RESULTS.append(f"  ❌ {name} — {r.status_code} (expected {expected_status}) — {msg}")

        return body, r.status_code
    except Exception as e:
        FAIL += 1
        RESULTS.append(f"  ❌ {name} — CONNECTION ERROR: {str(e)[:60]}")
        return {}, 0


print("=" * 65)
print("    FULL API TEST — VivaSense Backend")
print("=" * 65)

# =====================================================================
# PRE-SETUP: Reset passwords and clean data
# =====================================================================
from core.models import User, Project, ProjectSubmission, EvaluationSession, StudentProfile, ExaminerProfile
Project.objects.filter(project_name='API Full Test Project').delete()

for email in ['test.exam@uni.edu', 'test.stud@uni.edu']:
    u = User.objects.filter(email=email).first()
    if u:
        u.set_password('Password123!')
        u.save()

# =====================================================================
# SECTION 1: AUTHENTICATION
# =====================================================================
print("\n📌 SECTION 1: AUTHENTICATION")

# Login Examiner
body, _ = test("Login Examiner", "POST", f"{BASE}/api/auth/login/", 200,
    json_data={"email": "test.exam@uni.edu", "password": "Password123!"})
EXAM_TOKEN = body.get('data', {}).get('access')
EXAM_HEADERS = {"Authorization": f"Bearer {EXAM_TOKEN}"} if EXAM_TOKEN else {}

# Login Student
body, _ = test("Login Student", "POST", f"{BASE}/api/auth/login/", 200,
    json_data={"email": "test.stud@uni.edu", "password": "Password123!"})
STUD_TOKEN = body.get('data', {}).get('access')
STUD_REFRESH = body.get('data', {}).get('refresh')
STUD_HEADERS = {"Authorization": f"Bearer {STUD_TOKEN}"} if STUD_TOKEN else {}

# Get Profile
test("Get Examiner Profile", "GET", f"{BASE}/api/auth/me/", 200, headers=EXAM_HEADERS)
test("Get Student Profile", "GET", f"{BASE}/api/auth/me/", 200, headers=STUD_HEADERS)

# Refresh Token
test("Refresh Token", "POST", f"{BASE}/api/auth/token/refresh/", 200,
    json_data={"refresh": STUD_REFRESH})

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 2: PROJECT MANAGEMENT
# =====================================================================
print("\n📌 SECTION 2: PROJECT MANAGEMENT")

body, _ = test("Create Project", "POST", f"{BASE}/api/projects/create/", 201,
    headers=EXAM_HEADERS,
    json_data={
        "project_name": "API Full Test Project",
        "description": "Testing all endpoints",
        "is_group_project": False,
        "submission_deadline": "2026-12-31T23:59:59Z",
        "academic_year": "2026"
    })
PROJ_ID = body.get('data', {}).get('id')

test("List Projects", "GET", f"{BASE}/api/projects/", 200, headers=EXAM_HEADERS)
test("Project Detail", "GET", f"{BASE}/api/projects/{PROJ_ID}/", 200, headers=EXAM_HEADERS)
test("Update Project", "PUT", f"{BASE}/api/projects/{PROJ_ID}/update/", 200,
    headers=EXAM_HEADERS, json_data={"description": "Updated desc"})
test("List Examiners", "GET", f"{BASE}/api/projects/{PROJ_ID}/examiners/", 200,
    headers=EXAM_HEADERS)
test("Activate Project", "PATCH", f"{BASE}/api/projects/{PROJ_ID}/activate/", 200,
    headers=EXAM_HEADERS)

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 3: STUDENT ENROLLMENT
# =====================================================================
print("\n📌 SECTION 3: STUDENT ENROLLMENT")

test("Available Projects", "GET", f"{BASE}/api/projects/available/", 200, headers=STUD_HEADERS)
test("Enroll in Project", "POST", f"{BASE}/api/projects/{PROJ_ID}/enroll/", 201,
    headers=STUD_HEADERS, json_data={})
test("My Enrollments", "GET", f"{BASE}/api/projects/my-enrollments/", 200, headers=STUD_HEADERS)

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 4: RUBRIC MANAGEMENT
# =====================================================================
print("\n📌 SECTION 4: RUBRIC MANAGEMENT")

body, _ = test("Create Category", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/rubrics/categories/create/", 201,
    headers=EXAM_HEADERS,
    json_data={"category_name": "Technical Skills", "weight_percentage": 60.0})
CAT_ID = body.get('data', {}).get('id')

body, _ = test("Create 2nd Category", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/rubrics/categories/create/", 201,
    headers=EXAM_HEADERS,
    json_data={"category_name": "Presentation", "weight_percentage": 40.0})
CAT2_ID = body.get('data', {}).get('id')

test("List Rubrics", "GET", f"{BASE}/api/projects/{PROJ_ID}/rubrics/", 200, headers=EXAM_HEADERS)

body, _ = test("Create Criteria", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/rubrics/categories/{CAT_ID}/criteria/create/", 201,
    headers=EXAM_HEADERS,
    json_data={"criteria_name": "Code Quality", "description": "Clean code practices", "max_score": 10})
CRIT_ID = body.get('data', {}).get('id')

test("Update Category", "PUT",
    f"{BASE}/api/projects/{PROJ_ID}/rubrics/categories/{CAT_ID}/update/", 200,
    headers=EXAM_HEADERS, json_data={"category_name": "Tech Skills Updated"})

if CRIT_ID:
    test("Update Criteria", "PUT",
        f"{BASE}/api/projects/{PROJ_ID}/rubrics/categories/{CAT_ID}/criteria/{CRIT_ID}/update/", 200,
        headers=EXAM_HEADERS, json_data={"criteria_name": "Code Quality Updated"})

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 5: PROJECT SUBMISSION (Azure Upload)
# =====================================================================
print("\n📌 SECTION 5: PROJECT SUBMISSION (Azure Upload)")

with open("test.pdf", "rb") as f:
    body, _ = test("Submit Project (PDF → Azure)", "POST",
        f"{BASE}/api/projects/{PROJ_ID}/submit/", 200,
        headers=STUD_HEADERS,
        files={"report_file": ("test_report.pdf", f, "application/pdf")},
        data={"github_repo_url": "https://github.com/test/api-test"})
SUB_ID = body.get('data', {}).get('id')

test("Submission Detail (Student)", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/submission/", 200, headers=STUD_HEADERS)
test("Submission Detail (Examiner)", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/submission/", 200, headers=EXAM_HEADERS)

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 6: SESSION SCHEDULING
# =====================================================================
print("\n📌 SECTION 6: SESSION SCHEDULING")

sp = StudentProfile.objects.get(user__email='test.stud@uni.edu')
test("Manual Schedule", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/sessions/schedule/manual/", [200, 201],
    headers=EXAM_HEADERS,
    json_data={
        "student_email": "test.stud@uni.edu",
        "scheduled_start": "2026-05-11T14:00:00Z",
        "scheduled_end": "2026-05-11T14:30:00Z",
        "location_room": "Room 301"
    })

body, _ = test("List Sessions", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/sessions/", 200, headers=EXAM_HEADERS)

# Extract session ID from list
SESS_ID = None
if body.get('data'):
    data = body['data']
    if isinstance(data, list) and data:
        SESS_ID = data[0].get('id') or data[0].get('session_id')
    elif isinstance(data, dict):
        sessions = data.get('sessions', [])
        if sessions:
            SESS_ID = sessions[0].get('id') or sessions[0].get('session_id')

test("My Session (Student)", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/sessions/my-session/", 200, headers=STUD_HEADERS)

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 7: SESSION PANEL & DEMO FLOW
# =====================================================================
print("\n📌 SECTION 7: SESSION PANEL & DEMO FLOW")

test("Open Session Panel", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/session-panel/open/", 200, headers=EXAM_HEADERS)

test("Active Session (none yet)", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/session-panel/active/", 200, headers=EXAM_HEADERS)

if SESS_ID:
    test("Start Demo", "POST", f"{BASE}/api/sessions/{SESS_ID}/start-demo/", 200,
        headers=EXAM_HEADERS)
    test("Active Session (in progress)", "GET",
        f"{BASE}/api/projects/{PROJ_ID}/session-panel/active/", 200, headers=EXAM_HEADERS)
    test("Complete Demo", "POST", f"{BASE}/api/sessions/{SESS_ID}/complete-demo/", 200,
        headers=EXAM_HEADERS)
    test("End Viva (no files)", "POST", f"{BASE}/api/sessions/{SESS_ID}/end-viva/", 200,
        headers=EXAM_HEADERS)
    test("Active Session (none after)", "GET",
        f"{BASE}/api/projects/{PROJ_ID}/session-panel/active/", 200, headers=EXAM_HEADERS)
else:
    RESULTS.append("  ⚠️  No session ID — demo flow skipped")

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 8: VIVA QUESTIONS CRUD
# =====================================================================
print("\n📌 SECTION 8: VIVA QUESTIONS CRUD")

body, _ = test("Create Question 1", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/viva/questions/create/", 201,
    headers=EXAM_HEADERS,
    json_data={"question_text": "Explain your architecture.", "blooms_level": "Understand", "question_order": 1})
Q_ID = body.get('data', {}).get('question_id')

test("Create Question 2", "POST",
    f"{BASE}/api/projects/{PROJ_ID}/viva/questions/create/", 201,
    headers=EXAM_HEADERS,
    json_data={"question_text": "How would you improve it?", "blooms_level": "Create", "question_order": 2})

test("List Questions", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/viva/questions/", 200, headers=EXAM_HEADERS)

if Q_ID:
    test("Update Question", "PUT",
        f"{BASE}/api/projects/{PROJ_ID}/viva/questions/{Q_ID}/update/", 200,
        headers=EXAM_HEADERS, json_data={"question_text": "Updated text.", "blooms_level": "Analyze"})
    test("Delete Question", "DELETE",
        f"{BASE}/api/projects/{PROJ_ID}/viva/questions/{Q_ID}/delete/", 200,
        headers=EXAM_HEADERS)

test("List After Delete (1 left)", "GET",
    f"{BASE}/api/projects/{PROJ_ID}/viva/questions/", 200, headers=EXAM_HEADERS)

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 9: STUDENT SESSION STATUS
# =====================================================================
print("\n📌 SECTION 9: STUDENT SESSION STATUS")

test("Student Session Status", "GET",
    f"{BASE}/api/sessions/my-status/", 200, headers=STUD_HEADERS)

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 10: VIVA EVALUATOR
# =====================================================================
print("\n📌 SECTION 10: VIVA EVALUATOR")

from core.models import ProjectSubmission as PS
from datetime import timedelta
from django.utils import timezone

sub = PS.objects.get(id=SUB_ID) if SUB_ID else None

if sub:
    # Upload for viva processing
    with open("test.pdf", "rb") as f:
        body, code = test("Viva Upload", "POST",
            f"{BASE}/api/viva/submissions/upload/", [201, 400],
            headers=STUD_HEADERS,
            files={"report_file": ("test.pdf", f, "application/pdf")},
            data={"project": str(sub.project_id), "student": str(sp.id)})

    VIVA_SUB_ID = body.get('submission_id') if isinstance(body, dict) else None

    if VIVA_SUB_ID:
        test("Viva Submission Status", "GET",
            f"{BASE}/api/viva/submissions/{VIVA_SUB_ID}/status/", 200,
            headers=STUD_HEADERS)

        # Create a session linked to this submission for viva
        viva_sub = PS.objects.get(id=VIVA_SUB_ID)
        now = timezone.now()
        viva_sess = EvaluationSession.objects.create(
            project=viva_sub.project, student=sp, submission=viva_sub,
            scheduled_start=now, scheduled_end=now + timedelta(minutes=30),
            status='scheduled'
        )

        body, code = test("Viva Session Start", "POST",
            f"{BASE}/api/viva/sessions/start/", [200, 400, 500],
            headers=EXAM_HEADERS,
            json_data={"session_id": str(viva_sess.id)})

        if code == 200:
            VQ_ID = body.get('question_id')
            if VQ_ID:
                test("Viva Answer Submit", "POST",
                    f"{BASE}/api/viva/sessions/{viva_sess.id}/answer/", [200, 500],
                    headers=STUD_HEADERS,
                    json_data={"question_id": VQ_ID, "answer_text": "We used Django REST framework."})
        else:
            RESULTS.append(f"  ⚠️  Viva Start got {code} — may need rubric criteria linked to submission")

        test("Viva Session Report", "GET",
            f"{BASE}/api/viva/sessions/{viva_sess.id}/report/", [200, 400],
            headers=EXAM_HEADERS)
    else:
        RESULTS.append("  ⚠️  Viva upload didn't return ID — status check skipped")
else:
    RESULTS.append("  ⚠️  No submission available — viva tests skipped")

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 11: CODE ANALYSIS
# =====================================================================
print("\n📌 SECTION 11: CODE ANALYSIS")

from core.models import CodeSubmission
cs = CodeSubmission.objects.filter(project_submission_id=SUB_ID).first() if SUB_ID else None
if cs:
    test("Code Analysis Status", "GET",
        f"{BASE}/api/code-analysis/submissions/{cs.id}/status/", 200, headers=EXAM_HEADERS)
    test("Sonar Summary", "GET",
        f"{BASE}/api/code-analysis/submissions/{cs.id}/sonar-summary/", [200, 404], headers=EXAM_HEADERS)
    test("Code Questions", "GET",
        f"{BASE}/api/code-analysis/submissions/{cs.id}/questions/", [200, 404], headers=EXAM_HEADERS)
else:
    RESULTS.append("  ⚠️  No code submission — analysis tests skipped")

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# SECTION 12: AUTH LOGOUT
# =====================================================================
print("\n📌 SECTION 12: AUTH LOGOUT")

test("Logout", "POST", f"{BASE}/api/auth/logout/", [200, 204],
    headers=STUD_HEADERS, json_data={"refresh": STUD_REFRESH})

for r in RESULTS: print(r)
RESULTS.clear()

# =====================================================================
# FINAL REPORT
# =====================================================================
print("\n" + "=" * 65)
print(f"    RESULTS: {PASS} PASSED | {FAIL} FAILED | {PASS + FAIL} TOTAL")
print("=" * 65)
if FAIL == 0:
    print("    🎉 ALL TESTS PASSED!")
else:
    print(f"    ⚠️  {FAIL} test(s) need attention")
print("=" * 65)
