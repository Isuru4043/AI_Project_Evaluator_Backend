"""Speaker attribution API.

    POST /api/sessions/<id>/attribution/evidence/   ingest a provider batch
    POST /api/sessions/<id>/attribution/bind/       physical: bind faces to students
    GET  /api/sessions/<id>/attribution/bind/       current seat bindings
    GET  /api/sessions/<id>/attribution/answers/    examiner review queue
    POST /api/sessions/<id>/attribution/answers/<answer_id>/confirm/
    POST /api/sessions/<id>/attribution/reconcile/  re-resolve with post-hoc CV
    GET  /api/sessions/<id>/attribution/unknown-speakers/   people seen, not named
    POST /api/sessions/<id>/attribution/unknown-speakers/   name one of them

Evidence ingest is open to session participants and the kiosk — the students'
own browsers are the providers. Reviewing and confirming attribution is
examiner-only: it decides where a score is filed, which is not a student's
call to make.
"""

import logging

from django.utils import timezone
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from authentication.authentication import CookieJWTAuthentication
from attribution.authentication import (
    ExamStationAuthentication,
    is_station_principal,
)
from core.models import (
    EvaluationSession,
    ExaminerProfile,
    GroupMember,
    ProjectExaminer,
    StudentProfile,
    VivaAnswer,
)
from physical_evaluation.authentication import PhysicalKioskAuthentication
from physical_evaluation.models import PhysicalEvaluationRun, PhysicalKioskAccess
from attribution.models import AnswerAttribution, UnknownSpeaker
from attribution.services import binding as binding_service
from attribution.services import engine, ingest
from projects.permissions import IsExaminer

logger = logging.getLogger(__name__)

MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_BINDING_FRAMES = 10
MAX_BINDING_TOTAL_BYTES = 32 * 1024 * 1024
SPEAKER_TEST_EMAILS = {
    'student@university.edu',
    'isuru.akalanka8058@gmail.com',
}


def _err(message, code=400):
    return Response({'success': False, 'message': message}, status=code)


def _ok(data, message='', code=200):
    body = {'success': True, 'data': data}
    if message:
        body['message'] = message
    return Response(body, status=code)


def _read_binding_frames(request):
    """Read a bounded camera burst while accepting the old `frame` field."""
    uploads = request.FILES.getlist('frames') or request.FILES.getlist('frame')
    if not uploads:
        return None, 'Send one or more camera frames.'
    if len(uploads) > MAX_BINDING_FRAMES:
        return None, f'Too many frames (max {MAX_BINDING_FRAMES}).'
    if any(upload.size > MAX_FRAME_BYTES for upload in uploads):
        return None, 'A frame is too large (max 8MB each).'
    if sum(upload.size for upload in uploads) > MAX_BINDING_TOTAL_BYTES:
        return None, 'Camera burst is too large (max 32MB total).'
    return [upload.read() for upload in uploads], None


def _get_session(session_id):
    return (
        EvaluationSession.objects
        .filter(id=session_id)
        .select_related('project', 'student__user', 'group')
        .first()
    )


def _is_examiner_for(user, session) -> bool:
    ep = ExaminerProfile.objects.filter(user=user).first()
    return bool(ep) and ProjectExaminer.objects.filter(
        project=session.project, examiner=ep,
    ).exists()


def _is_participant(user, session) -> bool:
    """A session participant, the kiosk, or the assigned examiner."""
    if is_station_principal(user):
        return True
    if session.student and session.student.user_id == user.id:
        return True
    if session.group_id and GroupMember.objects.filter(
        group_id=session.group_id, student__user=user,
    ).exists():
        return True
    return _is_examiner_for(user, session)


def _is_trusted_device(request) -> bool:
    return bool(
        getattr(request.user, 'is_station', False)
        or isinstance(request.auth, PhysicalKioskAccess)
    )


def _filter_browser_volume_events(request, events):
    """A browser may report only its authenticated account's Agora UID."""
    from agora_service.token_builder import _uid_from_user_id

    expected_uid = str(_uid_from_user_id(request.user.id))
    return [
        event for event in events
        if isinstance(event, dict)
        and str(event.get('uid', '')) == expected_uid
    ]


class EvidenceIngestView(APIView):
    """POST /api/sessions/<session_id>/attribution/evidence/

    Body: {"source": "agora_volume" | "agora_stt" | "live_cv", "events": [...]}

    Providers batch their observations rather than posting per event — a
    volume indicator reports periodically and one request per sample would
    swamp the tier for no gain.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
        CookieJWTAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_participant(request.user, session):
            return _err('You are not part of this session.', code=403)
        if session.status != EvaluationSession.Status.IN_PROGRESS:
            return _err(
                'Speaker evidence is accepted only while the viva is in progress.',
                code=409,
            )

        source = (request.data.get('source') or '').strip()
        events = request.data.get('events') or []
        if not isinstance(events, list):
            return _err('events must be a list.')
        if not events:
            return _ok({'stored': 0})
        if len(events) > 2000:
            return _err('Too many events in one batch (max 2000).')

        trusted_device = _is_trusted_device(request)
        if trusted_device and source != 'live_cv':
            return _err('A physical station may submit only live_cv evidence.', code=403)
        if not trusted_device:
            if source != 'agora_volume':
                return _err('Browser participants may submit only agora_volume evidence.', code=403)
            events = _filter_browser_volume_events(request, events)
            if not events:
                return _ok({'stored': 0, 'source': source})

        try:
            if source == 'agora_volume':
                stored = ingest.ingest_agora_volume(session, events)
            elif source == 'agora_stt':
                stored = ingest.ingest_stt_turns(session, events)
            elif source == 'live_cv':
                stored = ingest.ingest_live_cv(session, events)
            else:
                return _err(
                    'source must be one of: agora_volume, agora_stt, live_cv.'
                )
        except Exception:
            logger.exception('Evidence ingest failed for session %s', session_id)
            return _err('Could not store the evidence batch.', code=500)

        return _ok({'stored': stored, 'source': source})


class SeatBindingView(APIView):
    """GET/POST /api/sessions/<session_id>/attribution/bind/

    POST binds the faces in one still frame to roster students (physical group
    sessions). Send either a `frame` file, or {"seating_order": [student_id...]}
    to fall back to left-to-right seating.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
        CookieJWTAuthentication,
    ]
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_participant(request.user, session):
            return _err('You are not part of this session.', code=403)
        return _ok({
            'bindings': binding_service.current_bindings(session),
            'missing_enrollment': binding_service.missing_enrollments(session),
        })

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_participant(request.user, session):
            return _err('You are not part of this session.', code=403)
        if session.status != EvaluationSession.Status.IN_PROGRESS:
            return _err(
                'Seat binding is available only while the viva is in progress.',
                code=409,
            )
        if not (_is_trusted_device(request) or _is_examiner_for(request.user, session)):
            return _err(
                'Only the kiosk, exam station, or assigned examiner may bind seats.',
                code=403,
            )

        seating = request.data.get('seating_order')
        if seating:
            if isinstance(seating, str):
                seating = [s for s in seating.split(',') if s]
            return _ok(
                binding_service.bind_by_seating(session, list(seating)),
                'Bound by seating order.',
            )

        frames, frame_error = _read_binding_frames(request)
        if frame_error:
            return _err(f'{frame_error} Alternatively, send a seating_order list.')

        try:
            result = binding_service.bind_from_frames(session, frames)
        except Exception as e:
            logger.exception('Seat binding failed for session %s', session_id)
            return _err(f'Face binding failed: {e}', code=502)

        run = PhysicalEvaluationRun.objects.filter(session=session).first()
        if run is not None:
            run.identity_verification = result
            update_fields = ['identity_verification', 'updated_at']
            if result.get('complete'):
                run.identity_status = PhysicalEvaluationRun.IdentityStatus.VERIFIED
                run.identity_verified_at = timezone.now()
                update_fields.extend(['identity_status', 'identity_verified_at'])
            elif run.identity_status != PhysicalEvaluationRun.IdentityStatus.OVERRIDDEN:
                run.identity_status = PhysicalEvaluationRun.IdentityStatus.PENDING
                run.identity_verified_at = None
                update_fields.extend(['identity_status', 'identity_verified_at'])
            run.save(update_fields=update_fields)

        message = (
            'All expected group members were verified.'
            if result.get('complete')
            else 'Identity review is incomplete. Retry or use an examiner override.'
        )
        # Incomplete verification is a valid review result, not a transport
        # error: the kiosk needs the roster diagnostics to explain what failed.
        return _ok(result, message)


class SpeakerDetectionTestView(APIView):
    """Examiner-only face binding test; never writes session evidence."""

    authentication_classes = [CookieJWTAuthentication]
    permission_classes = [IsAuthenticated, IsExaminer]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        frames, frame_error = _read_binding_frames(request)
        if frame_error:
            return _err(frame_error)

        students = list(
            StudentProfile.objects
            .filter(user__email__in=SPEAKER_TEST_EMAILS)
            .select_related('user')
        )
        found = {student.user.email.lower() for student in students}
        missing_accounts = sorted(SPEAKER_TEST_EMAILS - found)
        photos = {
            str(student.id): student.enrollment_face_photos()
            for student in students
            if student.enrollment_face_photos()
        }
        missing_photos = sorted(
            student.user.email
            for student in students
            if not student.enrollment_face_photos()
        )
        if not photos:
            return _err('Neither test student has an enrollment face photo.')

        try:
            matches = binding_service.match_test_frames(frames, photos)
        except Exception as exc:
            logger.exception('Speaker detection diagnostic binding failed')
            return _err(f'Face binding failed: {exc}', code=502)

        by_id = {str(student.id): student for student in students}
        bindings = []
        for match in matches:
            student = by_id.get(str(match.get('student_id') or ''))
            bindings.append({
                'student_id': str(student.id) if student else None,
                'student_name': student.user.full_name if student else 'Unknown face',
                'email': student.user.email if student else None,
                'bbox': match.get('bbox'),
                'confidence': float(match.get('confidence', 0) or 0),
                'identity_confidence': match.get('identity_confidence'),
                'votes': int(match.get('votes', 1) or 0),
                'frames_processed': int(
                    match.get('frames_processed', len(frames)) or len(frames)
                ),
            })
        frames_processed = max(
            (item['frames_processed'] for item in bindings),
            default=len(frames),
        )
        return _ok({
            'bindings': bindings,
            'accounts': [
                {
                    'student_id': str(student.id),
                    'student_name': student.user.full_name,
                    'email': student.user.email,
                    'has_photo': bool(student.enrollment_face_photos()),
                    'sample_count': len(student.enrollment_face_photos()),
                }
                for student in students
            ],
            'missing_accounts': missing_accounts,
            'missing_photos': missing_photos,
            'frames_processed': frames_processed,
        })


class StationArtifactView(APIView):
    """POST /api/sessions/<session_id>/attribution/artifact/

    Receives the end-of-session summary from a physical exam station running
    the CV engine locally (exam-station-cv's BackendSink). This is the live
    counterpart of the post-hoc path: `cv_analysis` feeds the same artifact in
    directly when analysis runs in the cloud.

    Storing it in CVSessionReport means the examiner's existing CV summary
    view, question timeline and recording player all work for a physical
    session with no changes.
    """

    authentication_classes = [
        ExamStationAuthentication,
        PhysicalKioskAuthentication,
        CookieJWTAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_participant(request.user, session):
            return _err('You are not part of this session.', code=403)
        if not _is_trusted_device(request):
            return _err(
                'Only the physical kiosk or exam station may upload an artifact.',
                code=403,
            )
        if session.status not in {
            EvaluationSession.Status.IN_PROGRESS,
            EvaluationSession.Status.COMPLETED,
        }:
            return _err('The session has not started.', code=409)

        summary = request.data.get('summary')
        if not isinstance(summary, dict):
            return _err('summary object is required.')

        from cv_analysis.models import CVSessionReport

        report, _ = CVSessionReport.objects.get_or_create(session=session)
        report.artifact = summary
        report.status = CVSessionReport.Status.COMPLETED
        recording_path = request.data.get('recording_path')
        if recording_path and not report.recording_url:
            report.recording_url = str(recording_path)
        report.save(update_fields=[
            'artifact', 'status', 'recording_url', 'updated_at',
        ])

        # The artifact is authoritative over anything streamed live, so fold
        # it in and re-resolve — without disturbing answers an examiner has
        # already confirmed.
        ingested = ingest.ingest_posthoc_artifact(session, summary)
        stats = engine.reconcile_session(session) if ingested else {}
        if stats.get('changed'):
            from viva_evaluator.services.session_reports import (
                refresh_draft_summary_reports,
            )
            stats['reports_refreshed'] = refresh_draft_summary_reports(session)

        return _ok({
            'evidence_ingested': ingested,
            'reconcile': stats,
        }, 'Artifact stored.')


class AttributionReviewView(APIView):
    """GET /api/sessions/<session_id>/attribution/answers/

    The examiner's review queue: every answer, who it was attributed to, how
    confident that was, and which answers still need a human decision.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_examiner_for(request.user, session):
            return _err('You are not assigned to this project.', code=403)

        rows = (
            AnswerAttribution.objects
            .filter(session=session)
            .select_related(
                'answer__question', 'student__user', 'provisional_student__user',
            )
            .order_by('answer__question__question_order')
        )

        items = [{
            'attribution_id': str(r.id),
            'answer_id': str(r.answer_id),
            'question_order': r.answer.question.question_order,
            'question_text': r.answer.question.question_text,
            'answer_text': r.answer.transcribed_answer,
            'student_id': str(r.student_id) if r.student_id else None,
            'student_name': _name(r.student),
            'provisional_student_id': (
                str(r.provisional_student_id) if r.provisional_student_id else None
            ),
            'provisional_student_name': _name(r.provisional_student),
            'share': r.share,
            'margin': r.margin,
            'outcome': r.outcome,
            'co_speakers': r.co_speakers,
            'source_breakdown': r.source_breakdown,
            'status': r.status,
            'needs_review': r.needs_review,
        } for r in rows]

        return _ok({
            'items': items,
            'needs_review_count': sum(1 for i in items if i['needs_review']),
            'roster': _roster(session),
        })


class UnknownSpeakerView(APIView):
    """GET/POST /api/sessions/<session_id>/attribution/unknown-speakers/

    GET  lists people the CV could follow but not name, with how much of the
         session's answers each is holding.
    POST {"unknown_speaker_id": "...", "student_id": "..."} names one, moving
         their held marks onto that student.

    This exists so a student who never uploaded an enrolment photo is not
    quietly written out of their own viva: their answers are kept together
    under one label and handed over intact once someone identifies them.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_examiner_for(request.user, session):
            return _err('You are not assigned to this project.', code=403)

        totals = engine.student_contribution_totals(session)
        held = {u['id']: u for u in totals['unknown']}

        rows = (
            UnknownSpeaker.objects
            .filter(session=session)
            .select_related('resolved_student__user')
        )
        items = []
        for row in rows:
            stats = held.get(str(row.id), {'answers': 0, 'total_share': 0.0})
            items.append({
                'unknown_speaker_id': str(row.id),
                'label': row.label,
                'answers_contributed': stats['answers'],
                'total_share': stats['total_share'],
                'first_seen': row.first_seen,
                'last_seen': row.last_seen,
                'resolved_student_id': (
                    str(row.resolved_student_id) if row.resolved_student_id else None
                ),
                'resolved_student_name': _name(row.resolved_student),
            })

        return _ok({
            'items': items,
            'unresolved_count': sum(
                1 for i in items if not i['resolved_student_id']
            ),
            'roster': _roster(session),
        })

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_examiner_for(request.user, session):
            return _err('You are not assigned to this project.', code=403)

        unknown = UnknownSpeaker.objects.filter(
            session=session, id=request.data.get('unknown_speaker_id'),
        ).first()
        if not unknown:
            return _err('Unknown speaker not found in this session.', code=404)

        student_id = request.data.get('student_id')
        if not student_id:
            return _err('student_id is required.')

        examiner = ExaminerProfile.objects.filter(user=request.user).first()
        try:
            engine.resolve_unknown_speaker(unknown, examiner, student_id)
        except ValueError as e:
            return _err(str(e))
        except Exception:
            logger.exception('Resolving unknown speaker failed for %s', session_id)
            return _err('Could not resolve that speaker.', code=500)
        try:
            from viva_evaluator.services.session_reports import (
                refresh_draft_summary_reports,
            )
            refresh_draft_summary_reports(session)
        except Exception:
            logger.exception(
                'Report refresh failed after resolving unknown speaker for %s',
                session_id,
            )

        return _ok({
            'unknown_speaker_id': str(unknown.id),
            'label': unknown.label,
            'resolved_student_id': str(unknown.resolved_student_id),
        }, f'{unknown.label} identified — their marks now count for that student.')


class AttributionConfirmView(APIView):
    """POST /api/sessions/<session_id>/attribution/answers/<answer_id>/confirm/

    Body: {"student_id": "<uuid>"} to override, or {} to confirm as resolved.
    The examiner's decision is final — it is never revisited by reconciliation.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request, session_id, answer_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_examiner_for(request.user, session):
            return _err('You are not assigned to this project.', code=403)

        attribution = (
            AnswerAttribution.objects
            .filter(session=session, answer_id=answer_id)
            .select_related('answer')
            .first()
        )
        if not attribution:
            return _err('No attribution exists for that answer.', code=404)

        examiner = ExaminerProfile.objects.filter(user=request.user).first()
        student_id = request.data.get('student_id')

        try:
            engine.confirm_attribution(attribution, examiner, student_id)
        except ValueError as e:
            return _err(str(e))
        except Exception:
            logger.exception('Confirm failed for answer %s', answer_id)
            return _err('Could not confirm the attribution.', code=500)
        try:
            from viva_evaluator.services.session_reports import (
                refresh_draft_summary_reports,
            )
            refresh_draft_summary_reports(session)
        except Exception:
            logger.exception(
                'Report refresh failed after confirming answer %s', answer_id,
            )

        return _ok({
            'answer_id': str(attribution.answer_id),
            'student_id': (
                str(attribution.student_id) if attribution.student_id else None
            ),
            'status': attribution.status,
        }, 'Attribution confirmed.')


class AttributionReconcileView(APIView):
    """POST /api/sessions/<session_id>/attribution/reconcile/

    Re-resolves every answer now that post-hoc CV evidence exists. Answers the
    examiner already confirmed are left alone; a disagreement on one of those
    is marked disputed for review rather than re-scored.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        session = _get_session(session_id)
        if not session:
            return _err('Session not found.', code=404)
        if not _is_examiner_for(request.user, session):
            return _err('You are not assigned to this project.', code=403)

        report = getattr(session, 'cv_report', None)
        ingested = 0
        if report is not None and report.artifact:
            ingested = ingest.ingest_posthoc_artifact(session, report.artifact)

        stats = engine.reconcile_session(session)
        if stats.get('changed'):
            from viva_evaluator.services.session_reports import (
                refresh_draft_summary_reports,
            )
            stats['reports_refreshed'] = refresh_draft_summary_reports(session)
        stats['posthoc_evidence_ingested'] = ingested
        return _ok(stats, 'Reconciliation complete.')


def _name(profile):
    """Display name. The custom User model carries full_name/email, not the
    Django default first_name/last_name/username."""
    if profile is None:
        return None
    user = profile.user
    full = (user.full_name or '').strip()
    if full and full.lower() != 'none':
        return full
    return user.email


def _roster(session):
    if session.group_id:
        members = (
            GroupMember.objects
            .filter(group_id=session.group_id)
            .select_related('student__user')
        )
        return [
            {'student_id': str(m.student_id), 'name': _name(m.student)}
            for m in members
        ]
    if session.student_id:
        return [{'student_id': str(session.student_id), 'name': _name(session.student)}]
    return []
