"""
Views for Rubric Management (Feature 4).
"""

from django.db.models import Sum
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from core.models import Project, RubricCategory, RubricCriteria
from projects.permissions import IsExaminer, IsExaminerOrStudent
from projects.serializers import (
    RubricCategoryCreateSerializer, RubricCategorySerializer,
    RubricCategoryUpdateSerializer, RubricCriteriaCreateSerializer,
    RubricCriteriaSerializer, RubricCriteriaUpdateSerializer,
)
from projects.views.project_views import _err, _get_examiner_profile, _is_assigned, _ok, _500


class RubricCategoryCreateView(APIView):
    """POST /api/projects/<project_id>/rubrics/categories/create/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            ser = RubricCategoryCreateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            current_weight = project.rubric_categories.aggregate(
                total=Sum('weight_percentage'),
            )['total'] or 0
            new_weight = current_weight + ser.validated_data['weight_percentage']
            if new_weight > 100:
                return _err(
                    f'Total weight exceeds 100%. Currently used: {current_weight}%'
                )

            cat = RubricCategory.objects.create(
                project=project,
                category_name=ser.validated_data['category_name'],
                weight_percentage=ser.validated_data['weight_percentage'],
                description=ser.validated_data.get('description'),
            )
            return _ok('Rubric category created.', RubricCategorySerializer(cat).data, 201)
        except Exception as e:
            return _500(e)


class RubricCategoryUpdateView(APIView):
    """PUT /api/projects/<project_id>/rubrics/categories/<category_id>/update/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def put(self, request, project_id, category_id):
        try:
            cat = RubricCategory.objects.filter(id=category_id, project_id=project_id).first()
            if not cat:
                return _err('Rubric category not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, cat.project):
                return _err('You are not assigned to this project.', code=403)

            ser = RubricCategoryUpdateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            data = ser.validated_data

            if 'weight_percentage' in data:
                other_weight = cat.project.rubric_categories.exclude(
                    id=cat.id,
                ).aggregate(total=Sum('weight_percentage'))['total'] or 0
                if other_weight + data['weight_percentage'] > 100:
                    return _err(
                        f'Total weight exceeds 100%. Other categories use: {other_weight}%'
                    )

            for field in ('category_name', 'weight_percentage', 'description'):
                if field in data:
                    setattr(cat, field, data[field])
            cat.save()

            return _ok('Rubric category updated.', RubricCategorySerializer(cat).data)
        except Exception as e:
            return _500(e)


class RubricCategoryDeleteView(APIView):
    """DELETE /api/projects/<project_id>/rubrics/categories/<category_id>/delete/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def delete(self, request, project_id, category_id):
        try:
            cat = RubricCategory.objects.filter(id=category_id, project_id=project_id).first()
            if not cat:
                return _err('Rubric category not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, cat.project):
                return _err('You are not assigned to this project.', code=403)

            cat.delete()
            return _ok('Rubric category deleted.')
        except Exception as e:
            return _500(e)


class RubricCriteriaCreateView(APIView):
    """POST /api/projects/<project_id>/rubrics/categories/<category_id>/criteria/create/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, project_id, category_id):
        try:
            cat = RubricCategory.objects.filter(id=category_id, project_id=project_id).first()
            if not cat:
                return _err('Rubric category not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, cat.project):
                return _err('You are not assigned to this project.', code=403)

            ser = RubricCriteriaCreateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            criteria = RubricCriteria.objects.create(
                category=cat,
                criteria_name=ser.validated_data['criteria_name'],
                max_score=ser.validated_data['max_score'],
                weight_in_category=ser.validated_data.get('weight_in_category'),
                description=ser.validated_data.get('description'),
            )
            return _ok('Rubric criteria created.', RubricCriteriaSerializer(criteria).data, 201)
        except Exception as e:
            return _500(e)


class RubricCriteriaUpdateView(APIView):
    """PUT .../criteria/<criteria_id>/update/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def put(self, request, project_id, category_id, criteria_id):
        try:
            criteria = RubricCriteria.objects.filter(
                id=criteria_id, category_id=category_id, category__project_id=project_id,
            ).first()
            if not criteria:
                return _err('Rubric criteria not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, criteria.category.project):
                return _err('You are not assigned to this project.', code=403)

            ser = RubricCriteriaUpdateSerializer(data=request.data)
            if not ser.is_valid():
                return _err('Validation failed.', ser.errors)

            for field in ('criteria_name', 'max_score', 'weight_in_category', 'description'):
                if field in ser.validated_data:
                    setattr(criteria, field, ser.validated_data[field])
            criteria.save()

            return _ok('Rubric criteria updated.', RubricCriteriaSerializer(criteria).data)
        except Exception as e:
            return _500(e)


class RubricCriteriaDeleteView(APIView):
    """DELETE .../criteria/<criteria_id>/delete/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def delete(self, request, project_id, category_id, criteria_id):
        try:
            criteria = RubricCriteria.objects.filter(
                id=criteria_id, category_id=category_id, category__project_id=project_id,
            ).first()
            if not criteria:
                return _err('Rubric criteria not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, criteria.category.project):
                return _err('You are not assigned to this project.', code=403)

            criteria.delete()
            return _ok('Rubric criteria deleted.')
        except Exception as e:
            return _500(e)


class RubricListView(APIView):
    """GET /api/projects/<project_id>/rubrics/"""
    permission_classes = [IsAuthenticated, IsExaminerOrStudent]

    def get(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            categories = project.rubric_categories.prefetch_related('criteria').all()
            return _ok('Rubrics retrieved.', RubricCategorySerializer(categories, many=True).data)
        except Exception as e:
            return _500(e)


class RubricExtractApplyView(APIView):
    """POST /api/projects/<project_id>/rubrics/extract/

    Examiner uploads a rubric document (.pdf/.docx/.md/.txt); Gemini extracts
    the structure and the categories + criteria are created directly on this
    project. The examiner reviews/edits afterwards with the normal rubric
    CRUD — this replaces typing every category and criterion by hand.
    """
    permission_classes = [IsAuthenticated, IsExaminer]

    MAX_SIZE = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTS = ('pdf', 'docx', 'md', 'markdown', 'txt')

    def post(self, request, project_id):
        try:
            project = Project.objects.filter(id=project_id).first()
            if not project:
                return _err('Project not found.', code=404)

            ep = _get_examiner_profile(request.user)
            if not _is_assigned(ep, project):
                return _err('You are not assigned to this project.', code=403)

            rubric_file = request.FILES.get('rubric_file')
            if not rubric_file:
                return _err('rubric_file is required.')
            ext = rubric_file.name.rsplit('.', 1)[-1].lower()
            if ext not in self.ALLOWED_EXTS:
                return _err('Only PDF, DOCX, MD and TXT files are accepted.')
            if rubric_file.size > self.MAX_SIZE:
                return _err('File too large. Maximum size is 10MB.')

            import os
            import tempfile
            from core.utils.document_parser import extract_text_from_file
            from viva_evaluator.services.rubric_extractor import (
                extract_rubric_from_text,
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
                for chunk in rubric_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            try:
                rubric_text = extract_text_from_file(tmp_path)
            finally:
                os.unlink(tmp_path)

            if not rubric_text.strip():
                return _err('Could not extract any text from the uploaded file.')

            from viva_evaluator.services.llm_service import LLMQuotaError
            try:
                extracted = extract_rubric_from_text(rubric_text)
            except LLMQuotaError:
                return _err(
                    'The AI service has hit its quota limit. Please try again '
                    'in a few minutes.',
                    code=503,
                )
            except RuntimeError:
                # llm_call exhausted its retries — provider is down or
                # overloaded. Nothing the examiner did wrong.
                return _err(
                    'The AI service is temporarily unavailable. Please try '
                    'uploading again in a moment.',
                    code=503,
                )

            if 'error' in extracted:
                return _err(f"Extraction failed: {extracted['error']}", code=500)

            categories_data = extracted.get('rubric_categories') or []
            if not categories_data:
                return _err('No rubric categories were found in the document.')

            from viva_evaluator.models import CriteriaQuestionHint

            created = []
            for cat_data in categories_data:
                category = RubricCategory.objects.create(
                    project=project,
                    category_name=str(cat_data.get('category_name', 'Untitled'))[:255],
                    weight_percentage=cat_data.get('weight_percentage') or 0,
                    description=cat_data.get('description') or '',
                )
                for cri_data in (cat_data.get('criteria') or []):
                    criteria = RubricCriteria.objects.create(
                        category=category,
                        criteria_name=str(cri_data.get('criteria_name', 'Untitled'))[:255],
                        max_score=cri_data.get('max_score') or 10,
                        weight_in_category=cri_data.get('weight_in_category'),
                        description=cri_data.get('description') or '',
                        questions_to_ask=int(cri_data.get('questions_to_ask') or 3),
                    )
                    for hint in (cri_data.get('question_hints') or []):
                        hint_text = (hint or {}).get('hint_text', '')
                        if hint_text:
                            CriteriaQuestionHint.objects.create(
                                criteria=criteria,
                                hint_text=hint_text,
                                order=int((hint or {}).get('order') or 1),
                            )
                created.append(category)

            return _ok(
                f'Extracted {len(created)} categories from "{rubric_file.name}". '
                'Review and edit them below.',
                RubricCategorySerializer(created, many=True).data,
                201,
            )
        except Exception as e:
            return _500(e)
