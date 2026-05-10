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
