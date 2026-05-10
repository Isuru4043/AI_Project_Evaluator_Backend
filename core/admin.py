from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import (
	CodeSubmission,
	ExaminerProfile,
	GeneratedVivaQuestion,
	Project,
	ProjectSubmission,
	StudentProfile,
	User,
)


@admin.register(User)
class CustomUserAdmin(UserAdmin):
	ordering = ("-created_at",)
	list_display = ("email", "full_name", "role", "is_staff", "is_active")
	search_fields = ("email", "full_name")
	fieldsets = (
		(None, {"fields": ("email", "password")}),
		("Personal info", {"fields": ("full_name", "role", "profile_picture_url")}),
		("Permissions", {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
		("Important dates", {"fields": ("last_login",)}),
	)
	add_fieldsets = (
		(
			None,
			{
				"classes": ("wide",),
				"fields": ("email", "full_name", "role", "password1", "password2"),
			},
		),
	)
	filter_horizontal = ("groups", "user_permissions")


@admin.register(ExaminerProfile)
class ExaminerProfileAdmin(admin.ModelAdmin):
	list_display = ("user", "employee_id", "department", "designation")
	search_fields = ("user__email", "user__full_name", "employee_id")


@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
	list_display = ("user", "registration_number", "degree_program", "academic_year")
	search_fields = ("user__email", "user__full_name", "registration_number")


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
	list_display = ("project_name", "status", "academic_year", "created_at")
	search_fields = ("project_name",)


@admin.register(ProjectSubmission)
class ProjectSubmissionAdmin(admin.ModelAdmin):
	list_display = ("project", "student", "group", "submitted_at")
	search_fields = ("project__project_name",)


@admin.register(CodeSubmission)
class CodeSubmissionAdmin(admin.ModelAdmin):
	list_display = ("project_submission", "source_type", "analysis_status", "uploaded_at")
	search_fields = ("project_submission__project__project_name",)


@admin.register(GeneratedVivaQuestion)
class GeneratedVivaQuestionAdmin(admin.ModelAdmin):
	list_display = ("code_submission", "blooms_level", "source_type", "created_at")
	search_fields = ("question_text",)
