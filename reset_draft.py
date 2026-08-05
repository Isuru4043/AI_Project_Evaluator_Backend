import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")
django.setup()

from core.models import SessionSummaryReport

for report in SessionSummaryReport.objects.all():
    report.scores_status = "draft"
    report.save()

print("All reports reset to draft.")
