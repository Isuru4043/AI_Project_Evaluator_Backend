# Generated for Week 4 — examiner-in-the-loop knowledge accumulation.

import uuid
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0009_merge_20260513_0922'),
        ('viva_evaluator', '0005_kg_graph_field'),
    ]

    operations = [
        migrations.CreateModel(
            name='ApprovedDomainBrief',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('technology', models.CharField(max_length=200)),
                ('tech_version', models.CharField(blank=True, max_length=50, null=True)),
                ('brief_json', models.JSONField()),
                ('status', models.CharField(choices=[('pending', 'Pending Review'), ('active', 'Active'), ('archived', 'Archived')], default='pending', max_length=20)),
                ('scope', models.CharField(choices=[('examiner', 'Examiner Scope'), ('department', 'Department Scope')], default='examiner', max_length=20)),
                ('tier', models.IntegerField(default=2)),
                ('drafted_at', models.DateTimeField(auto_now_add=True)),
                ('approved_at', models.DateTimeField(blank=True, null=True)),
                ('last_verified_at', models.DateTimeField(blank=True, null=True)),
                ('approved_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='approved_briefs', to='core.examinerprofile')),
                ('drafted_for_submission', models.ForeignKey(blank=True, help_text='Submission whose tech extraction triggered this draft.', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='drafted_briefs', to='core.projectsubmission')),
            ],
            options={
                'verbose_name': 'Approved Domain Brief',
                'verbose_name_plural': 'Approved Domain Briefs',
                'ordering': ['-drafted_at'],
                'indexes': [
                    models.Index(fields=['technology', 'status'], name='viva_eval_a_technol_a59ddc_idx'),
                    models.Index(fields=['status', 'scope'], name='viva_eval_a_status_b5b6f1_idx'),
                ],
            },
        ),
    ]
