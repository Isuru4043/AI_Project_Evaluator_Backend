from django.db import migrations, models
import django.db.models.deletion


def initialize_identity_status(apps, schema_editor):
    Run = apps.get_model('physical_evaluation', 'PhysicalEvaluationRun')
    for run in Run.objects.select_related('session').iterator():
        run.identity_status = 'pending' if run.session.group_id else 'not_required'
        run.save(update_fields=['identity_status'])


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
        ('physical_evaluation', '0002_chunked_recording_upload'),
    ]

    operations = [
        migrations.AddField(
            model_name='physicalevaluationrun',
            name='identity_status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending identity review'),
                    ('verified', 'All expected members verified'),
                    ('overridden', 'Examiner PIN override'),
                    ('not_required', 'Individual session'),
                ],
                default='pending',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='physicalevaluationrun',
            name='identity_verification',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='physicalevaluationrun',
            name='identity_verified_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='physicalevaluationrun',
            name='identity_override_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='physicalevaluationrun',
            name='identity_override_reason',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='physicalevaluationrun',
            name='identity_override_by',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name='physical_identity_overrides',
                to='core.examinerprofile',
            ),
        ),
        migrations.RunPython(initialize_identity_status, migrations.RunPython.noop),
    ]
