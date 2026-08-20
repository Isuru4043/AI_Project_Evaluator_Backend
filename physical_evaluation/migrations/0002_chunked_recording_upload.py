import django.db.models.deletion
import uuid
from django.db import migrations, models


def release_completed_legacy_runs(apps, schema_editor):
    PhysicalEvaluationRun = apps.get_model('physical_evaluation', 'PhysicalEvaluationRun')
    PhysicalEvaluationRun.objects.filter(
        status='viva_in_progress',
        session__status='completed',
        recording__isnull=True,
    ).update(status='recording_failed')


class Migration(migrations.Migration):

    dependencies = [
        ('physical_evaluation', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='physicalevaluationrun',
            name='status',
            field=models.CharField(
                choices=[
                    ('demo_in_progress', 'Demo in progress'),
                    ('viva_in_progress', 'Viva in progress'),
                    ('recording_uploading', 'Recording uploading'),
                    ('recording_failed', 'Recording failed'),
                    ('completed', 'Completed'),
                ],
                max_length=30,
            ),
        ),
        migrations.CreateModel(
            name='PhysicalRecordingUpload',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('capturing', 'Capturing'), ('uploading', 'Uploading'), ('finalizing', 'Finalizing'), ('ready', 'Ready'), ('failed', 'Failed')], default='capturing', max_length=20)),
                ('blob_path', models.CharField(blank=True, default='', max_length=512)),
                ('mime_type', models.CharField(blank=True, default='video/webm', max_length=100)),
                ('expected_chunks', models.PositiveIntegerField(blank=True, null=True)),
                ('uploaded_chunk_indices', models.JSONField(blank=True, default=list)),
                ('duration_seconds', models.PositiveIntegerField(blank=True, null=True)),
                ('finalization_requested', models.BooleanField(default=False)),
                ('error_message', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('finalized_at', models.DateTimeField(blank=True, null=True)),
                ('run', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='recording_upload', to='physical_evaluation.physicalevaluationrun')),
            ],
        ),
        migrations.RunPython(release_completed_legacy_runs, migrations.RunPython.noop),
    ]
