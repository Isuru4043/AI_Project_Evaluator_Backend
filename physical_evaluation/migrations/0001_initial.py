import django.db.models.deletion
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('core', '0028_project_evaluation_mode'),
    ]

    operations = [
        migrations.CreateModel(
            name='PhysicalProjectConfig',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('location', models.CharField(max_length=255)),
                ('panel_pin_hash', models.CharField(max_length=128)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('created_by', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='physical_project_configs', to='core.examinerprofile')),
                ('project', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='physical_config', to='core.project')),
            ],
        ),
        migrations.CreateModel(
            name='PhysicalKioskAccess',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('token_digest', models.CharField(db_index=True, max_length=64, unique=True)),
                ('expires_at', models.DateTimeField()),
                ('closed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('last_activity_at', models.DateTimeField(auto_now_add=True)),
                ('config', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='kiosk_accesses', to='physical_evaluation.physicalprojectconfig')),
                ('opened_by', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='opened_physical_kiosks', to='core.examinerprofile')),
            ],
        ),
        migrations.CreateModel(
            name='PhysicalEvaluationRun',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('demo_in_progress', 'Demo in progress'), ('viva_in_progress', 'Viva in progress'), ('completed', 'Completed')], max_length=30)),
                ('recording_started_at', models.DateTimeField()),
                ('viva_started_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('kiosk_access', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='runs', to='physical_evaluation.physicalkioskaccess')),
                ('recording', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='physical_run', to='core.sessionrecording')),
                ('session', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='physical_run', to='core.evaluationsession')),
            ],
        ),
    ]
