from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
        ('attribution', '0003_fix_speaker_evidence_deduplication'),
    ]

    operations = [
        migrations.CreateModel(
            name='FaceEnrollmentEmbeddingCache',
            fields=[
                ('photo_fingerprint', models.CharField(db_index=True, max_length=64)),
                ('embeddings', models.JSONField(blank=True, default=list)),
                ('engine_version', models.CharField(blank=True, default='', max_length=64)),
                ('status', models.CharField(
                    choices=[
                        ('pending', 'Pending'), ('ready', 'Ready'),
                        ('unusable', 'No usable enrollment'), ('failed', 'Failed'),
                    ],
                    default='pending', max_length=20,
                )),
                ('error_message', models.TextField(blank=True, default='')),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('student', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    primary_key=True,
                    related_name='face_embedding_cache',
                    serialize=False,
                    to='core.studentprofile',
                )),
            ],
        ),
    ]
