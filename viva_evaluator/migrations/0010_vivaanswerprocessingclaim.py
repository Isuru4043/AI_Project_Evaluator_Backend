import django.db.models.deletion
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0029_vivaanswer_deduplication_key'),
        ('viva_evaluator', '0009_vivaquestionextension_validation_audit'),
    ]

    operations = [
        migrations.CreateModel(
            name='VivaAnswerProcessingClaim',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('speaker_key', models.CharField(max_length=100)),
                ('idempotency_key', models.CharField(max_length=160)),
                ('request_hash', models.CharField(max_length=64)),
                ('status', models.CharField(choices=[('processing', 'Processing'), ('completed', 'Completed'), ('failed', 'Failed')], default='processing', max_length=20)),
                ('owner_token', models.UUIDField(default=uuid.uuid4)),
                ('lease_expires_at', models.DateTimeField()),
                ('response_payload', models.JSONField(blank=True, null=True)),
                ('response_status', models.PositiveSmallIntegerField(default=200)),
                ('error_code', models.CharField(blank=True, default='', max_length=100)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('question', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='processing_claims', to='core.vivaquestion')),
                ('session', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='answer_processing_claims', to='core.evaluationsession')),
            ],
        ),
        migrations.AddConstraint(
            model_name='vivaanswerprocessingclaim',
            constraint=models.UniqueConstraint(fields=('question', 'speaker_key'), name='uniq_answer_claim_question_speaker'),
        ),
        migrations.AddConstraint(
            model_name='vivaanswerprocessingclaim',
            constraint=models.UniqueConstraint(fields=('session', 'idempotency_key'), name='uniq_answer_claim_session_key'),
        ),
        migrations.AddIndex(
            model_name='vivaanswerprocessingclaim',
            index=models.Index(fields=['status', 'lease_expires_at'], name='viva_evalua_status_452dc9_idx'),
        ),
    ]
