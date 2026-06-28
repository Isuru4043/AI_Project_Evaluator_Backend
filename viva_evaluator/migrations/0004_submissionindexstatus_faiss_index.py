# Generated for RAG Phase 1 implementation.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viva_evaluator', '0003_criteriaquestionhint'),
    ]

    operations = [
        migrations.AddField(
            model_name='submissionindexstatus',
            name='faiss_index_blob',
            field=models.BinaryField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name='submissionindexstatus',
            name='faiss_chunks_json',
            field=models.JSONField(null=True, blank=True),
        ),
    ]
