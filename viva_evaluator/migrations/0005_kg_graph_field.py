# Generated for Week 3 — Knowledge Graph persistence.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viva_evaluator', '0004_submissionindexstatus_faiss_index'),
    ]

    operations = [
        migrations.AddField(
            model_name='submissionindexstatus',
            name='kg_graph_json',
            field=models.JSONField(null=True, blank=True),
        ),
    ]
