from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viva_evaluator', '0008_vivaanswerextension_detailed_ai_analysis'),
    ]

    operations = [
        migrations.AddField(
            model_name='vivaquestionextension',
            name='fallback_used',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='vivaquestionextension',
            name='generation_audit',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='vivaquestionextension',
            name='validation_degraded',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='vivaquestionextension',
            name='validation_status',
            field=models.CharField(default='not_applicable', max_length=32),
        ),
    ]
