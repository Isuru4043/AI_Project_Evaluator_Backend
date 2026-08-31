from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('physical_evaluation', '0003_identity_review'),
    ]

    operations = [
        migrations.AlterField(
            model_name='physicalevaluationrun',
            name='recording_started_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
