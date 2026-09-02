from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('physical_evaluation', '0004_nullable_recording_start'),
    ]

    operations = [
        migrations.AlterField(
            model_name='physicalevaluationrun',
            name='identity_status',
            field=models.CharField(
                choices=[
                    ('pending', 'Pending identity review'),
                    ('verified', 'All expected members verified'),
                    ('partial', 'Present group members verified'),
                    ('overridden', 'Examiner PIN override'),
                    ('not_required', 'Individual session'),
                ],
                default='pending',
                max_length=20,
            ),
        ),
    ]
