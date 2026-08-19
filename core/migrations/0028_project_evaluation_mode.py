from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0027_evaluationsession_viva_weight_percentage'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='evaluation_mode',
            field=models.CharField(
                choices=[('remote', 'Remote'), ('physical', 'Physical')],
                default='remote',
                max_length=20,
            ),
        ),
    ]
