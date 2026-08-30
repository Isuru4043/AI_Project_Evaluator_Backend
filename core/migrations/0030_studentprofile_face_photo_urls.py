from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0029_vivaanswer_deduplication_key'),
    ]

    operations = [
        migrations.AddField(
            model_name='studentprofile',
            name='face_photo_urls',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
