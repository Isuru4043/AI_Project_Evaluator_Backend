from django.db import migrations, models


def repair_legacy_group_criteria(apps, schema_editor):
    """Old clients could not set scoring scope, so every row became shared."""
    RubricCriteria = apps.get_model('core', 'RubricCriteria')
    RubricCriteria.objects.filter(
        category__project__is_group_project=True,
        is_individual=False,
    ).update(is_individual=True)


class Migration(migrations.Migration):
    dependencies = [
        ('core', '0030_studentprofile_face_photo_urls'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rubriccriteria',
            name='is_individual',
            field=models.BooleanField(default=True),
        ),
        migrations.RunPython(
            repair_legacy_group_criteria,
            migrations.RunPython.noop,
        ),
    ]
