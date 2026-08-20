from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('core', '0028_project_evaluation_mode')]

    operations = [
        migrations.AddField(
            model_name='vivaanswer',
            name='deduplication_key',
            field=models.CharField(
                blank=True,
                help_text='Stable speaker key used to prevent duplicate answers.',
                max_length=100,
                null=True,
            ),
        ),
        migrations.AddConstraint(
            model_name='vivaanswer',
            constraint=models.UniqueConstraint(
                condition=models.Q(deduplication_key__isnull=False),
                fields=('question', 'deduplication_key'),
                name='uniq_viva_answer_question_speaker',
            ),
        ),
    ]
