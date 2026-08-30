from django.db import migrations, models


def remove_duplicate_evidence(apps, schema_editor):
    SpeakerEvidence = apps.get_model('attribution', 'SpeakerEvidence')

    groups = (
        SpeakerEvidence.objects
        .filter(student__isnull=False, unknown_speaker__isnull=True)
        .values('session', 'source', 'student', 't_start', 't_end')
        .annotate(row_count=models.Count('id'))
        .filter(row_count__gt=1)
    )
    for group in groups.iterator():
        filters = {key: group[key] for key in (
            'session', 'source', 'student', 't_start', 't_end',
        )}
        duplicate_ids = list(
            SpeakerEvidence.objects.filter(**filters)
            .order_by('created_at', 'id')
            .values_list('id', flat=True)
        )[1:]
        SpeakerEvidence.objects.filter(id__in=duplicate_ids).delete()

    groups = (
        SpeakerEvidence.objects
        .filter(unknown_speaker__isnull=False)
        .values('session', 'source', 'unknown_speaker', 't_start', 't_end')
        .annotate(row_count=models.Count('id'))
        .filter(row_count__gt=1)
    )
    for group in groups.iterator():
        filters = {key: group[key] for key in (
            'session', 'source', 'unknown_speaker', 't_start', 't_end',
        )}
        duplicate_ids = list(
            SpeakerEvidence.objects.filter(**filters)
            .order_by('created_at', 'id')
            .values_list('id', flat=True)
        )[1:]
        SpeakerEvidence.objects.filter(id__in=duplicate_ids).delete()

    groups = (
        SpeakerEvidence.objects
        .filter(student__isnull=True, unknown_speaker__isnull=True)
        .values('session', 'source', 't_start', 't_end')
        .annotate(row_count=models.Count('id'))
        .filter(row_count__gt=1)
    )
    for group in groups.iterator():
        filters = {key: group[key] for key in (
            'session', 'source', 't_start', 't_end',
        )}
        filters.update(student__isnull=True, unknown_speaker__isnull=True)
        duplicate_ids = list(
            SpeakerEvidence.objects.filter(**filters)
            .order_by('created_at', 'id')
            .values_list('id', flat=True)
        )[1:]
        SpeakerEvidence.objects.filter(id__in=duplicate_ids).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('attribution', '0002_answercontribution_unknownspeaker_and_more'),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name='speakerevidence',
            name='uniq_speaker_evidence_span',
        ),
        migrations.RunPython(
            remove_duplicate_evidence,
            migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name='speakerevidence',
            constraint=models.UniqueConstraint(
                fields=('session', 'source', 'student', 't_start', 't_end'),
                condition=models.Q(
                    student__isnull=False,
                    unknown_speaker__isnull=True,
                ),
                name='uniq_known_speaker_evidence_span',
            ),
        ),
        migrations.AddConstraint(
            model_name='speakerevidence',
            constraint=models.UniqueConstraint(
                fields=(
                    'session', 'source', 'unknown_speaker', 't_start', 't_end',
                ),
                condition=models.Q(unknown_speaker__isnull=False),
                name='uniq_unknown_speaker_evidence_span',
            ),
        ),
        migrations.AddConstraint(
            model_name='speakerevidence',
            constraint=models.UniqueConstraint(
                fields=('session', 'source', 't_start', 't_end'),
                condition=models.Q(
                    student__isnull=True,
                    unknown_speaker__isnull=True,
                ),
                name='uniq_unattributed_evidence_span',
            ),
        ),
    ]
