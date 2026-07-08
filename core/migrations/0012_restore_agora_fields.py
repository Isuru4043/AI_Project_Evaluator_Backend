"""Repair migration: restore the Agora columns on core_evaluationsession.

Migration 0011 added agora_channel_name / agora_stt_task_id /
transcript_file_url, but the shared database also carries drop migrations
(0012_drop_* .. 0014_drop_*) that were applied from a branch whose files
never landed in this repo — leaving the DB without columns the current
models require.

Uses ADD COLUMN IF NOT EXISTS so it is safe to run against any environment
regardless of whether the drops were applied there. No state operations:
Django's model state already contains these fields from 0011.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0011_evaluationsession_agora_channel_name_and_more'),
    ]

    operations = [
        migrations.RunSQL(
            sql=[
                "ALTER TABLE core_evaluationsession "
                "ADD COLUMN IF NOT EXISTS agora_channel_name varchar(128) NOT NULL DEFAULT ''",
                "ALTER TABLE core_evaluationsession "
                "ADD COLUMN IF NOT EXISTS agora_stt_task_id varchar(256) NOT NULL DEFAULT ''",
                "ALTER TABLE core_evaluationsession "
                "ADD COLUMN IF NOT EXISTS transcript_file_url text NOT NULL DEFAULT ''",
            ],
            reverse_sql=migrations.RunSQL.noop,
        ),
    ]
