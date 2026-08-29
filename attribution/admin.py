from django.contrib import admin

from attribution.models import AnswerAttribution, SpeakerBinding, SpeakerEvidence


@admin.register(SpeakerBinding)
class SpeakerBindingAdmin(admin.ModelAdmin):
    list_display = ('session', 'student', 'track_ref', 'method', 'confidence',
                    'bound_at', 'superseded_at')
    list_filter = ('method', 'superseded_at')
    search_fields = ('session__id', 'student__id')


@admin.register(SpeakerEvidence)
class SpeakerEvidenceAdmin(admin.ModelAdmin):
    list_display = ('session', 'source', 'student', 't_start', 't_end', 'confidence')
    list_filter = ('source',)
    search_fields = ('session__id', 'student__id')
    date_hierarchy = 't_start'


@admin.register(AnswerAttribution)
class AnswerAttributionAdmin(admin.ModelAdmin):
    list_display = ('answer', 'student', 'outcome', 'share', 'margin', 'status')
    list_filter = ('status', 'outcome')
    search_fields = ('session__id', 'answer__id')
    readonly_fields = ('source_breakdown', 'co_speakers')
