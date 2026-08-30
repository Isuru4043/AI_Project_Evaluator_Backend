from django.contrib import admin

from physiology.models import BaselineWindow, PhysioDevice, PhysioSample


@admin.register(PhysioDevice)
class PhysioDeviceAdmin(admin.ModelAdmin):
    list_display = ('device_id', 'session', 'student', 'battery_pct',
                    'bound_at', 'unbound_at')
    search_fields = ('device_id', 'session__id', 'student__id')


@admin.register(PhysioSample)
class PhysioSampleAdmin(admin.ModelAdmin):
    list_display = ('t', 'session', 'student', 'bpm', 'contact')
    list_filter = ('contact',)
    date_hierarchy = 't'


@admin.register(BaselineWindow)
class BaselineWindowAdmin(admin.ModelAdmin):
    list_display = ('student', 'session', 'started_at', 'ended_at',
                    'hr_mean', 'rmssd', 'beat_count', 'quality')
    readonly_fields = ('computed_at',)
