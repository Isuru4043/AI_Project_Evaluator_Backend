"""Physiological signals from the exam-station BLE band.

A finger-clip PPG sensor (MAX30102 on an ESP32) streams heart rate and
inter-beat intervals over the standard Bluetooth Heart Rate Service. A sidecar
on the exam station relays them here, stamped on the session clock, so the
examiner can see when a student's arousal rose and against which question.

PHYSICAL SESSIONS ONLY. There is no device in a remote viva.

INVARIANTS (carried from exam-station-cv's contract):
- Physiological signals are ADVISORY. Nothing here may enter a score, a grade,
  or any fusion that produces one. An arousal marker is a pointer telling the
  examiner where to look, exactly like a gaze flag.
- Arousal is not deception. Elevated heart rate means a student found a moment
  demanding; it does not mean they were dishonest, and nothing in this app may
  present it as though it did.
- Absence of data is not calm. With one device in a group session only the
  wearer is measured, and unmeasured students must never be rendered as
  though they were serene.

Everything is expressed RELATIVE TO THE STUDENT'S OWN BASELINE, captured
during a dedicated calm period. Resting heart rate varies enormously between
people (60-100 bpm is all normal), so an absolute threshold would flag the
naturally fast-hearted and miss everyone else.
"""

import uuid

from django.db import models

from core.models import EvaluationSession, StudentProfile


class PhysioDevice(models.Model):
    """The band, and who is wearing it for this session.

    One device per session in the current hardware. The binding is what makes
    a sample attributable: without it the stream is anonymous telemetry.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='physio_devices',
    )
    # BLE identity as advertised (name or MAC suffix), stable across a session.
    device_id = models.CharField(max_length=64)
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.CASCADE,
        related_name='physio_devices',
    )
    bound_at = models.DateTimeField(auto_now_add=True)
    # Set when the band is moved to someone else, so historical samples keep
    # the binding that was true when they were captured.
    unbound_at = models.DateTimeField(null=True, blank=True)
    battery_pct = models.IntegerField(null=True, blank=True)

    class Meta:
        verbose_name = 'Physio Device'
        verbose_name_plural = 'Physio Devices'
        ordering = ['-bound_at']
        indexes = [models.Index(fields=['session', 'unbound_at'])]

    def __str__(self):
        return f'{self.device_id} -> {self.student_id}'


class PhysioSample(models.Model):
    """One notification from the band: a heart rate plus the beats behind it.

    `ibi_ms` is the list of inter-beat intervals accumulated since the previous
    notification. It is the payload that matters: HRV is beat-to-beat
    variability, so it can only be computed from the intervals themselves. A
    smoothed BPM has had exactly that information filtered out of it.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='physio_samples',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.CASCADE,
        related_name='physio_samples',
    )
    device = models.ForeignKey(
        PhysioDevice,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='samples',
    )
    # Station wall-clock at receipt. The band has no RTC; BLE notification
    # latency (tens of ms) is irrelevant against 30 s analysis windows.
    t = models.DateTimeField()
    bpm = models.IntegerField(null=True, blank=True)
    ibi_ms = models.JSONField(default=list, blank=True)
    # Sensor Contact bit from the Heart Rate Measurement flags. False means the
    # clip was off the finger: the window is unusable, not calm.
    contact = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Physio Sample'
        verbose_name_plural = 'Physio Samples'
        ordering = ['t']
        indexes = [models.Index(fields=['session', 'student', 't'])]
        constraints = [
            # The sidecar retries on network failure; a replayed batch must not
            # double-count beats into the variability maths.
            models.UniqueConstraint(
                fields=['session', 'student', 't'],
                name='uniq_physio_sample_instant',
            ),
        ]

    def __str__(self):
        return f'{self.bpm} bpm @ {self.t}'


class BaselineWindow(models.Model):
    """The calm period a student's own resting values are measured over.

    Deliberately a dedicated window rather than "the demo phase": presenting
    is itself arousing, so a baseline taken during it would be inflated and
    would mask exactly the rises worth seeing.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='physio_baseline_windows',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.CASCADE,
        related_name='physio_baseline_windows',
    )
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)

    # Derived once the window closes. Null means it could not be computed -
    # too few clean beats - and downstream must then report "no baseline"
    # rather than comparing against zero.
    hr_mean = models.FloatField(null=True, blank=True)
    hr_sd = models.FloatField(null=True, blank=True)
    rmssd = models.FloatField(null=True, blank=True)
    sdnn = models.FloatField(null=True, blank=True)
    beat_count = models.IntegerField(default=0)
    # Share of beats that survived artifact rejection, 0..1.
    quality = models.FloatField(default=0.0)
    computed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Baseline Window'
        verbose_name_plural = 'Baseline Windows'
        ordering = ['-started_at']

    def __str__(self):
        state = 'open' if self.ended_at is None else f'{self.beat_count} beats'
        return f'baseline {self.student_id} ({state})'

    @property
    def is_usable(self) -> bool:
        """Whether this baseline can support a comparison at all.

        The floor tracks the length of the calm window (see
        ``MIN_BASELINE_BEATS``); it used to be hard-coded to the 30 s analysis
        figure, which a short calm period can never reach, so shortening the
        window alone would have left every baseline unusable.
        """
        from physiology.services.metrics import MIN_BASELINE_BEATS

        return (
            self.hr_mean is not None
            and self.rmssd is not None
            and self.beat_count >= MIN_BASELINE_BEATS
            and self.quality >= 0.5
        )
