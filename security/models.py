from django.db import models
from django.utils import timezone

class Camera(models.Model):
    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
    ]
    name = models.CharField(max_length=100, unique=True)
    stream_url = models.CharField(max_length=255, default="0")
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='offline')
    last_seen_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.get_status_display()})"

class DetectionLog(models.Model):
    track_id = models.IntegerField()
    confidence = models.FloatField()
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='detections')
    detected_at = models.DateTimeField(default=timezone.now)
    screenshot_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-detected_at']

    def __str__(self):
        return f"Track {self.track_id} on {self.camera.name} at {self.detected_at}"
