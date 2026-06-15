from rest_framework import serializers
from .models import Camera, DetectionLog

class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ['id', 'name', 'stream_url', 'status', 'last_seen_at', 'created_at']

class DetectionLogSerializer(serializers.ModelSerializer):
    camera_name = serializers.CharField(source='camera.name', read_only=True)
    
    class Meta:
        model = DetectionLog
        fields = ['id', 'track_id', 'confidence', 'camera', 'camera_name', 'detected_at', 'screenshot_path', 'created_at']
