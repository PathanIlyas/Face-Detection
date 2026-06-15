import os
from datetime import datetime, timedelta
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Count, Q
from django.core.paginator import Paginator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

from .models import Camera, DetectionLog
from .serializers import CameraSerializer, DetectionLogSerializer

# --- HTML Views ---

@login_required
def dashboard_view(request):
    """
    Renders the central security monitoring dashboard.
    Optimizes queries to show counts, recent events, camera status cards, and the image gallery.
    """
    total_detections = DetectionLog.objects.count()
    
    # Today's detections
    today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_detections = DetectionLog.objects.filter(detected_at__gte=today_start).count()
    
    # Active persons (seen in the last 2 minutes)
    active_cutoff = timezone.now() - timedelta(minutes=2)
    active_persons = DetectionLog.objects.filter(detected_at__gte=active_cutoff).values('track_id').distinct().count()
    
    # Camera metrics
    cameras_online = Camera.objects.filter(status='online').count()
    total_cameras = Camera.objects.count()
    
    # Recent logs and photo gallery (last 10/8 items with pre-selected cameras)
    recent_detections = DetectionLog.objects.select_related('camera').order_by('-detected_at')[:10]
    gallery_detections = DetectionLog.objects.select_related('camera').exclude(screenshot_path='').order_by('-detected_at')[:8]
    cameras_list = Camera.objects.all()
    
    context = {
        'total_detections': total_detections,
        'today_detections': today_detections,
        'active_persons': active_persons,
        'cameras_online': cameras_online,
        'total_cameras': total_cameras,
        'recent_detections': recent_detections,
        'gallery_detections': gallery_detections,
        'cameras_list': cameras_list,
    }
    return render(request, 'security/dashboard.html', context)

@login_required
def logs_view(request):
    """
    Renders security event logs with search capability, pagination, and multi-filter criteria.
    """
    logs_list = DetectionLog.objects.select_related('camera').all()
    
    # Filter forms input
    search_query = request.GET.get('search', '').strip()
    camera_id = request.GET.get('camera', '').strip()
    track_id = request.GET.get('track_id', '').strip()
    date_query = request.GET.get('date', '').strip()
    
    if search_query:
        logs_list = logs_list.filter(
            Q(camera__name__icontains=search_query) |
            Q(track_id__icontains=search_query)
        )
        
    if camera_id:
        logs_list = logs_list.filter(camera_id=camera_id)
        
    if track_id:
        logs_list = logs_list.filter(track_id=track_id)
        
    if date_query:
        try:
            filter_date = datetime.strptime(date_query, "%Y-%m-%d").date()
            logs_list = logs_list.filter(detected_at__date=filter_date)
        except ValueError:
            pass

    # Pagination setup: 15 logs per page
    paginator = Paginator(logs_list, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    cameras = Camera.objects.all()
    
    context = {
        'page_obj': page_obj,
        'cameras': cameras,
        'selected_camera': camera_id,
        'selected_track': track_id,
        'selected_date': date_query,
        'search_query': search_query,
    }
    return render(request, 'security/logs.html', context)


# --- REST API View Layer ---

class DashboardAPIView(APIView):
    """
    Endpoint: GET /api/dashboard/
    Returns summary counters for security integrations.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        total_detections = DetectionLog.objects.count()
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_detections = DetectionLog.objects.filter(detected_at__gte=today_start).count()
        
        active_cutoff = timezone.now() - timedelta(minutes=2)
        active_persons = DetectionLog.objects.filter(detected_at__gte=active_cutoff).values('track_id').distinct().count()
        
        cameras_online = Camera.objects.filter(status='online').count()
        total_cameras = Camera.objects.count()
        
        return Response({
            'total_detections': total_detections,
            'today_detections': today_detections,
            'active_persons': active_persons,
            'cameras_online': cameras_online,
            'total_cameras': total_cameras,
        })


class DetectionLogAPIView(APIView):
    """
    Endpoint: GET /api/detections/
    Returns list of 50 recent detection logs.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        camera_id = request.query_params.get('camera')
        logs = DetectionLog.objects.select_related('camera').all()
        if camera_id:
            logs = logs.filter(camera_id=camera_id)
            
        logs = logs[:50]
        serializer = DetectionLogSerializer(logs, many=True)
        return Response(serializer.data)


class CameraAPIView(APIView):
    """
    Endpoint: GET /api/cameras/
    Returns the listing and operational state of registered cameras.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        cameras = Camera.objects.all()
        serializer = CameraSerializer(cameras, many=True)
        return Response(serializer.data)


class StatisticsAPIView(APIView):
    """
    Endpoint: GET /api/statistics/ (or /api/stats/)
    Returns analytics trends, weekly graphs, and hourly breakdown lists.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # 1. Total count grouped by camera
        detections_by_camera = Camera.objects.annotate(
            logs_count=Count('detections')
        ).values('name', 'logs_count')
        
        # 2. Weekly Trend
        today = timezone.now().date()
        weekly_trend = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            count = DetectionLog.objects.filter(detected_at__date=day).count()
            weekly_trend.append({
                'date': day.strftime("%Y-%m-%d"),
                'count': count
            })
            
        # 3. Hourly trend for today
        hourly_trend = []
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        for hour in range(24):
            start = today_start + timedelta(hours=hour)
            end = start + timedelta(hours=1)
            count = DetectionLog.objects.filter(detected_at__gte=start, detected_at__lt=end).count()
            hourly_trend.append({
                'hour': f"{hour:02d}:00",
                'count': count
            })
            
        return Response({
            'detections_by_camera': list(detections_by_camera),
            'weekly_trend': weekly_trend,
            'hourly_trend': hourly_trend,
        })
