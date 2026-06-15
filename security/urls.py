from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # HTML Pages
    path('', views.dashboard_view, name='dashboard'),
    path('logs/', views.logs_view, name='logs'),
    
    # Authentication Views
    path('login/', auth_views.LoginView.as_view(template_name='security/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    
    # REST API Layer
    path('api/dashboard/', views.DashboardAPIView.as_view(), name='api_dashboard'),
    path('api/detections/', views.DetectionLogAPIView.as_view(), name='api_detections'),
    path('api/cameras/', views.CameraAPIView.as_view(), name='api_cameras'),
    path('api/statistics/', views.StatisticsAPIView.as_view(), name='api_statistics'),
    path('api/stats/', views.StatisticsAPIView.as_view(), name='api_stats'),
]
