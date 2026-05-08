"""
URL configuration for AI_Evaluator_Backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
"""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),

    # Authentication endpoints — /api/auth/...
    path('api/auth/', include('authentication.urls')),
]
