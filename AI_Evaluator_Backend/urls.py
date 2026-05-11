"""
URL configuration for AI_Evaluator_Backend project.
"""

from django.contrib import admin
from django.urls import include, path

from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [

    path('admin/', admin.site.urls),

    # Authentication endpoints
    path('api/auth/', include('authentication.urls')),

    # Viva evaluator endpoints
    path('api/viva/', include('viva_evaluator.urls')),
    path('api/code-analysis/', include('code_analysis.urls')),

    # Projects endpoints — /api/projects/...
    path('api/projects/', include('projects.urls')),
]

# Media file serving
urlpatterns += static(
    settings.MEDIA_URL,
    document_root=settings.MEDIA_ROOT
)