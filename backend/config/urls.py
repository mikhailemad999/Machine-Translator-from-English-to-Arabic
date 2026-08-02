"""
==============================================================================
ROOT URL CONFIGURATION MODULE — Django System-Wide Route Dispatcher
==============================================================================

Purpose:
  This module defines the top-level URL patterns for the Django application:
  1. '/admin/' -> Connects to Django Administration web dashboard.
  2. '/api/'   -> Delegates API endpoints to the `api.urls` routing module.
  3. Static/Media serving -> Exposes generated charts and uploads during local dev.
==============================================================================
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static('/charts/', document_root=settings.CHARTS_DIR)
