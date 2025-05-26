from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),

    # URLs de autenticación (login con Gmail, logout, registro, etc.)
    path('accounts/', include('allauth.urls')),

    # Tu app principal (chat, carga de imagen, etc.)
    path('', include('detector.urls')),
]

# Servir archivos de usuario (media) en modo desarrollo
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
