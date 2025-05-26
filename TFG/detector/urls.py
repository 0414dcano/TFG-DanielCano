from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat'),
    #path('inferencia/', views.inferencia_view, name='inferencia'),
]
