from django.db import models
from django.contrib.auth.models import User
from django.db import models

from django.db import models

class ModeloGuardado(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    modelo_binario = models.BinaryField(null=True, blank=True)
    mean_binario = models.BinaryField(default=b'')
    cov_binario = models.BinaryField(default=b'')
    fecha_guardado = models.DateTimeField(auto_now=True)



class Conversacion(models.Model):
    usuario = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversaciones')
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversación {self.id} de {self.usuario.username} ({self.fecha.strftime('%Y-%m-%d %H:%M')})"

class Mensaje(models.Model):
    conversacion = models.ForeignKey(Conversacion, on_delete=models.CASCADE, related_name='mensajes')
    imagen = models.ImageField(upload_to='conversaciones/')
    respuesta = models.TextField()
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Mensaje {self.id} en {self.conversacion}"
