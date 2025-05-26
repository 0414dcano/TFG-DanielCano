import os
import sys
import django

# Agrega la carpeta raíz del proyecto (un nivel arriba) al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ahora configura la variable de entorno para el settings de Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TFG.settings')

# Inicializa Django
django.setup()

# Aquí puedes poner el resto de tu código para entrenar y guardar el modelo
# Por ejemplo:
from detector.models import ModeloGuardado  # si tienes este modelo para guardar el .pt

# Ejemplo básico de uso
print("Django configurado correctamente desde detector/guardar_modelo.py")
