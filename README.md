# Sistema de Detección de Anomalías con Servidor Web

Este proyecto permite ejecutar un sistema de detección de anomalías a través de un servidor web. A continuación, se detallan los pasos necesarios para su instalación y puesta en marcha.

## Requisitos previos

Antes de comenzar, asegúrese de tener instalado lo siguiente:

- Python 3.10  
- `pip` (gestor de paquetes de Python)  
- Git (opcional, si va a clonar el repositorio)

## Instalación

1. **Clonar el repositorio** (opcional):

   ```bash
   git clone https://github.com/usuario/nombre-del-repositorio.git
   cd nombre-del-repositorio
   ```

2. **Crear y activar un entorno virtual** (opcional pero recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate   # En Windows: venv\Scripts\activate
   ```

3. **Instalar las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el script de inicialización**:

   Una vez instaladas las dependencias, ejecute el siguiente comando para iniciar el servidor:

   ```bash
   python setup.py
   ```

   Espere unos segundos mientras se realiza la configuración inicial. El servidor se iniciará automáticamente.

## Uso

Una vez que el servidor esté en ejecución, podrá acceder a la interfaz web a través de su navegador, normalmente en:

```
http://localhost:8000
```

Desde allí, podrá subir imágenes y recibir análisis del sistema de detección de anomalías.

## Notas adicionales

- Asegúrese de que el puerto 8000 esté libre antes de iniciar el servidor.  
- Para detener el servidor, presione `Ctrl+C` en la terminal.

---

© 2025 - Proyecto de Detección de Anomalías
