# Sistema de Detección de Anomalías con Servidor Web

Este proyecto permite ejecutar un sistema de detección de anomalías a través de un servidor web. A continuación, se detallan los pasos necesarios para su configuración y puesta en marcha.

## Requisitos previos

Antes de comenzar, asegúrese de tener instalado lo siguiente:

- **Python 3.10** (exclusivamente)
- Git (opcional, si va a clonar el repositorio)

## Descarga del repositorio

Puede clonar el repositorio con el siguiente comando:

```bash
git clone https://github.com/usuario/nombre-del-repositorio.git
cd nombre-del-repositorio
```

También puede descargar el código manualmente desde GitHub si lo prefiere o copiar el codigo fuente adjuntado en la entrega del TFG.

## Preparación del Dataset

Este sistema utiliza el dataset [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads).

1. Descargue el dataset desde el siguiente enlace:  
   👉 https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

2. Descomprima el archivo descargado en cualquier ubicación de su equipo.

3. Abra el archivo `entrenar_guardar_modelo.py` y localice la variable `dataset_path`. Modifíquela con la ruta donde haya descomprimido el dataset. Por ejemplo:

   ```python
   dataset_path = r"ruta/al/dataset/"
   ```

   Asegúrese de mantener el prefijo `r` para que la ruta se interprete correctamente como literal.

## Ejecución del servidor

Una vez configurado el dataset, ejecute el siguiente comando para iniciar el servidor:

```bash
python setup.py
```

Espere unos minutos mientras se realiza la configuración inicial. El servidor se iniciará automáticamente.

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
