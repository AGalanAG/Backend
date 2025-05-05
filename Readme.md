# Prototipo 1 TT 
Este sistema permite gestionar múltiples cámaras RTSP, visualizar transmisiones en tiempo real, grabar videos automáticamente y detectar prendas de vestimenta usando inteligencia artificial (YOLOv8). Incluye un backend en FastAPI y un frontend en React con Material UI.

## Características principales

- **Visualización en tiempo real** de múltiples cámaras RTSP
- **Grabación automática** organizada por cámara/fecha/hora
- **Detección de vestimenta** usando modelo YOLOv8
- **Búsqueda avanzada** de grabaciones por fecha/hora/cámara
- **Búsqueda de vestimenta** por tipo/color
- **Sistema de autenticación** con roles (administrador, operador, visualizador)
- **Administración de usuarios**
- **Panel de configuración** del sistema de detección

## Requisitos previos

- Python 3.8+ 
- Node.js 14+ y npm 
- FFmpeg (para optimización de videos)
- Acceso a cámaras RTSP
- CUDA (opcional, para aceleración GPU de detección)

## Estructura del proyecto

El proyecto está dividido en dos partes:

- **Backend**: API REST en FastAPI que gestiona cámaras, grabaciones y detección
- **Frontend**: Aplicación React que proporciona la interfaz de usuario

## Instalación y configuración

### 1. Configuración del Backend

#### 1.1 Crear entorno virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

#### 1.2 Instalar dependencias de Python

```bash
pip install fastapi uvicorn opencv-python numpy torch torchvision ultralytics python-jose[cryptography] passlib bcrypt python-multipart websockets
```

#### 1.3 Instalar FFmpeg

**En Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**En Windows:**
1. Descargar FFmpeg desde [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extraer los archivos y agregar la carpeta `bin` al PATH del sistema

#### 1.4 Configurar las cámaras

Editar el archivo `app.py` para configurar las cámaras RTSP:

```python
# Modificar esta sección con tus cámaras
cameras = {
    "cam1": {
        "url": "rtsp://usuario:contraseña@ip_camara:puerto/stream",
        "name": "Nombre de la cámara",
        "rtsp_transport": "tcp",
        "timeout": 15,
        "retry_interval": 2,
        "max_retries": 5
    },
    # Añadir más cámaras según sea necesario
}
```

#### 1.5 Configurar clave secreta JWT

Editar `auth_utils.py`:

```python
# Cambiar por una clave segura (para pruebas no es necesario)
SECRET_KEY = "tu_clave_secreta_segura_aqui"
```

### 1.6 Inicializar el Backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```





