# app.py - Backend principal actualizado con detección de vestimenta
import asyncio
import cv2
import uvicorn
import logging
import base64
import numpy as np
import time
import threading
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import contextmanager
from fastapi.security import HTTPBearer
from auth_endpoints import router as auth_router
from auth_middleware import viewer_required, operator_required, admin_required

# Comentarios explicativos sobre los niveles de acceso
VIEWER = 1    # Solo puede ver cámaras
OPERATOR = 2  # Puede ver cámaras, grabaciones y búsqueda
ADMIN = 3     # Acceso completo, puede gestionar usuarios

# Importar el gestor de grabaciones
from recording_manager import RecordingManager

# Importar el router de video endpoints
from video_endpoints import router as video_router

# Importar los nuevos módulos de detección de vestimenta
from clothing_detector import ClothingDetector
from detection_db import DetectionDB
from detection_service import DetectionService
from clothing_endpoints import router as clothing_router, configure as configure_clothing_router

app = FastAPI(title="Sistema de Visualización, Grabación y Detección de Vestimenta en Cámaras RTSP")

# Configuración CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar el origen exacto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Crear directorios necesarios
os.makedirs("recordings", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Inicializar el gestor de grabaciones
recording_manager = RecordingManager(
    storage_path="recordings",
    segment_duration=5,  # 5 minutos por segmento
    retention_days=7     # Conservar 7 días de grabaciones
)

# Inicializar componentes de detección de vestimenta
detection_db = DetectionDB(db_path="./data/detections.db")

# Inicializar detector de vestimenta
try:
    clothing_detector = ClothingDetector(
        model_path=None,  # Usar modelo predeterminado (se descargará si es necesario)
        confidence_threshold=0.4
    )
    
    # Inicializar servicio de detección
    detection_service = DetectionService(
        detector=clothing_detector,
        db=detection_db,
        detection_interval=2.0,  # Procesar cada 2 segundos para economizar recursos
        min_confidence=0.4,
        enabled=True
    )
    
    # Configurar router de detección de vestimenta
    configure_clothing_router(detection_service, detection_db)
    
    logger.info("Sistema de detección de vestimenta inicializado correctamente")
except Exception as e:
    logger.error(f"Error al inicializar sistema de detección de vestimenta: {e}")
    detection_service = None
    logger.warning("El sistema funcionará sin detección de vestimenta")

# Clase para gestionar conexiones WebSocket (sin cambios)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.camera_tasks = {}
        self.client_ips: Dict[str, Set[str]] = {}  # Seguimiento por IP para estadísticas

    async def connect(self, websocket: WebSocket, camera_id: str, client_info: dict):
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
        
        # Guardar información del cliente para análisis
        client_ip = client_info.get("client_ip", "unknown")
        if camera_id not in self.client_ips:
            self.client_ips[camera_id] = set()
        self.client_ips[camera_id].add(client_ip)
        
        self.active_connections[camera_id].append(websocket)
        logger.info(f"Cliente {client_ip} conectado a cámara {camera_id}. "
                   f"Total: {len(self.active_connections[camera_id])}, "
                   f"Clientes únicos: {len(self.client_ips[camera_id])}")

    def disconnect(self, websocket: WebSocket, camera_id: str, client_info: dict = None):
        if camera_id in self.active_connections:
            if websocket in self.active_connections[camera_id]:
                self.active_connections[camera_id].remove(websocket)
                
                logger.info(f"Cliente desconectado de cámara {camera_id}. "
                           f"Restantes: {len(self.active_connections[camera_id]) if camera_id in self.active_connections else 0}")

    async def broadcast(self, camera_id: str, data: str):
        if camera_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[camera_id]:
                try:
                    await connection.send_text(data)
                except Exception as e:
                    logger.error(f"Error al enviar datos: {e}")
                    dead_connections.append(connection)
            
            # Limpiar conexiones muertas
            for dead_conn in dead_connections:
                self.disconnect(dead_conn, camera_id)
    
    def get_stats(self):
        """Devuelve estadísticas de conexiones activas"""
        stats = {
            "total_connections": sum(len(conns) for conns in self.active_connections.values()),
            "cameras_with_clients": len([cam for cam, conns in self.active_connections.items() if len(conns) > 0]),
            "connections_per_camera": {cam: len(conns) for cam, conns in self.active_connections.items()},
            "unique_clients_per_camera": {cam: len(ips) for cam, ips in self.client_ips.items()}
        }
        return stats

manager = ConnectionManager()

# Clase de gestión de conexiones RTSP actualizada para soportar detección
class RTSPManager:
    def __init__(self, rtsp_url, camera_id, camera_name, timeout=10):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.timeout = timeout
        self.cap = None
        self.connected = False
        self.last_frame_time = 0
        self.connection_attempts = 0
        self.max_attempts = 5
        self.backoff_time = 1  # segundos para esperar entre intentos, aumenta exponencialmente
        
        # Caché del último frame
        self.last_frame = None
        self.last_frame_quality = 65  # Calidad JPEG por defecto
        self.last_frame_base64 = None
        
        # Contador de clientes y métricas
        self.active_client_count = 0
        self.frames_processed = 0
        self.start_time = time.time()
        
        # Control de grabación
        self.recording_enabled = True  # Habilitado por defecto
        
        # Control de detección de vestimenta
        self.detection_enabled = True  # Habilitado por defecto
        
        # Variables para el cálculo de FPS
        self.fps_calculation_start = time.time()
        self.fps_frame_count = 0
        self.actual_fps = 15.0  # Valor inicial conservador
        self.fps_update_interval = 5  # Actualizar cada 5 segundos
        
        # Estadísticas de detección para esta cámara
        self.detection_stats = {
            "detected_items": 0,
            "last_detection_time": 0,
            "detection_success_rate": 0,
            "frames_processed_for_detection": 0
        }
        
        # Iniciar grabación automáticamente
        if self.recording_enabled:
            recording_manager.start_recording(camera_id, camera_name)
    
    async def connect(self):
        """Intenta conectar a la fuente RTSP con reintentos y tiempo de espera exponencial"""
        self.connection_attempts += 1
        
        logger.info(f"Intento de conexión #{self.connection_attempts} a {self.rtsp_url}")
        
        # Liberar cualquier conexión existente
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Diferentes opciones de conexión para probar (incluyendo comandos FFMPEG)
        connection_options = [
            # Opción 1: OpenCV estándar
            lambda: cv2.VideoCapture(self.rtsp_url),
            
            # Opción 2: OpenCV con transporte TCP explícito via ffmpeg
            lambda: cv2.VideoCapture(f"ffmpeg -rtsp_transport tcp -i {self.rtsp_url} -f rawvideo -pix_fmt bgr24 -"),
            
            # Opción 3: Configuración con buffer y timeouts mínimos
            lambda: self._configure_capture(cv2.VideoCapture(self.rtsp_url))
        ]
        
        # Probar cada opción en secuencia
        for i, create_capture in enumerate(connection_options):
            try:
                logger.info(f"Probando método de conexión #{i+1}")
                self.cap = create_capture()
                
                # Verificar si la conexión fue exitosa
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read(cv2.CAP_PROP_BUFFERSIZE)
                    if ret and test_frame is not None:
                        logger.info(f"Conexión exitosa usando método #{i+1}")
                        self.connected = True
                        self.connection_attempts = 0  # Reiniciar contador de intentos
                        self.backoff_time = 1  # Reiniciar tiempo de espera
                        
                        # Si la conexión se recupera, reiniciar grabación si estaba activa
                        if self.recording_enabled:
                            recording_manager.start_recording(self.camera_id, self.camera_name)
                            
                        return True
                
                # Si llegamos aquí, este método no funcionó
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
            
            except Exception as e:
                logger.warning(f"Error intentando método de conexión #{i+1}: {str(e)}")
        
        # Si todos los métodos fallaron
        logger.error(f"No se pudo conectar a {self.rtsp_url} después de probar todos los métodos")
        
        # Implementar espera exponencial entre intentos
        if self.connection_attempts < self.max_attempts:
            wait_time = min(30, self.backoff_time * (2 ** (self.connection_attempts - 1)))
            logger.info(f"Esperando {wait_time} segundos antes del próximo intento...")
            await asyncio.sleep(wait_time)
            return await self.connect()  # Retry recursively
        else:
            logger.error(f"Se alcanzó el número máximo de intentos ({self.max_attempts}). Abortando conexión.")
            self.connected = False
            return False
    
    def _configure_capture(self, cap):
        """Configura parámetros avanzados en el objeto de captura"""
        if cap.isOpened():
            # Configurar para mínima latencia
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Intentar varias opciones de backend
            try:
                # Intentar usar FFMPEG como backend
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            except:
                pass
                
            # Otras propiedades que se pueden ajustar (dependiendo del soporte de la cámara)
            try:
                # Reducir cualquier timeout interno
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)  # 1 segundo timeout para leer
            except:
                pass
        
        return cap
    
    async def read_frame(self):
        """Lee un frame con timeout y manejo de errores, procesa detección si está habilitada"""
        if not self.connected or self.cap is None:
            success = await self.connect()
            if not success:
                return False, None
        
        try:
            # Implementar timeout en la lectura a través de un executor
            loop = asyncio.get_event_loop()
            read_task = loop.run_in_executor(None, self.cap.read)
            
            # Esperar con timeout
            try:
                ret, frame = await asyncio.wait_for(read_task, timeout=self.timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout al leer frame de {self.rtsp_url}")
                # Marcar como desconectado para forzar reconexión
                self.connected = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False, None
            
            # Verificar resultado
            if not ret or frame is None:
                logger.warning(f"No se pudo leer frame de {self.rtsp_url}")
                self.connected = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False, None
            
            # Frame exitoso - actualizar cálculo de FPS
            self.last_frame_time = time.time()
            current_time = self.last_frame_time
            
            # Actualizar caché
            self.last_frame = frame.copy()
            
            # Incrementar contador de frames procesados
            self.frames_processed += 1
            
            # Actualizar cálculo de FPS
            self.fps_frame_count += 1
            elapsed = current_time - self.fps_calculation_start
            
            # Actualizar el FPS cada X segundos para tener una medición más estable
            if elapsed >= self.fps_update_interval:
                new_fps = self.fps_frame_count / elapsed
                # Actualizar solo si es un valor razonable
                if 1.0 <= new_fps <= 60.0:
                    # Actualización suave del FPS (70% valor anterior, 30% nuevo valor)
                    self.actual_fps = 0.7 * self.actual_fps + 0.3 * new_fps
                    logger.info(f"FPS actualizado para cámara {self.camera_id}: {self.actual_fps:.2f}")
                
                # Reiniciar contadores
                self.fps_frame_count = 0
                self.fps_calculation_start = current_time
            
            # Grabar el frame si la grabación está habilitada
            video_path = None
            video_time = 0
            if self.recording_enabled:
                result, video_info = recording_manager.record_frame(self.camera_id, frame, self.actual_fps)
                if result and video_info:
                    video_path = video_info.get("path")
                    video_time = video_info.get("time")
            
            # Procesar detección de vestimenta si está habilitada y el servicio está disponible
            if self.detection_enabled and detection_service and video_path:
                try:
                    self.detection_stats["frames_processed_for_detection"] += 1
                    
                    # Callback para actualizar estadísticas
                    def detection_callback(detections):
                        self.detection_stats["detected_items"] += len(detections)
                        self.detection_stats["last_detection_time"] = time.time()
                        if self.detection_stats["frames_processed_for_detection"] > 0:
                            self.detection_stats["detection_success_rate"] = (
                                self.detection_stats["detected_items"] / 
                                self.detection_stats["frames_processed_for_detection"]
                            )
                        
                    # Procesar el frame para detección
                    was_processed = detection_service.process_frame(
                        frame=frame,
                        camera_id=self.camera_id,
                        video_path=video_path,
                        video_time=video_time,
                        callback=detection_callback
                    )
                        
                except Exception as e:
                    logger.error(f"Error en detección de vestimenta para cámara {self.camera_id}: {e}")
            
            return True, frame
        
        except Exception as e:
            logger.error(f"Error al leer frame: {str(e)}")
            self.connected = False
            if self.cap:
                self.cap.release()
                self.cap = None
            return False, None
    
    def get_cached_frame_base64(self, target_width=640, quality=65):
        """Devuelve el último frame capturado como base64 para envío rápido a nuevos clientes"""
        if self.last_frame is None:
            return None
            
        # Si ya tenemos una versión procesada con la misma calidad, reutilizarla
        if self.last_frame_base64 is not None and self.last_frame_quality == quality:
            return self.last_frame_base64
            
        try:
            # Redimensionar proporcionalmente para mantener relación de aspecto
            h, w = self.last_frame.shape[:2]
            target_height = int(h * (target_width / w))
            resized = cv2.resize(self.last_frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Compresión JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode('.jpg', resized, encode_param)
            self.last_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            self.last_frame_quality = quality
            
            return self.last_frame_base64
        except Exception as e:
            logger.error(f"Error al procesar frame en caché: {str(e)}")
            return None
    
    def close(self):
        """Cierra y libera recursos"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.connected = False
        
        # Detener grabación
        if self.recording_enabled:
            recording_manager.stop_recording(self.camera_id)
        
    def get_stats(self):
        """Devuelve estadísticas de rendimiento"""
        elapsed = time.time() - self.start_time
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        
        # Obtener estado de grabación
        recording_status = recording_manager.get_recording_status(self.camera_id)
        
        # Preparar información de detección
        detection_info = {
            "enabled": self.detection_enabled,
            "detected_items": self.detection_stats["detected_items"],
            "last_detection_ago": round(time.time() - self.detection_stats["last_detection_time"], 1) 
                                  if self.detection_stats["last_detection_time"] > 0 else -1,
            "success_rate": round(self.detection_stats["detection_success_rate"] * 100, 1)
        }
        
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "url": self.rtsp_url,
            "connected": self.connected,
            "last_frame_time": self.last_frame_time,
            "active_clients": self.active_client_count,
            "frames_processed": self.frames_processed,
            "uptime_seconds": elapsed,
            "average_fps": round(fps, 2),
            "current_fps": round(self.actual_fps, 2),
            "recording": {
                "enabled": self.recording_enabled,
                "status": recording_status if recording_status else "No activa"
            },
            "detection": detection_info
        }
    
    def toggle_recording(self, enabled: bool) -> bool:
        """Activa o desactiva la grabación"""
        self.recording_enabled = enabled
        
        if enabled:
            recording_manager.start_recording(self.camera_id, self.camera_name)
            logger.info(f"Grabación activada para cámara {self.camera_id}")
        else:
            recording_manager.stop_recording(self.camera_id)
            logger.info(f"Grabación desactivada para cámara {self.camera_id}")
        
        return self.recording_enabled
    
    def toggle_detection(self, enabled: bool) -> bool:
        """Activa o desactiva la detección de vestimenta para esta cámara"""
        self.detection_enabled = enabled
        logger.info(f"Detección de vestimenta {('activada' if enabled else 'desactivada')} para cámara {self.camera_id}")
        return self.detection_enabled

# Cámaras disponibles (se puede cargar desde configuración o base de datos)
cameras = {
    "cam1": {
        "url": "rtsp://agalan:SpiderMan0818$@192.168.100.188:554/stream1",
        "name": "Entrada Principal",
        "rtsp_transport": "tcp",  # tcp, udp, http, https
        "timeout": 15,            # segundos
        "retry_interval": 2,      # segundos
        "max_retries": 5
    }
}

# Diccionario para almacenar los gestores RTSP activos
rtsp_managers = {}

# Configurar directorio de grabaciones como estático para servir archivos
recordings_path = Path("recordings")
recordings_path.mkdir(exist_ok=True)
app.mount("/recordings", StaticFiles(directory="recordings"), name="recordings")

# Endpoints API
@app.get("/", dependencies=[Depends(viewer_required)])
async def get_root():
    return {"message": "Servicio de streaming, grabación y detección de vestimenta RTSP activo"}

@app.get("/cameras", dependencies=[Depends(viewer_required)])
async def get_cameras():
    """Devuelve la lista de cámaras disponibles"""
    return [{"id": id, "name": details["name"]} for id, details in cameras.items()]

@app.get("/camera/{camera_id}/status", dependencies=[Depends(viewer_required)])
async def get_camera_status(camera_id: str):
    """Devuelve el estado de conexión de una cámara específica"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    if camera_id in rtsp_managers:
        return rtsp_managers[camera_id].get_stats()
    else:
        return {
            "id": camera_id,
            "name": cameras[camera_id]["name"],
            "connected": False,
            "last_frame_time": 0,
            "active_clients": 0,
            "recording": {"enabled": False, "status": "No activa"},
            "detection": {"enabled": False}
        }

@app.post("/camera/{camera_id}/recording", dependencies=[Depends(operator_required)])
async def toggle_camera_recording(camera_id: str, enabled: bool = True):
    """Activa o desactiva la grabación para una cámara específica"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    if camera_id not in rtsp_managers:
        # La cámara no está conectada actualmente
        raise HTTPException(status_code=400, detail="La cámara no está activa actualmente")
    
    result = rtsp_managers[camera_id].toggle_recording(enabled)
    
    return {
        "camera_id": camera_id,
        "recording_enabled": result
    }

@app.post("/camera/{camera_id}/detection", dependencies=[Depends(operator_required)])
async def toggle_camera_detection(camera_id: str, enabled: bool = True):
    """Activa o desactiva la detección de vestimenta para una cámara específica"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Cámara no encontrada")
    
    if camera_id not in rtsp_managers:
        # La cámara no está conectada actualmente
        raise HTTPException(status_code=400, detail="La cámara no está activa actualmente")
    
    # Verificar que el servicio de detección esté disponible
    if not detection_service:
        raise HTTPException(status_code=503, detail="Servicio de detección no disponible")
    
    result = rtsp_managers[camera_id].toggle_detection(enabled)
    
    return {
        "camera_id": camera_id,
        "detection_enabled": result
    }

@app.get("/status", dependencies=[Depends(viewer_required)])
async def get_server_status():
    """Devuelve estadísticas del servidor y todas las cámaras"""
    camera_stats = {camera_id: rtsp_managers[camera_id].get_stats() 
                   if camera_id in rtsp_managers else {"connected": False} 
                   for camera_id in cameras}
    
    connection_stats = manager.get_stats()
    
    # Estadísticas de grabación
    recording_stats = {
        "storage": recording_manager.get_storage_usage(),
        "camera_status": {cam_id: recording_manager.get_recording_status(cam_id) 
                         for cam_id in cameras}
    }
    
    # Estadísticas de detección
    detection_stats = None
    if detection_service:
        detection_stats = detection_service.get_stats()
    
    return {
        "server": {
            "uptime": time.time() - server_start_time,
            "version": "1.3.0",  # Versión con soporte de detección de vestimenta
        },
        "connections": connection_stats,
        "cameras": camera_stats,
        "recording": recording_stats,
        "detection": detection_stats
    }

# Endpoints para gestionar grabaciones
@app.get("/recordings", dependencies=[Depends(operator_required)])
async def list_recordings(
    camera_id: Optional[str] = None,
    date: Optional[str] = None,
    hour: Optional[str] = None
):
    """Lista las grabaciones disponibles con filtros opcionales"""
    try:
        recordings = recording_manager.list_recordings(camera_id, date, hour)
        
        # Añadir URLs para reproducción y descarga
        for recording in recordings:
            recording["url"] = f"/recordings/{recording['path']}"
            recording["stream_url"] = f"/api/recordings/stream/{recording['path']}"
        
        return {
            "count": len(recordings),
            "recordings": recordings
        }
    except Exception as e:
        logger.error(f"Error al listar grabaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al listar grabaciones: {str(e)}")

@app.get("/api/recordings/download/{path:path}", dependencies=[Depends(operator_required)])
async def download_recording(path: str):
    """Endpoint para descargar un archivo de grabación"""
    try:
        full_path = recording_manager.get_recording_path(path)
        
        if not full_path:
            raise HTTPException(status_code=404, detail="Grabación no encontrada")
        
        # Determinar el tipo de contenido basado en la extensión del archivo
        extension = os.path.splitext(full_path)[1].lower()
        media_type = "video/x-msvideo" if extension == ".avi" else "video/mp4"
        
        return FileResponse(
            full_path, 
            media_type=media_type,
            filename=os.path.basename(full_path)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al descargar grabación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al descargar grabación: {str(e)}")

@app.get("/api/recordings/stream/{path:path}", dependencies=[Depends(operator_required)])
async def stream_recording(path: str, response: Response):
    """Endpoint para transmitir un archivo de grabación en streaming"""
    try:
        full_path = recording_manager.get_recording_path(path)
        
        if not full_path:
            raise HTTPException(status_code=404, detail="Grabación no encontrada")
        
        # Obtener el tamaño del archivo
        file_size = os.path.getsize(full_path)
        
        # Determinar el tipo de contenido basado en la extensión del archivo
        extension = os.path.splitext(full_path)[1].lower()
        media_type = "video/x-msvideo" if extension == ".avi" else "video/mp4"
        
        # Función para streaming por bloques
        def iterfile():
            with open(full_path, mode="rb") as file_like:
                while chunk := file_like.read(8192):  # Leer en bloques de 8KB
                    yield chunk
        
        # Configurar headers adecuados para video streaming
        response.headers["Content-Type"] = media_type
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Content-Length"] = str(file_size)
        
        return StreamingResponse(
            iterfile(),
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al transmitir grabación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al transmitir grabación: {str(e)}")

@app.get("/api/recordings/dates", dependencies=[Depends(operator_required)])
async def get_recording_dates(camera_id: Optional[str] = None):
    """Devuelve las fechas disponibles con grabaciones"""
    try:
        dates = set()
        recordings = recording_manager.list_recordings(camera_id)
        
        for recording in recordings:
            dates.add(recording["date"])
        
        return {
            "dates": sorted(list(dates), reverse=True)
        }
    except Exception as e:
        logger.error(f"Error al obtener fechas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener fechas: {str(e)}")

@app.get("/api/recordings/hours", dependencies=[Depends(operator_required)])
async def get_recording_hours(camera_id: Optional[str] = None, date: Optional[str] = None):
    """Devuelve las horas disponibles con grabaciones para una fecha específica"""
    try:
        hours = set()
        recordings = recording_manager.list_recordings(camera_id, date)
        
        for recording in recordings:
            hours.add(recording["hour"])
        
        return {
            "hours": sorted(list(hours))
        }
    except Exception as e:
        logger.error(f"Error al obtener horas: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener horas: {str(e)}")

@app.get("/api/video/byterange/{path:path}", dependencies=[Depends(operator_required)])
async def get_video_with_byte_range(path: str, request: Request, response: Response):
    """
    Endpoint optimizado para videos que implementa el protocolo HTTP Range
    para permitir que los navegadores soliciten partes específicas del video.
    """
    try:
        # Construir ruta completa al directorio de grabaciones
        full_path = os.path.join("recordings", path)
        
        if not os.path.exists(full_path):
            logger.error(f"Archivo no encontrado: {full_path}")
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {path}")
        
        # Obtener el tamaño del archivo
        file_size = os.path.getsize(full_path)
        
        # Verificar si se solicitó un rango específico
        range_header = request.headers.get("Range", None)
        
        # Implementar soporte para byte ranges
        async def send_file_partial(start, end):
            with open(full_path, 'rb') as f:
                f.seek(start)
                chunk_size = 8192  # 8KB por chunk
                remaining = end - start + 1
                
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        # Si hay un encabezado Range, procesar la solicitud parcial
        if range_header:
            try:
                range_header = range_header.replace("bytes=", "").split("-")
                start = int(range_header[0])
                end = int(range_header[1]) if range_header[1] else file_size - 1
            except (IndexError, ValueError):
                # Si el rango no es válido, usar todo el archivo
                start = 0
                end = file_size - 1
                
            # Limitar el fin al tamaño del archivo
            end = min(end, file_size - 1)
            content_length = end - start + 1
            
            # Configurar headers para respuesta parcial
            response.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            response.headers["Content-Length"] = str(content_length)
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Content-Type"] = "video/mp4"
            response.status_code = 206  # Partial content
            
            return StreamingResponse(
                send_file_partial(start, end),
                media_type="video/mp4",
                status_code=206
            )
        else:
            # Devolver todo el archivo
            response.headers["Content-Length"] = str(file_size)
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Content-Type"] = "video/mp4"
            
            return StreamingResponse(
                send_file_partial(0, file_size - 1),
                media_type="video/mp4"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al transmitir video con range: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al transmitir video: {str(e)}")

# Obtener cliente IP para estadísticas
@app.middleware("http")
async def add_client_info(request: Request, call_next):
    client_host = request.client.host if request.client else "unknown"
    request.state.client_info = {"client_ip": client_host}
    return await call_next(request)

# WebSocket para streaming de video
@app.websocket("/ws/stream/{camera_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    camera_id: str, 
    quality: int = Query(65, ge=10, le=95)  # Permitir al cliente especificar calidad
):
    # Verificar si la cámara existe
    if camera_id not in cameras:
        await websocket.accept()
        await websocket.send_text(str({
            "c": camera_id,
            "error": "Cámara no encontrada",
            "t": int(time.time() * 1000)
        }))
        await websocket.close(code=1008, reason="Cámara no encontrada")
        return
    
    # Obtener información del cliente para estadísticas
    client_info = {"client_ip": "unknown"}
    if hasattr(websocket, "scope") and "client" in websocket.scope:
        client_host = websocket.scope.get("client", ("unknown", 0))[0]
        client_info = {"client_ip": client_host}
    
    # Conexión y registro
    await manager.connect(websocket, camera_id, client_info)
    
    # Crear o recuperar el gestor RTSP para esta cámara
    if camera_id not in rtsp_managers:
        camera_config = cameras[camera_id]
        rtsp_managers[camera_id] = RTSPManager(
            rtsp_url=camera_config["url"],
            camera_id=camera_id,
            camera_name=camera_config["name"],
            timeout=camera_config.get("timeout", 10)
        )
    
    # Actualizar contador de clientes activos
    rtsp_managers[camera_id].active_client_count += 1
    
    try:
        # Iniciar transmisión en un task separado si no está ya en ejecución
        if camera_id not in manager.camera_tasks or manager.camera_tasks[camera_id].done():
            stream_task = asyncio.create_task(stream_camera(camera_id, quality))
            manager.camera_tasks[camera_id] = stream_task
        
        # Si ya tenemos un frame en caché, enviarlo inmediatamente para reducir la espera inicial
        if rtsp_managers[camera_id].last_frame is not None:
            cached_frame = rtsp_managers[camera_id].get_cached_frame_base64(quality=quality)
            if cached_frame:
                initial_data = {
                    "c": camera_id,
                    "t": int(time.time() * 1000),
                    "i": cached_frame
                }
                await websocket.send_text(str(initial_data))
        
        # Mantener conexión y manejar mensajes del cliente
        while True:
            data = await websocket.receive_text()
            # Aquí se pueden implementar comandos como pausar/reanudar
            if data == "stop":
                break
    except WebSocketDisconnect:
        logger.info(f"Cliente desconectado de cámara {camera_id}")
    except Exception as e:
        logger.error(f"Error en conexión WebSocket para cámara {camera_id}: {str(e)}")
    finally:
        manager.disconnect(websocket, camera_id, client_info)
        
        # Actualizar contador de clientes activos
        if camera_id in rtsp_managers:
            rtsp_managers[camera_id].active_client_count = max(0, rtsp_managers[camera_id].active_client_count - 1)
        
        # Si no quedan clientes, cerrar el stream después de un tiempo de inactividad
        if camera_id not in manager.active_connections or len(manager.active_connections[camera_id]) == 0:
            # No cancelamos inmediatamente para permitir reconexiones rápidas
            asyncio.create_task(close_stream_after_inactivity(camera_id, 30))  # 30 segundos de inactividad

async def close_stream_after_inactivity(camera_id: str, timeout_seconds: int):
    """Cierra el stream después de un período de inactividad si no hay clientes"""
    await asyncio.sleep(timeout_seconds)
    
    # Verificar si todavía no hay clientes después del tiempo de espera
    if camera_id not in manager.active_connections or len(manager.active_connections[camera_id]) == 0:
        if camera_id in manager.camera_tasks and not manager.camera_tasks[camera_id].done():
            logger.info(f"Cerrando stream de cámara {camera_id} después de {timeout_seconds}s de inactividad")
            manager.camera_tasks[camera_id].cancel()
        
        # Nota: No detenemos la grabación, sigue en segundo plano

async def stream_camera(camera_id: str, quality: int = 65):
    """Función para transmitir frames de una cámara RTSP específica con calidad adaptativa"""
    rtsp_manager = rtsp_managers[camera_id]
    
    logger.info(f"Iniciando streaming para cámara {camera_id}: {rtsp_manager.rtsp_url}")
    
    # Intentar conectar inicialmente
    connected = await rtsp_manager.connect()
    if not connected:
        logger.error(f"No se pudo establecer la conexión inicial con la cámara {camera_id}")
        # Notificar a los clientes que la cámara no está disponible
        error_data = {
            "c": camera_id,
            "error": "No se pudo conectar a la cámara",
            "t": int(time.time() * 1000)
        }
        await manager.broadcast(camera_id, str(error_data))
        return
    
    try:
        # Configurar FPS objetivo según número de clientes
        base_fps = 20  # FPS base cuando hay pocos clientes
        min_fps = 5    # FPS mínimo cuando hay muchos clientes
        
        # Calcular FPS basado en número de clientes (reduce FPS cuando hay muchos clientes)
        client_count = len(manager.active_connections.get(camera_id, []))
        adaptive_fps = max(min_fps, base_fps - (client_count // 3))
        
        frame_interval = 1.0 / adaptive_fps
        last_frame_time = asyncio.get_event_loop().time()
        frames_sent = 0
        
        # Ajustar calidad JPEG según número de clientes
        adaptive_quality = max(30, quality - (client_count * 2))
        
        while True:
            # Optimización: Sincronización precisa para mantener FPS constante
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - last_frame_time
            
            # Sólo esperar si estamos procesando más rápido que el objetivo
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            # Leer frame con manejo de errores integrado
            success, frame = await rtsp_manager.read_frame()
            
            if not success:
                # El RTSPManager ya habrá intentado reconectar
                # Esperar un poco antes de intentar de nuevo
                await asyncio.sleep(0.5)
                continue
            
            # Verificar si aún hay clientes conectados para streaming
            client_count = len(manager.active_connections.get(camera_id, []))
            if client_count == 0:
                logger.info(f"No hay clientes conectados a cámara {camera_id}, pausando streaming")
                await asyncio.sleep(1.0)  # Verificar periódicamente
                
                # Nota: Aunque no haya clientes, seguimos leyendo frames para grabación
                # pero a una tasa reducida para ahorrar recursos
                if rtsp_manager.recording_enabled:
                    success, frame = await rtsp_manager.read_frame()
                
                continue
            
            # Procesar frame exitoso
            # Redimensionar proporcionalmente para mantener relación de aspecto
            h, w = frame.shape[:2]
            target_width = 640
            target_height = int(h * (target_width / w))
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Actualizar FPS y calidad adaptativamente cada 50 frames
            if frames_sent % 50 == 0:
                client_count = len(manager.active_connections.get(camera_id, []))
                adaptive_fps = max(min_fps, base_fps - (client_count // 3))
                frame_interval = 1.0 / adaptive_fps
                adaptive_quality = max(30, quality - (client_count * 2))
                
                logger.debug(f"Cámara {camera_id}: Ajustando a {adaptive_fps} FPS y calidad {adaptive_quality} "
                           f"para {client_count} clientes")
            
            # Compresión JPEG eficiente con calidad adaptativa
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, adaptive_quality]
            _, buffer = cv2.imencode('.jpg', resized, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Simplificar datos para reducir tamaño
            frame_data = {
                "c": camera_id,
                "t": int(current_time * 1000),
                "i": jpg_as_text
            }
            
            # Enviar a todos los clientes conectados
            await manager.broadcast(camera_id, str(frame_data))
            
            # Actualizar estadísticas
            frames_sent += 1
            last_frame_time = current_time
    
    except asyncio.CancelledError:
        logger.info(f"Streaming de cámara {camera_id} cancelado")
    except Exception as e:
        logger.error(f"Error en streaming de cámara {camera_id}: {str(e)}")
    finally:
        # No detenemos la grabación, sólo el streaming
        logger.info(f"Finalizando tarea de streaming para cámara {camera_id}")

# Incluir los routers de API definidos
app.include_router(video_router)
app.include_router(clothing_router)

# Modificar el RecordingManager para devolver información del archivo en record_frame
def update_recording_manager():
    """Actualizar el método record_frame del RecordingManager para devolver información del archivo"""
    original_record_frame = recording_manager.record_frame
    
    def new_record_frame(camera_id, frame, fps=None):
        """Versión modificada que devuelve información del archivo grabado"""
        # Llamar a la implementación original
        result = original_record_frame(camera_id, frame, fps)
        
        # Si el frame fue grabado, devolver información adicional
        if result:
            # Obtener información del archivo actual
            recording_info = recording_manager.active_recordings.get(camera_id, {})
            current_filepath = recording_info.get("current_filepath")
            
            if current_filepath:
                # Calcular tiempo dentro del video
                segment_start_time = recording_info.get("segment_start_time", 0)
                video_time = time.time() - segment_start_time
                
                # Convertir ruta absoluta a ruta relativa para las URL
                rel_path = os.path.relpath(current_filepath, recording_manager.storage_path)
                
                return True, {
                    "path": rel_path,
                    "time": video_time,
                    "fps": recording_info.get("fps", 15.0)
                }
        
        return result, None
    
    # Reemplazar el método en el objeto
    recording_manager.record_frame = new_record_frame
    logger.info("RecordingManager.record_frame modificado para devolver información de archivo")

# Iniciar la actualización del RecordingManager
update_recording_manager()

# Almacenar tiempo de inicio del servidor para estadísticas
server_start_time = time.time()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
