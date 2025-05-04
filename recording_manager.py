import cv2
import os
import time
import logging
import threading
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class RecordingManager:
    """
    Gestor de grabaciones de cámaras que maneja:
    - Grabación automática en segmentos de 5 minutos
    - Organización de archivos por cámara/fecha/hora
    - Listado y recuperación de grabaciones
    """
    
    def __init__(self, 
                 storage_path: str = "recordings", 
                 segment_duration: int = 5,      # duración en minutos
                 retention_days: int = 7):        # días de retención
        self.storage_path = Path(storage_path)
        self.segment_duration = segment_duration
        self.retention_days = retention_days
        
        # Asegurar que el directorio de almacenamiento exista
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Estado interno
        self.active_recordings: Dict[str, Dict] = {}  # camera_id -> recording_info
        self.cleanup_thread = None
        
        # Iniciar limpieza automática
        self._start_cleanup_thread()
        
        logger.info(f"RecordingManager inicializado: almacenamiento en {self.storage_path}, "
                   f"segmentos de {segment_duration} minutos, retención de {retention_days} días")
    
    def _start_cleanup_thread(self):
        """Inicia un thread para la limpieza periódica de grabaciones antiguas"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(
                target=self._periodic_cleanup,
                daemon=True
            )
            self.cleanup_thread.start()
            
    def _periodic_cleanup(self):
        """Ejecuta la limpieza de grabaciones antiguas periódicamente"""
        while True:
            try:
                logger.info("Iniciando limpieza de grabaciones antiguas...")
                self.cleanup_old_recordings()
                logger.info(f"Limpieza completada. Próxima ejecución en 24 horas.")
                # Dormir 24 horas antes de la siguiente limpieza
                time.sleep(86400)  # 24 horas en segundos
            except Exception as e:
                logger.error(f"Error durante la limpieza periódica: {str(e)}")
                # Si hay error, esperar una hora e intentar de nuevo
                time.sleep(3600)
    
    def start_recording(self, camera_id: str, camera_name: str):
        """
        Inicia la grabación para una cámara específica
        
        Args:
            camera_id: Identificador único de la cámara
            camera_name: Nombre descriptivo de la cámara
        """
        if camera_id in self.active_recordings:
            logger.warning(f"La grabación para cámara {camera_id} ya está activa")
            return
        
        # Información del proceso de grabación
        self.active_recordings[camera_id] = {
            "camera_name": camera_name,
            "start_time": time.time(),
            "current_writer": None,
            "current_filepath": None,
            "segment_start_time": time.time(),
            "frames_recorded": 0,
            "active": True,
            "fps": 15.0  # FPS predeterminado conservador
        }
        
        logger.info(f"Grabación iniciada para cámara {camera_id} ({camera_name})")
    
    def stop_recording(self, camera_id: str):
        """
        Detiene la grabación para una cámara específica
        
        Args:
            camera_id: Identificador único de la cámara
        """
        if camera_id not in self.active_recordings:
            logger.warning(f"No hay grabación activa para cámara {camera_id}")
            return
        
        recording_info = self.active_recordings[camera_id]
        
        # Liberar recursos del writer si existe
        if recording_info["current_writer"] is not None:
            try:
                recording_info["current_writer"].release()
                
                # Si hay un archivo de video creado, optimizarlo para web
                if recording_info["current_filepath"] and os.path.exists(recording_info["current_filepath"]):
                    self._optimize_video_for_web(recording_info["current_filepath"], recording_info["fps"])
                
            except Exception as e:
                logger.error(f"Error al liberar writer: {str(e)}")
        
        # Marcar como inactivo pero mantener en el diccionario para estadísticas
        recording_info["active"] = False
        recording_info["end_time"] = time.time()
        
        logger.info(f"Grabación detenida para cámara {camera_id}")
    
    def record_frame(self, camera_id: str, frame, fps=None):
        """
        Graba un frame para la cámara especificada y devuelve información para detección
        
        Args:
            camera_id: Identificador único de la cámara
            frame: Imagen/frame capturado (numpy array)
            fps: Velocidad de frames por segundo (opcional)
        
        Returns:
            tuple: (success: bool, video_info: dict)
                - success: True si el frame fue grabado correctamente
                - video_info: Diccionario con información del video para usar en detección
        """
        if camera_id not in self.active_recordings or not self.active_recordings[camera_id]["active"]:
            return False, None
        
        recording_info = self.active_recordings[camera_id]
        current_time = time.time()
        
        # Actualizar FPS si se proporciona
        if fps is not None and 1.0 <= fps <= 60.0:
            recording_info["fps"] = fps
        
        # Verificar si necesitamos crear un nuevo segmento
        if (recording_info["current_writer"] is None or 
            current_time - recording_info["segment_start_time"] > self.segment_duration * 60):
            
            # Cerrar el writer anterior si existe
            if recording_info["current_writer"] is not None:
                try:
                    recording_info["current_writer"].release()
                    logger.info(f"Segmento completado para cámara {camera_id}: {recording_info['current_filepath']}")
                    
                    # Optimizar el video recién grabado para reproducción web
                    if recording_info["current_filepath"] and os.path.exists(recording_info["current_filepath"]):
                        self._optimize_video_for_web(recording_info["current_filepath"], recording_info["fps"])
                        
                except Exception as e:
                    logger.error(f"Error al finalizar segmento: {str(e)}")
            
            # Crear un nuevo archivo para el siguiente segmento
            try:
                new_filepath = self._create_segment_filepath(camera_id, recording_info["camera_name"])
                
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(new_filepath), exist_ok=True)
                
                # Obtener dimensiones del frame para el writer
                height, width = frame.shape[:2]
                
                # Crear nuevo writer usando el codec mp4v (será convertido después)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    new_filepath, 
                    fourcc, 
                    recording_info["fps"],  # Usar el FPS calculado de la cámara
                    (width, height)
                )
                
                if not writer.isOpened():
                    logger.error(f"No se pudo crear el video writer para {new_filepath}")
                    return False, None
                
                recording_info["current_writer"] = writer
                recording_info["current_filepath"] = new_filepath
                recording_info["segment_start_time"] = current_time
                recording_info["frames_recorded"] = 0
                
                logger.info(f"Nuevo segmento iniciado para cámara {camera_id}: {new_filepath} (FPS: {recording_info['fps']:.2f})")
            
            except Exception as e:
                logger.error(f"Error al crear nuevo segmento: {str(e)}")
                return False, None
        
        # Escribir el frame
        try:
            recording_info["current_writer"].write(frame)
            recording_info["frames_recorded"] += 1
            
            # Calcular tiempo dentro del segmento
            video_time = current_time - recording_info["segment_start_time"]
            
            # Construir ruta relativa para API
            relative_path = os.path.relpath(recording_info["current_filepath"], self.storage_path)
            
            # Devolver información de grabación para detección
            return True, {
                "path": relative_path,
                "time": video_time,
                "fps": recording_info["fps"],
                "camera_id": camera_id,
                "camera_name": recording_info["camera_name"]
            }
        except Exception as e:
            logger.error(f"Error al escribir frame: {str(e)}")
            return False, None
    
    def _optimize_video_for_web(self, video_path, fps=None):
        """
        Convierte un video grabado con codec mp4v a H.264 para compatibilidad web
        preservando el framerate original
        
        Args:
            video_path: Ruta al archivo de video
            fps: Framerate a usar (si None, se detectará del video)
        """
        try:
            import subprocess
            import os
            
            # Verificar si FFmpeg está disponible
            try:
                result = subprocess.run(
                    ["ffmpeg", "-version"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                if result.returncode != 0:
                    logger.warning("FFmpeg no está disponible, no se puede optimizar el video")
                    return False
            except FileNotFoundError:
                logger.warning("FFmpeg no está instalado, no se puede optimizar el video")
                return False
            
            # Si no se proporciona FPS, intentar detectarlo del video
            fps_value = fps
            if fps_value is None:
                try:
                    # Usar FFprobe para obtener FPS del video
                    probe_cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate",
                        "-of", "csv=p=0",
                        video_path
                    ]
                    
                    probe_result = subprocess.run(
                        probe_cmd, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    
                    if probe_result.returncode == 0 and probe_result.stdout.strip():
                        # El formato suele ser "num/den"
                        fps_raw = probe_result.stdout.strip()
                        if '/' in fps_raw:
                            num, den = fps_raw.split('/')
                            fps_value = float(num) / float(den)
                        else:
                            fps_value = float(fps_raw)
                        
                        # Validar que sea un valor razonable
                        if fps_value < 1.0 or fps_value > 60.0:
                            fps_value = 15.0  # Valor predeterminado seguro
                except Exception as e:
                    logger.warning(f"Error al detectar FPS: {str(e)}")
                    fps_value = 15.0  # Valor predeterminado seguro
            
            # Asegurar que tenemos un valor de FPS válido
            if not fps_value or fps_value < 1.0:
                fps_value = 15.0
            
            # Archivo temporal para la conversión
            temp_path = f"{video_path}.temp.mp4"
            
            # Comando de conversión con FFmpeg
            cmd = [
                "ffmpeg",
                "-i", video_path,           # Archivo de entrada
                "-c:v", "libx264",          # Codec H.264
                "-r", str(fps_value),       # Especificar framerate explícitamente
                "-profile:v", "baseline",   # Perfil compatible con web
                "-level", "3.0",            # Nivel compatible
                "-pix_fmt", "yuv420p",      # Formato de pixel compatible
                "-preset", "ultrafast",     # Máxima velocidad (prioridad)
                "-crf", "23",               # Calidad razonable
                "-movflags", "+faststart",  # Optimizar para web
                "-y",                       # Sobrescribir si existe
                temp_path                   # Archivo temporal
            ]
            
            # Ejecutar conversión
            logger.info(f"Optimizando video para web: {video_path} (FPS: {fps_value:.2f})")
            convert_result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            # Verificar resultado
            if convert_result.returncode == 0:
                # Verificar que el archivo temporal existe y tiene un tamaño razonable
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                    # Reemplazar archivo original
                    os.replace(temp_path, video_path)
                    logger.info(f"Video optimizado correctamente: {video_path}")
                    return True
                else:
                    logger.error(f"Archivo convertido no válido: {temp_path}")
                    return False
            else:
                logger.error(f"Error en la conversión de video: {convert_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error al optimizar video: {str(e)}")
            return False
    
    def _create_segment_filepath(self, camera_id: str, camera_name: str) -> str:
        """
        Crea una ruta de archivo para un nuevo segmento de grabación
        
        Args:
            camera_id: Identificador único de la cámara
            camera_name: Nombre descriptivo de la cámara
            
        Returns:
            str: Ruta completa al archivo de grabación
        """
        # Formato: recordings/camera_id/YYYY-MM-DD/HH/camera_name_YYYY-MM-DD_HH-MM-SS.mp4
        now = datetime.datetime.now()
        
        # Crear estructura de directorios
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%H")
        time_str = now.strftime("%H-%M-%S")
        
        # Sanitizar el nombre de la cámara para uso en nombres de archivo
        safe_camera_name = "".join(c if c.isalnum() else "_" for c in camera_name)
        
        # Construir ruta
        relative_path = Path(camera_id) / date_str / hour_str
        filename = f"{safe_camera_name}_{date_str}_{time_str}.mp4"
        
        return str(self.storage_path / relative_path / filename)
    
    def get_recording_status(self, camera_id: str) -> Optional[dict]:
        """
        Obtiene el estado actual de grabación para una cámara
        
        Args:
            camera_id: Identificador único de la cámara
            
        Returns:
            dict: Estado de grabación, o None si no está activa
        """
        if camera_id not in self.active_recordings:
            return None
            
        recording_info = self.active_recordings[camera_id]
        if not recording_info["active"]:
            return None
            
        current_time = time.time()
        return {
            "active": True,
            "duration": int(current_time - recording_info["start_time"]),
            "segment_duration": int(current_time - recording_info["segment_start_time"]),
            "frames_recorded": recording_info["frames_recorded"],
            "current_fps": recording_info["fps"],
            "file": os.path.basename(recording_info["current_filepath"]) if recording_info["current_filepath"] else None
        }
    
    def list_recordings(self, camera_id: Optional[str] = None, date: Optional[str] = None, hour: Optional[str] = None) -> List[dict]:
        """
        Lista las grabaciones disponibles con filtrado opcional
        
        Args:
            camera_id: Filtrar por ID de cámara (opcional)
            date: Filtrar por fecha específica YYYY-MM-DD (opcional)
            hour: Filtrar por hora específica HH (opcional)
            
        Returns:
            List[dict]: Lista de objetos con información de las grabaciones
        """
        recordings = []
        
        # Determinar el directorio base según los filtros
        base_path = self.storage_path
        if camera_id:
            base_path = base_path / camera_id
            if date:
                base_path = base_path / date
                if hour:
                    base_path = base_path / hour
        
        # Si el directorio no existe, devolver lista vacía
        if not base_path.exists():
            return []
        
        # Recorrer los directorios según la estructura
        if camera_id and date and hour:
            # Caso específico: listar sólo los archivos de una hora concreta
            for video_file in base_path.glob("*.mp4"):
                recordings.append(self._get_recording_info(video_file, camera_id, date, hour))
        elif camera_id and date:
            # Caso: listar todas las horas de un día para una cámara
            for hour_dir in base_path.glob("*"):
                if hour_dir.is_dir():
                    for video_file in hour_dir.glob("*.mp4"):
                        recordings.append(self._get_recording_info(video_file, camera_id, date, hour_dir.name))
        elif camera_id:
            # Caso: listar todos los días para una cámara
            for date_dir in base_path.glob("*"):
                if date_dir.is_dir():
                    for hour_dir in date_dir.glob("*"):
                        if hour_dir.is_dir():
                            for video_file in hour_dir.glob("*.mp4"):
                                recordings.append(self._get_recording_info(video_file, camera_id, date_dir.name, hour_dir.name))
        else:
            # Caso general: listar todo
            for cam_dir in base_path.glob("*"):
                if cam_dir.is_dir():
                    cam_id = cam_dir.name
                    for date_dir in cam_dir.glob("*"):
                        if date_dir.is_dir():
                            for hour_dir in date_dir.glob("*"):
                                if hour_dir.is_dir():
                                    for video_file in hour_dir.glob("*.mp4"):
                                        recordings.append(self._get_recording_info(
                                            video_file, cam_id, date_dir.name, hour_dir.name))
        
        # Ordenar por fecha y hora, más recientes primero
        recordings.sort(key=lambda x: (x["date"], x["hour"], x["timestamp"]), reverse=True)
        return recordings
    
    def _get_recording_info(self, video_path: Path, camera_id: str, date: str, hour: str) -> dict:
        """
        Extrae información relevante de un archivo de grabación
        
        Args:
            video_path: Ruta al archivo de video
            camera_id: ID de la cámara
            date: Fecha del video (YYYY-MM-DD)
            hour: Hora del video (HH)
            
        Returns:
            dict: Información sobre el video
        """
        # Obtener estadísticas del archivo
        stat = video_path.stat()
        
        # Extraer timestamp y duración del nombre
        filename = video_path.name
        timestamp = None
        try:
            # El timestamp está en el nombre del archivo (por ejemplo, Camera_2023-05-10_14-30-00.mp4)
            time_part = filename.split('_')[-1].replace('.mp4', '')
            timestamp = f"{date}T{time_part.replace('-', ':')}"
        except:
            timestamp = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Construir la ruta relativa para URLs
        relative_path = str(video_path.relative_to(self.storage_path))
        
        # Tratar de obtener duración y resolución del video
        duration, width, height = self._get_video_metadata(video_path)
        
        # Crear objeto de información
        return {
            "id": video_path.stem,
            "camera_id": camera_id,
            "filename": filename,
            "path": relative_path,
            "date": date,
            "hour": hour,
            "timestamp": timestamp,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "duration_seconds": duration,
            "resolution": f"{width}x{height}" if width and height else "Unknown"
        }
    
    def _get_video_metadata(self, video_path: Path) -> tuple:
        """
        Obtiene metadatos del video como duración y resolución
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            tuple: (duration_seconds, width, height)
        """
        try:
            # Intentar primero con FFprobe (más rápido y preciso)
            try:
                cmd = [
                    "ffprobe", 
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,duration",
                    "-of", "csv=s=,:p=0",
                    str(video_path)
                ]
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 3:
                        width, height, duration = parts
                        return float(duration), int(width), int(height)
            except:
                pass
                
            # Si FFprobe no está disponible o falla, intentar con OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None, None, None
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return duration, width, height
        except Exception as e:
            logger.warning(f"Error al obtener metadatos del video {video_path}: {str(e)}")
            return None, None, None
            
    def get_recording_path(self, relative_path: str) -> Optional[str]:
        """
        Obtiene la ruta completa a un archivo de grabación
        
        Args:
            relative_path: Ruta relativa al directorio de grabaciones
            
        Returns:
            str: Ruta completa al archivo, o None si no existe
        """
        full_path = self.storage_path / relative_path
        
        if full_path.exists() and full_path.is_file():
            return str(full_path)
        return None
    
    def cleanup_old_recordings(self) -> int:
        """
        Elimina grabaciones más antiguas que el período de retención
        
        Returns:
            int: Número de archivos eliminados
        """
        # Calcular la fecha límite
        retention_limit = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
        deleted_count = 0
        
        try:
            # Recorrer todos los directorios de cámaras
            for camera_dir in self.storage_path.glob("*"):
                if not camera_dir.is_dir():
                    continue
                    
                # Recorrer directorios de fechas
                for date_dir in camera_dir.glob("*"):
                    if not date_dir.is_dir():
                        continue
                        
                    try:
                        # Convertir nombre de directorio a objeto datetime
                        dir_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d")
                        
                        # Si es más antiguo que el límite, eliminar todo el directorio
                        if dir_date < retention_limit:
                            # Eliminar todos los archivos y subdirectorios
                            for item in date_dir.glob("**/*"):
                                if item.is_file():
                                    item.unlink()
                                    deleted_count += 1
                            
                            # Eliminar directorios vacíos
                            for hour_dir in date_dir.glob("*"):
                                if hour_dir.is_dir():
                                    try:
                                        hour_dir.rmdir()  # Sólo se elimina si está vacío
                                    except:
                                        pass
                                        
                            # Intentar eliminar el directorio de fecha
                            try:
                                date_dir.rmdir()  # Sólo se elimina si está vacío
                                logger.info(f"Eliminado directorio antiguo: {date_dir}")
                            except:
                                logger.warning(f"No se pudo eliminar directorio: {date_dir}")
                    except ValueError:
                        # El nombre del directorio no sigue el formato esperado
                        logger.warning(f"Formato de directorio inesperado: {date_dir}")
                        continue
            
            logger.info(f"Limpieza completada: {deleted_count} archivos eliminados")
            return deleted_count
        
        except Exception as e:
            logger.error(f"Error durante la limpieza de grabaciones: {str(e)}")
            return 0
    
    def get_storage_usage(self) -> dict:
        """
        Obtiene información de uso de almacenamiento
        
        Returns:
            dict: Estadísticas de almacenamiento
        """
        total_size = 0
        file_count = 0
        oldest_date = None
        newest_date = None
        camera_sizes = {}
        
        try:
            # Recorrer directorios y calcular estadísticas
            for camera_dir in self.storage_path.glob("*"):
                if not camera_dir.is_dir():
                    continue
                    
                camera_id = camera_dir.name
                camera_sizes[camera_id] = {"size_bytes": 0, "file_count": 0}
                
                for date_dir in camera_dir.glob("*"):
                    if not date_dir.is_dir():
                        continue
                        
                    # Intentar parsear la fecha
                    try:
                        dir_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                        if oldest_date is None or dir_date < oldest_date:
                            oldest_date = dir_date
                        if newest_date is None or dir_date > newest_date:
                            newest_date = dir_date
                    except:
                        pass
                    
                    # Recorrer todos los archivos
                    for file_path in date_dir.glob("**/*.mp4"):
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        file_count += 1
                        camera_sizes[camera_id]["size_bytes"] += file_size
                        camera_sizes[camera_id]["file_count"] += 1
            
            # Convertir tamaños a más legibles
            for camera_id in camera_sizes:
                camera_sizes[camera_id]["size_mb"] = round(camera_sizes[camera_id]["size_bytes"] / (1024 * 1024), 2)
                camera_sizes[camera_id]["size_gb"] = round(camera_sizes[camera_id]["size_bytes"] / (1024 * 1024 * 1024), 2)
            
            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2),
                "file_count": file_count,
                "oldest_date": oldest_date.isoformat() if oldest_date else None,
                "newest_date": newest_date.isoformat() if newest_date else None,
                "retention_days": self.retention_days,
                "cameras": camera_sizes
            }
        except Exception as e:
            logger.error(f"Error al calcular uso de almacenamiento: {str(e)}")
            return {"error": str(e)}
