# detection_service.py - Servicio para procesar detecciones de vestimenta
import threading
import time
import logging
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from clothing_detector import ClothingDetector
from detection_db import DetectionDB

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DetectionService:
    """
    Servicio para procesar detecciones de vestimenta en tiempo real.
    Maneja la detección en segundo plano para no bloquear el stream principal.
    """
    
    def __init__(self, 
                detector: ClothingDetector,
                db: DetectionDB,
                detection_interval: float = 1.0,
                min_confidence: float = 0.4,
                enabled: bool = True):
        """
        Inicializar el servicio de detección.
        
        Args:
            detector: Instancia de ClothingDetector para realizar detecciones
            db: Instancia de DetectionDB para almacenar resultados
            detection_interval: Intervalo en segundos entre detecciones (para ahorrar recursos)
            min_confidence: Confianza mínima para almacenar detecciones
            enabled: Si el servicio está habilitado al inicio
        """
        self.detector = detector
        self.db = db
        self.detection_interval = detection_interval
        self.min_confidence = min_confidence
        self.enabled = enabled
        
        # Estado interno
        self.last_detection_time: Dict[str, float] = {}  # camera_id -> last_time
        self.processing_lock = threading.Lock()
        self.currently_processing: Dict[str, bool] = {}  # camera_id -> is_processing
        
        # Estadísticas
        self.stats = {
            "processed_frames": 0,
            "detected_items": 0,
            "detection_time_total": 0,
            "detection_time_avg": 0,
            "last_detection_timestamp": 0
        }
        
        logger.info(f"Servicio de detección inicializado (intervalo={detection_interval}s, "
                  f"confianza_min={min_confidence}, habilitado={enabled})")
    
    def process_frame(self, 
                     frame: np.ndarray, 
                     camera_id: str, 
                     video_path: str, 
                     video_time: float,
                     callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None) -> bool:
        """
        Procesar un frame para detección si ha pasado el intervalo.
        
        Args:
            frame: Frame de video a procesar
            camera_id: ID de la cámara
            video_path: Ruta al archivo de video
            video_time: Tiempo actual en segundos dentro del video
            callback: Función opcional a llamar con las detecciones (útil para UI)
            
        Returns:
            True si se procesó el frame, False en caso contrario
        """
        if not self.enabled:
            return False
        
        current_time = time.time()
        
        # Verificar si es momento de procesar este frame para esta cámara
        if (camera_id not in self.last_detection_time or 
            current_time - self.last_detection_time.get(camera_id, 0) >= self.detection_interval):
            
            # Verificar si ya estamos procesando un frame para esta cámara
            if self.currently_processing.get(camera_id, False):
                return False
                
            # Adquirir bloqueo para evitar procesamiento concurrente
            if self.processing_lock.acquire(blocking=False):
                try:
                    # Marcar como en procesamiento
                    self.currently_processing[camera_id] = True
                    
                    # Actualizar último tiempo de detección
                    self.last_detection_time[camera_id] = current_time
                    
                    # Iniciar detección en un hilo separado para evitar bloquear
                    detection_thread = threading.Thread(
                        target=self._detect_and_store,
                        args=(frame.copy(), camera_id, video_path, video_time, callback),
                        daemon=True
                    )
                    detection_thread.start()
                    
                    return True
                finally:
                    self.processing_lock.release()
        
        return False
    
    def _detect_and_store(self, 
                         frame: np.ndarray, 
                         camera_id: str, 
                         video_path: str, 
                         video_time: float,
                         callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None):
        """
        Realizar detección y almacenar resultados en base de datos.
        Esto se ejecuta en un hilo separado.
        
        Args:
            frame: Frame de video a procesar
            camera_id: ID de la cámara
            video_path: Ruta al archivo de video
            video_time: Tiempo actual en segundos dentro del video
            callback: Función opcional a llamar con las detecciones (útil para UI)
        """
        try:
            start_time = time.time()
            
            # Realizar detección
            detections = self.detector.detect_clothing(frame)
            
            # Filtrar por confianza mínima
            detections = [d for d in detections if d['confidence'] >= self.min_confidence]
            
            # Actualizar estadísticas
            detection_time = time.time() - start_time
            self.stats["processed_frames"] += 1
            self.stats["detected_items"] += len(detections)
            self.stats["detection_time_total"] += detection_time
            self.stats["detection_time_avg"] = (
                self.stats["detection_time_total"] / self.stats["processed_frames"]
            )
            self.stats["last_detection_timestamp"] = time.time()
            
            # Almacenar detecciones en base de datos
            for detection in detections:
                self.db.add_detection(
                    camera_id,
                    video_path,
                    video_time,
                    detection
                )
            
            if detections:
                logger.info(f"Cámara {camera_id}: Detectados {len(detections)} items de vestimenta en {video_time:.2f}s "
                           f"(procesamiento: {detection_time:.3f}s)")
                
                # Llamar al callback si existe
                if callback:
                    callback(detections)
                    
        except Exception as e:
            logger.error(f"Error en procesamiento de detección: {e}")
        finally:
            # Marcar como no en procesamiento
            self.currently_processing[camera_id] = False
    
    def enable(self):
        """Habilitar el servicio de detección."""
        self.enabled = True
        logger.info("Servicio de detección habilitado")
    
    def disable(self):
        """Deshabilitar el servicio de detección."""
        self.enabled = False
        logger.info("Servicio de detección deshabilitado")
    
    def set_detection_interval(self, interval: float):
        """
        Establecer el intervalo entre detecciones.
        
        Args:
            interval: Intervalo en segundos (mínimo 0.1)
        """
        self.detection_interval = max(0.1, interval)
        logger.info(f"Intervalo de detección actualizado a {self.detection_interval}s")
    
    def set_min_confidence(self, confidence: float):
        """
        Establecer la confianza mínima para detecciones.
        
        Args:
            confidence: Valor de confianza (0.0-1.0)
        """
        self.min_confidence = max(0.1, min(1.0, confidence))
        logger.info(f"Confianza mínima actualizada a {self.min_confidence}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del servicio de detección.
        
        Returns:
            Dict con estadísticas
        """
        return {
            "enabled": self.enabled,
            "detection_interval": self.detection_interval,
            "min_confidence": self.min_confidence,
            "processed_frames": self.stats["processed_frames"],
            "detected_items": self.stats["detected_items"],
            "detection_time_avg": round(self.stats["detection_time_avg"], 3) if self.stats["processed_frames"] > 0 else 0,
            "last_detection_ago": round(time.time() - self.stats["last_detection_timestamp"], 1) if self.stats["last_detection_timestamp"] > 0 else -1,
            "cameras_processed": len(self.last_detection_time)
        }
