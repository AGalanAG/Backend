# clothing_detector.py - Módulo para detección de vestimenta en frames de video
import cv2
import numpy as np
import time
import logging
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ClothingDetector:
    """
    Detector de vestimenta usando modelo YOLOv8.
    
    Esta clase maneja:
    1. Carga del modelo YOLOv8
    2. Detección de prendas de vestir en frames
    3. Identificación de colores de las prendas detectadas
    4. Devolución de resultados estructurados
    """
    
    # Mapeo de IDs de clase YOLO a categorías de vestimenta
    # Clases de COCO (80 clases) relevantes para vestimenta
    CLOTHING_CLASSES = {
        0: "person",  # persona completa
        # Clases específicas de ropa en COCO
        24: "backpack",  # mochila
        25: "umbrella",  # paraguas (puede ser útil como complemento)
        27: "tie",       # corbata
        28: "suitcase",  # maleta
        32: "sports ball", # pelota (útil para contexto)
        33: "kite",      # cometa (útil para contexto exterior)
        43: "tennis racket", # raqueta (contexto)
        # Clases de persona
        1: "person-upper", # parte superior de persona
        2: "person-lower", # parte inferior de persona
    }
    
    # Rangos de identificación de colores básicos (HSV)
    COLOR_RANGES = {
        "red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (180, 255, 255)],
        "orange": [(10, 100, 100), (25, 255, 255)],
        "yellow": [(25, 100, 100), (35, 255, 255)],
        "green": [(35, 80, 80), (85, 255, 255)],
        "blue": [(85, 100, 100), (130, 255, 255)],
        "purple": [(130, 80, 80), (155, 255, 255)],
        "pink": [(155, 80, 80), (165, 255, 255)],
        "brown": [(10, 100, 60), (20, 255, 200)],
        "black": [(0, 0, 0), (180, 100, 50)],
        "white": [(0, 0, 200), (180, 30, 255)],
        "gray": [(0, 0, 70), (180, 30, 200)]
    }
    
    def __init__(self, model_path=None, confidence_threshold=0.4):
        """
        Inicializar el detector de vestimenta.
        
        Args:
            model_path: Ruta al archivo del modelo YOLOv8 (si es None, descargará directamente)
            confidence_threshold: Umbral mínimo de confianza para detecciones
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Primero, intentar cargar modelo desde ruta específica
            if model_path is not None:
                model_path = Path(model_path)
                if model_path.exists():
                    logger.info(f"Cargando modelo YOLOv8 desde archivo: {model_path}")
                    from ultralytics import YOLO
                    self.model = YOLO(str(model_path))
                else:
                    raise FileNotFoundError(f"Archivo de modelo no encontrado: {model_path}")
            else:
                # Si no hay ruta específica, cargar modelo directamente con YOLO
                logger.info("Descargando modelo YOLOv8n directamente...")
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')  # Esto descarga el modelo si es necesario
                
            logger.info(f"Modelo cargado exitosamente en {self.device}")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo YOLOv8: {e}")
            raise
    
    def detect_clothing(self, frame) -> List[Dict[str, Any]]:
        """
        Detectar prendas de vestir en un frame.
        
        Args:
            frame: Imagen/frame de OpenCV (numpy array)
            
        Returns:
            Lista de detecciones con tipo de vestimenta, color, confianza y cuadro delimitador
        """
        if self.model is None:
            logger.error("Modelo no cargado")
            return []
        
        try:
            # Realizar inferencia
            results = self.model(frame)
            
            # Procesar detecciones
            detections = []
            detected_persons = []
            
            # Las nuevas versiones de ultralytics devuelven los resultados de forma diferente
            # Primero, encontrar todas las personas
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy()  # Formato [x1, y1, x2, y2]
                    
                    # Extraer coordenadas
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    if cls_id == 0:  # Clase 'persona'
                        detected_persons.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": conf
                        })
            
            # Si hay personas, dividirlas en parte superior e inferior
            for person in detected_persons:
                x1, y1, x2, y2 = person["bbox"]
                confidence = person["confidence"]
                
                # Dividir la persona en parte superior e inferior
                mid_y = (y1 + y2) // 2
                
                # Región superior (camisa, chaqueta, etc.)
                upper_roi = frame[int(y1):int(mid_y), int(x1):int(x2)]
                if upper_roi.size > 0:
                    upper_color = self._identify_dominant_color(upper_roi)
                    detections.append({
                        "type": "upper_clothing",
                        "color": upper_color,
                        "confidence": float(confidence * 0.9),  # Reducir un poco la confianza
                        "bbox": [int(x1), int(y1), int(x2), int(mid_y)]
                    })
                
                # Región inferior (pantalones, falda, etc.)
                lower_roi = frame[int(mid_y):int(y2), int(x1):int(x2)]
                if lower_roi.size > 0:
                    lower_color = self._identify_dominant_color(lower_roi)
                    detections.append({
                        "type": "lower_clothing",
                        "color": lower_color,
                        "confidence": float(confidence * 0.85),  # Reducir un poco más la confianza
                        "bbox": [int(x1), int(mid_y), int(x2), int(y2)]
                    })
            
            # Luego, buscar objetos específicos de ropa 
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    if conf < self.confidence_threshold:
                        continue
                    
                    if cls_id in self.CLOTHING_CLASSES and cls_id != 0:  # Ignorar la clase 'persona' aquí
                        clothing_type = self.CLOTHING_CLASSES[cls_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # Extraer coordenadas
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Extraer la región para análisis de color
                        bbox = [x1, y1, x2, y2]
                        roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        
                        # Identificar color dominante
                        color = self._identify_dominant_color(roi)
                        
                        # Agregar detección
                        detections.append({
                            "type": clothing_type,
                            "color": color,
                            "confidence": float(conf),
                            "bbox": bbox
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error durante la detección: {e}")
            return []
    
    def _identify_dominant_color(self, roi):
        """
        Identificar el color dominante en la región de interés.
        
        Args:
            roi: Región de interés (recorte de prenda)
            
        Returns:
            Nombre del color dominante como string
        """
        try:
            # Si ROI está vacío o es muy pequeño, devolver "desconocido"
            if roi is None or roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                return "unknown"
            
            # Convertir a HSV para mejor análisis de color
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Obtener una máscara de píxeles que no son fondo (asumiendo que el fondo es mayormente uniforme)
            mask = self._create_foreground_mask(roi)
            
            # Aplicar la máscara
            masked_roi = cv2.bitwise_and(hsv_roi, hsv_roi, mask=mask)
            
            # Si no tenemos suficientes píxeles después del enmascaramiento, devolver desconocido
            if np.count_nonzero(mask) < 20:
                return "unknown"
            
            # Obtener histograma de matiz, saturación, valor
            hist_h = cv2.calcHist([masked_roi], [0], mask, [180], [0, 180])
            hist_s = cv2.calcHist([masked_roi], [1], mask, [256], [0, 256])
            hist_v = cv2.calcHist([masked_roi], [2], mask, [256], [0, 256])
            
            # Encontrar valores HSV dominantes
            h_max = np.argmax(hist_h)
            s_max = np.argmax(hist_s)
            v_max = np.argmax(hist_v)
            
            # Algoritmo simple para detección de negro/blanco/gris
            if v_max < 50:
                return "black"
            if v_max > 200 and s_max < 50:
                return "white"
            if s_max < 50:
                return "gray"
            
            # Verificar matiz dominante contra rangos de color
            hsv_value = (h_max, s_max, v_max)
            for color_name, ranges in self.COLOR_RANGES.items():
                for i in range(0, len(ranges), 2):
                    if i + 1 < len(ranges):  # Asegurar que hay un par superior
                        lower = ranges[i]
                        upper = ranges[i+1]
                        
                        # Verificar si el valor HSV está en este rango
                        if (lower[0] <= hsv_value[0] <= upper[0] and 
                            lower[1] <= hsv_value[1] <= upper[1] and
                            lower[2] <= hsv_value[2] <= upper[2]):
                            return color_name
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error en identificación de color: {e}")
            return "unknown"
    
    def _create_foreground_mask(self, bgr_roi):
        """
        Crear una máscara para separar primer plano (ropa) del fondo.
        
        Args:
            bgr_roi: Región de interés BGR
            
        Returns:
            Máscara binaria donde los píxeles de primer plano son 255 y el fondo 0
        """
        try:
            # Enfoque simple: los bordes tienden a ser parte del primer plano
            gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilatar bordes para obtener más del primer plano
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Rellenar huecos
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            
            # Si la máscara es muy pequeña, usar todo el ROI
            if np.sum(mask) / 255 < (mask.shape[0] * mask.shape[1] * 0.1):
                return np.ones_like(gray, dtype=np.uint8) * 255
                
            return mask
            
        except Exception as e:
            logger.error(f"Error al crear máscara de primer plano: {e}")
            # Devolver una máscara predeterminada (todos los píxeles como primer plano)
            return np.ones((bgr_roi.shape[0], bgr_roi.shape[1]), np.uint8) * 255

    def get_available_classes(self):
        """
        Obtener la lista de clases de vestimenta disponibles.
        
        Returns:
            Diccionario con clases disponibles
        """
        return self.CLOTHING_CLASSES
    
    def get_available_colors(self):
        """
        Obtener la lista de colores que el detector puede identificar.
        
        Returns:
            Lista de nombres de colores
        """
        return list(self.COLOR_RANGES.keys()) + ["unknown"]