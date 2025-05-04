# clothing_endpoints.py - Router para endpoints de detección de vestimenta
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from auth_middleware import viewer_required, operator_required, admin_required

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic para validación
class ClothingSearchParams(BaseModel):
    camera_id: Optional[str] = None
    clothing_types: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    confidence: Optional[float] = 0.5
    limit: Optional[int] = 100

class ClothingDetection(BaseModel):
    id: int
    camera_id: str
    timestamp: str
    video_path: str
    video_time_seconds: float
    clothing_type: str
    color: str
    confidence: float
    bbox: List[int]

# Crear router para los endpoints de vestimenta
router = APIRouter(
    prefix="/api/clothing",
    tags=["clothing"],
    responses={404: {"description": "No encontrado"}},
)

# Variable global para almacenar el servicio de detección
# Será asignada desde app.py cuando se inicie
detection_service = None
detection_db = None

# Configuración del router
def configure(service, db):
    """
    Configura el router con las dependencias necesarias.
    
    Args:
        service: Instancia de DetectionService
        db: Instancia de DetectionDB
    """
    global detection_service, detection_db
    detection_service = service
    detection_db = db
    logger.info("Router de detección de vestimenta configurado")

# Endpoints
@router.get("/stats", response_model=Dict[str, Any], dependencies=[Depends(operator_required)])
async def get_detection_stats():
    """Obtener estadísticas del sistema de detección"""
    if not detection_service or not detection_db:
        raise HTTPException(status_code=500, detail="Servicio de detección no configurado")
    
    # Combinar estadísticas del servicio y la base de datos
    service_stats = detection_service.get_stats()
    db_stats = detection_db.get_db_stats()
    
    return {
        "service": service_stats,
        "database": db_stats
    }

@router.get("/available", dependencies=[Depends(viewer_required)])
async def get_available_classifications():
    """Obtener tipos de vestimenta y colores disponibles para búsqueda"""
    if not detection_service:
        raise HTTPException(status_code=500, detail="Servicio de detección no configurado")
    
    try:
        clothing_types = detection_service.detector.get_available_classes()
        colors = detection_service.detector.get_available_colors()
        
        return {
            "clothing_types": clothing_types,
            "colors": colors
        }
    except Exception as e:
        logger.error(f"Error al obtener clasificaciones disponibles: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/search", dependencies=[Depends(operator_required)])
async def search_clothing(
    camera_id: Optional[str] = None,
    clothing_type: Optional[str] = Query(None, description="Tipo de vestimenta"),
    color: Optional[str] = Query(None, description="Color de la prenda"),
    start_date: Optional[str] = Query(None, description="Fecha de inicio (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Fecha de fin (YYYY-MM-DD)"),
    confidence: float = Query(0.5, description="Confianza mínima (0.0-1.0)"),
    limit: int = Query(100, description="Límite de resultados")
):
    """Buscar detecciones de vestimenta con filtros"""
    if not detection_db:
        raise HTTPException(status_code=500, detail="Base de datos de detección no configurada")
    
    try:
        # Convertir clothing_type y color a listas si no son None
        clothing_types = [clothing_type] if clothing_type else None
        colors = [color] if color else None
        
        # Realizar búsqueda
        results = detection_db.search_detections(
            camera_id=camera_id,
            clothing_types=clothing_types,
            colors=colors,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=confidence,
            limit=limit
        )
        
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error en búsqueda de vestimenta: {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

@router.get("/video/{video_path:path}/timeline", dependencies=[Depends(operator_required)])
async def get_video_timeline(video_path: str):
    """Obtener línea temporal de detecciones para un video específico"""
    if not detection_db:
        raise HTTPException(status_code=500, detail="Base de datos de detección no configurada")
    
    try:
        # Asegurarnos de manejar correctamente la ruta
        if not video_path.startswith("recordings/"):
            normalized_path = f"recordings/{video_path}"
        else:
            normalized_path = video_path
            video_path = video_path.replace("recordings/", "")
        
        # Depuración para verificar la ruta que se está consultando
        logger.info(f"Consultando marcadores para: {video_path}")
        
        # Intentar primero con la ruta original
        timeline = detection_db.get_detection_timeline(video_path)
        
        # Si no hay resultados, intentar con la ruta normalizada 
        if not timeline and video_path != normalized_path:
            logger.info(f"Intento con ruta alternativa: {normalized_path}")
            timeline = detection_db.get_detection_timeline(normalized_path)
            
        # Imprimir para depuración
        logger.info(f"Encontrados {len(timeline)} marcadores")
        
        # Devolver datos formateados para UI
        ui_timeline = []
        for point in timeline:
            # Resumir detecciones para cada punto de tiempo
            summary = {}
            for detection in point["detections"]:
                clothing_type = detection["clothing_type"]
                color = detection["color"]
                
                # Agrupar por tipo y color
                key = f"{clothing_type}_{color}"
                if key not in summary:
                    summary[key] = {
                        "type": clothing_type,
                        "color": color,
                        "count": 0,
                        "confidence": 0
                    }
                
                summary[key]["count"] += 1
                summary[key]["confidence"] = max(summary[key]["confidence"], detection["confidence"])
            
            ui_timeline.append({
                "time": point["time"],
                "summary": list(summary.values()),
                "raw_detections": point["detections"]
            })
        
        return {
            "video_path": video_path,
            "markers_count": len(ui_timeline),
            "timeline": ui_timeline
        }
    except Exception as e:
        logger.error(f"Error al obtener línea temporal: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener línea temporal: {str(e)}")

@router.post("/config", dependencies=[Depends(admin_required)])
async def update_detection_config(
    enabled: Optional[bool] = None,
    interval: Optional[float] = None,
    min_confidence: Optional[float] = None
):
    """Actualizar configuración del servicio de detección"""
    if not detection_service:
        raise HTTPException(status_code=500, detail="Servicio de detección no configurado")
    
    config_changed = False
    
    try:
        if enabled is not None:
            if enabled:
                detection_service.enable()
            else:
                detection_service.disable()
            config_changed = True
        
        if interval is not None:
            detection_service.set_detection_interval(interval)
            config_changed = True
        
        if min_confidence is not None:
            detection_service.set_min_confidence(min_confidence)
            config_changed = True
        
        if not config_changed:
            return {"message": "No se proporcionaron cambios de configuración"}
        
        # Devolver configuración actualizada
        return {
            "message": "Configuración actualizada",
            "config": {
                "enabled": detection_service.enabled,
                "detection_interval": detection_service.detection_interval,
                "min_confidence": detection_service.min_confidence
            }
        }
    except Exception as e:
        logger.error(f"Error al actualizar configuración: {e}")
        raise HTTPException(status_code=500, detail=f"Error al actualizar configuración: {str(e)}")

@router.get("/counts", dependencies=[Depends(operator_required)])
async def get_detection_counts(camera_id: Optional[str] = None):
    """Obtener recuentos de detecciones agrupados por tipo y color"""
    if not detection_db:
        raise HTTPException(status_code=500, detail="Base de datos de detección no configurada")
    
    try:
        counts = detection_db.get_detection_counts(camera_id)
        return counts
    except Exception as e:
        logger.error(f"Error al obtener recuentos: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener recuentos: {str(e)}")
