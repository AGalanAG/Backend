import os
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/video",
    tags=["video"],
    responses={404: {"description": "No encontrado"}},
)

@router.get("/info/{path:path}")
async def get_video_info(path: str):
    """Obtiene información básica sobre un archivo de video"""
    try:
        # Ejemplo: usar la ruta relativa a 'recordings'
        full_path = os.path.join("recordings", path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
            
        # Obtener información básica del archivo
        stat_info = os.stat(full_path)
        
        return {
            "filename": os.path.basename(full_path),
            "size_bytes": stat_info.st_size,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "path": path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener información del video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar solicitud: {str(e)}")

@router.get("/thumbnail/{path:path}")
async def get_video_thumbnail(path: str):
    """Endpoint para generar y obtener una miniatura del video"""
    # Esta implementación es un placeholder
    # En una implementación real, se generaría una miniatura del video
    
    try:
        full_path = os.path.join("recordings", path)
        
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Archivo de video no encontrado")
            
        # Aquí iría el código para generar una miniatura del video
        # Por ahora retornamos un error informativo
        
        raise HTTPException(
            status_code=501, 
            detail="Generación de miniaturas no implementada todavía"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar miniatura: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar solicitud: {str(e)}")
