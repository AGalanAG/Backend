# detection_db.py - Módulo para gestionar la base de datos de detecciones de vestimenta
import sqlite3
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class DetectionDB:
    """
    Base de datos para almacenar detecciones de vestimenta con marcas temporales.
    Usa SQLite por simplicidad y rendimiento.
    """
    
    def __init__(self, db_path="./data/detections.db"):
        """
        Inicializar la base de datos de detecciones.
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        # Asegurar que el directorio existe
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self.db_path = db_path
        self._create_tables_if_not_exist()
    
    def _create_tables_if_not_exist(self):
        """Crear tablas de base de datos si no existen."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crear tabla para detecciones de vestimenta
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS clothing_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                video_path TEXT NOT NULL,
                video_time_seconds REAL NOT NULL,
                clothing_type TEXT NOT NULL,
                color TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT NOT NULL
            )
            ''')
            
            # Crear índices para búsquedas más rápidas
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_camera_type_color 
            ON clothing_detections (camera_id, clothing_type, color)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
            ON clothing_detections (timestamp)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_video_path
            ON clothing_detections (video_path)
            ''')
            
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_video_time
            ON clothing_detections (video_path, video_time_seconds)
            ''')
            
            conn.commit()
            logger.info("Base de datos de detecciones inicializada")
            
        except sqlite3.Error as e:
            logger.error(f"Error de SQLite: {e}")
        finally:
            if conn:
                conn.close()
    
    def add_detection(self, camera_id: str, video_path: str, video_time_seconds: float, detection: Dict[str, Any]) -> bool:
        """
        Agregar una detección de vestimenta a la base de datos.
        
        Args:
            camera_id: ID de la cámara
            video_path: Ruta al archivo de video
            video_time_seconds: Tiempo en segundos dentro del video
            detection: Dict con detalles de detección (type, color, confidence, bbox)
        
        Returns:
            True si exitoso, False en caso contrario
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Marca de tiempo actual
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Serializar cuadro delimitador a JSON
            bbox_json = json.dumps(detection['bbox'])
            
            # Insertar la detección
            cursor.execute('''
            INSERT INTO clothing_detections 
            (camera_id, timestamp, video_path, video_time_seconds, clothing_type, color, confidence, bbox)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                camera_id,
                timestamp,
                video_path,
                video_time_seconds,
                detection['type'],
                detection['color'],
                detection['confidence'],
                bbox_json
            ))
            
            conn.commit()
            logger.debug(f"Detección agregada: {camera_id}, {detection['type']}, {detection['color']}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al agregar detección: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def search_detections(self, 
                        camera_id: Optional[str] = None, 
                        clothing_types: Optional[List[str]] = None, 
                        colors: Optional[List[str]] = None, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None, 
                        confidence_threshold: float = 0.5,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Buscar detecciones de vestimenta con varios filtros.
        
        Args:
            camera_id: ID de cámara opcional para filtrar
            clothing_types: Lista opcional de tipos de vestimenta para buscar
            colors: Lista opcional de colores para buscar
            start_date: Fecha de inicio opcional para rango de búsqueda
            end_date: Fecha de fin opcional para rango de búsqueda
            confidence_threshold: Nivel mínimo de confianza
            limit: Número máximo de resultados a devolver
            
        Returns:
            Lista de registros de detección que coinciden con los criterios
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Devolver resultados como diccionarios
            cursor = conn.cursor()
            
            # Construir consulta con filtros
            query = '''
            SELECT id, camera_id, timestamp, video_path, video_time_seconds, 
                   clothing_type, color, confidence, bbox
            FROM clothing_detections
            WHERE confidence >= ?
            '''
            params = [confidence_threshold]
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            if clothing_types and len(clothing_types) > 0:
                placeholders = ','.join(['?'] * len(clothing_types))
                query += f" AND clothing_type IN ({placeholders})"
                params.extend(clothing_types)
            
            if colors and len(colors) > 0:
                placeholders = ','.join(['?'] * len(colors))
                query += f" AND color IN ({placeholders})"
                params.extend(colors)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            # Ordenar por marca de tiempo, más reciente primero
            query += " ORDER BY timestamp DESC"
            
            # Limitar número de resultados
            query += f" LIMIT {limit}"
            
            # Ejecutar consulta
            cursor.execute(query, params)
            
            # Procesar resultados
            results = []
            for row in cursor.fetchall():
                # Convertir fila a dict y analizar JSON de bbox
                detection = dict(row)
                detection['bbox'] = json.loads(detection['bbox'])
                results.append(detection)
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al buscar detecciones: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_detection_counts(self, camera_id: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """
        Obtener recuentos de detecciones por tipo de vestimenta y color.
        
        Args:
            camera_id: ID de cámara opcional para filtrar
            
        Returns:
            Dict con recuentos por tipo y color
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Construir consulta
            type_query = '''
            SELECT clothing_type, COUNT(*) as count
            FROM clothing_detections
            '''
            
            color_query = '''
            SELECT color, COUNT(*) as count
            FROM clothing_detections
            '''
            
            params = []
            if camera_id:
                type_query += " WHERE camera_id = ?"
                color_query += " WHERE camera_id = ?"
                params.append(camera_id)
            
            type_query += " GROUP BY clothing_type ORDER BY count DESC"
            color_query += " GROUP BY color ORDER BY count DESC"
            
            # Ejecutar consultas
            cursor.execute(type_query, params)
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute(color_query, params)
            color_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "by_type": type_counts,
                "by_color": color_counts
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al obtener recuentos de detección: {e}")
            return {"by_type": {}, "by_color": {}}
        finally:
            if conn:
                conn.close()
    
    def get_detections_for_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Obtener todas las detecciones para un archivo de video específico.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            Lista de detecciones para el video
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Consulta para detecciones de este video
            cursor.execute('''
            SELECT id, camera_id, timestamp, video_path, video_time_seconds, 
                   clothing_type, color, confidence, bbox
            FROM clothing_detections
            WHERE video_path = ?
            ORDER BY video_time_seconds
            ''', (video_path,))
            
            # Procesar resultados
            results = []
            for row in cursor.fetchall():
                detection = dict(row)
                detection['bbox'] = json.loads(detection['bbox'])
                results.append(detection)
            
            return results
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al obtener detecciones de video: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_detection_timeline(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Obtener línea temporal de detecciones para un video específico,
        ideal para mostrar marcadores en la línea de tiempo del reproductor.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            Lista de marcas de tiempo con detecciones agrupadas
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Probar variantes de la ruta para manejar inconsistencias
            possible_paths = [
                video_path,
                video_path.replace("recordings/", ""),
                f"recordings/{video_path}"
            ]
            
            # Consulta parametrizada con múltiples posibles rutas
            query = f"""
            SELECT DISTINCT video_time_seconds
            FROM clothing_detections
            WHERE video_path IN ({','.join(['?']*len(possible_paths))})
            ORDER BY video_time_seconds
            """
            
            cursor.execute(query, possible_paths)
            
            time_points = [row[0] for row in cursor.fetchall()]
            
            # Imprimir información de depuración
            logger.info(f"Consultando video: {video_path}")
            logger.info(f"Puntos de tiempo encontrados: {len(time_points)}")
            
            # Para cada punto de tiempo, obtener todas las detecciones
            timeline = []
            for time_point in time_points:
                # Consulta para todas las posibles rutas
                query = f"""
                SELECT id, clothing_type, color, confidence, bbox
                FROM clothing_detections
                WHERE video_path IN ({','.join(['?']*len(possible_paths))}) 
                    AND video_time_seconds = ?
                ORDER BY confidence DESC
                """
                
                params = possible_paths + [time_point]
                cursor.execute(query, params)
                
                detections = []
                for row in cursor.fetchall():
                    detection = dict(row)
                    detection['bbox'] = json.loads(detection['bbox'])
                    detections.append(detection)
                
                timeline.append({
                    "time": time_point,
                    "detections": detections
                })
            
            return timeline
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al obtener línea temporal: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def cleanup_old_detections(self, days_to_keep: int = 30) -> int:
        """
        Limpiar detecciones antiguas de la base de datos.
        
        Args:
            days_to_keep: Número de días de detecciones a conservar
            
        Returns:
            Número de detecciones eliminadas
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calcular fecha límite
            cutoff_date = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                          .timestamp() - (days_to_keep * 86400))
            cutoff_date_str = datetime.fromtimestamp(cutoff_date).strftime('%Y-%m-%d %H:%M:%S')
            
            # Eliminar detecciones antiguas
            cursor.execute('''
            DELETE FROM clothing_detections
            WHERE timestamp < ?
            ''', (cutoff_date_str,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Limpieza completada: {deleted_count} detecciones antiguas eliminadas")
            return deleted_count
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al limpiar detecciones antiguas: {e}")
            return 0
        finally:
            if conn:
                conn.close()
    
    def get_db_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de la base de datos.
        
        Returns:
            Dict con estadísticas
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener recuento total
            cursor.execute('SELECT COUNT(*) FROM clothing_detections')
            total_count = cursor.fetchone()[0]
            
            # Obtener fecha más antigua
            cursor.execute('SELECT MIN(timestamp) FROM clothing_detections')
            oldest_date = cursor.fetchone()[0]
            
            # Obtener fecha más reciente
            cursor.execute('SELECT MAX(timestamp) FROM clothing_detections')
            newest_date = cursor.fetchone()[0]
            
            # Obtener número de videos distintos
            cursor.execute('SELECT COUNT(DISTINCT video_path) FROM clothing_detections')
            videos_count = cursor.fetchone()[0]
            
            # Obtener número de cámaras distintas
            cursor.execute('SELECT COUNT(DISTINCT camera_id) FROM clothing_detections')
            cameras_count = cursor.fetchone()[0]
            
            # Obtener recuentos por tipo de vestimenta
            cursor.execute('''
            SELECT clothing_type, COUNT(*) as count
            FROM clothing_detections
            GROUP BY clothing_type
            ORDER BY count DESC
            ''')
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Obtener recuentos por color
            cursor.execute('''
            SELECT color, COUNT(*) as count
            FROM clothing_detections
            GROUP BY color
            ORDER BY count DESC
            ''')
            color_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "total_detections": total_count,
                "oldest_date": oldest_date,
                "newest_date": newest_date,
                "unique_videos": videos_count,
                "unique_cameras": cameras_count,
                "types": type_counts,
                "colors": color_counts
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error SQLite al obtener estadísticas: {e}")
            return {}
        finally:
            if conn:
                conn.close()
