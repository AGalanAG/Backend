# auth_db.py - Gestiona la autenticación de usuarios y operaciones de base de datos
import sqlite3
import bcrypt
import datetime
import logging
import os
from typing import Dict, Optional, List

# Constantes para roles de usuario
ROLE_VIEWER = 1    # Solo puede ver cámaras
ROLE_OPERATOR = 2  # Puede ver cámaras, grabaciones y búsqueda
ROLE_ADMIN = 3     # Acceso completo, puede gestionar usuarios

class AuthDB:
    def __init__(self, db_path="./data/users.db"):
        self.db_path = db_path
        
        # Asegurar que el directorio existe
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        self._create_tables_if_not_exist()
        self._create_default_admin()
        
    def _create_tables_if_not_exist(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role INTEGER NOT NULL,
            created_at DATETIME NOT NULL,
            last_login DATETIME
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_default_admin(self):
        # Verificar si el admin existe
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_exists = cursor.fetchone()[0] > 0
        
        if not admin_exists:
            # Crear admin predeterminado con contraseña 'admin'
            hashed_password = bcrypt.hashpw('admin'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            now = datetime.datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                ('admin', hashed_password, ROLE_ADMIN, now)
            )
            
            conn.commit()
            logging.info("Usuario admin predeterminado creado")
        
        conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            # Actualizar último login
            now = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user['id']))
            conn.commit()
            
            # Convertir a dict para devolver
            user_dict = dict(user)
            user_dict.pop('password_hash')  # No devolver el hash de la contraseña
            
            conn.close()
            return user_dict
        
        conn.close()
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        conn.close()
        
        if user:
            user_dict = dict(user)
            user_dict.pop('password_hash')  # No devolver el hash de la contraseña
            return user_dict
        
        return None
    
    def list_users(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, role, created_at, last_login FROM users")
        users = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return users
    
    def create_user(self, username: str, password: str, role: int) -> Optional[Dict]:
        if role not in [ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN]:
            raise ValueError("Rol inválido")
            
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Generar hash de la contraseña
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            now = datetime.datetime.now().isoformat()
            
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, hashed_password, role, now)
            )
            
            conn.commit()
            
            # Obtener el usuario creado
            cursor.execute("SELECT id, username, role, created_at FROM users WHERE id = ?", (cursor.lastrowid,))
            user = dict(cursor.fetchone())
            
            conn.close()
            return user
        except sqlite3.IntegrityError:
            # Nombre de usuario ya existe
            conn.close()
            return None
    
    def update_user(self, user_id: int, username: Optional[str] = None, 
                   password: Optional[str] = None, role: Optional[int] = None) -> bool:
        if role is not None and role not in [ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN]:
            raise ValueError("Rol inválido")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Construir la consulta de actualización basada en los campos proporcionados
        update_parts = []
        params = []
        
        if username is not None:
            update_parts.append("username = ?")
            params.append(username)
        
        if password is not None:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            update_parts.append("password_hash = ?")
            params.append(hashed_password)
        
        if role is not None:
            update_parts.append("role = ?")
            params.append(role)
        
        if not update_parts:
            conn.close()
            return False  # Nada que actualizar
        
        # Añadir user_id a los parámetros
        params.append(user_id)
        
        query = f"UPDATE users SET {', '.join(update_parts)} WHERE id = ?"
        cursor.execute(query, params)
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_user(self, user_id: int) -> bool:
        # No permitir eliminar al usuario admin
        if user_id == 1:
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success