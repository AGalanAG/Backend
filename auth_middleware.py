# auth_middleware.py - Middleware de autenticación para FastAPI
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Callable
import logging

from auth_utils import decode_token
from auth_db import AuthDB, ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN

security = HTTPBearer()
auth_db = AuthDB()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Payload de token inválido")
    
    user = auth_db.get_user_by_id(int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    
    return user

def role_required(min_role: int):
    def _role_required(user: dict = Depends(get_current_user)):
        if user["role"] < min_role:
            raise HTTPException(
                status_code=403,
                detail=f"Permisos insuficientes. Rol requerido: {min_role}, Rol de usuario: {user['role']}"
            )
        return user
    return _role_required

# Funciones de conveniencia para verificación de roles
def viewer_required(user: dict = Depends(get_current_user)):
    return user

def operator_required(user: dict = Depends(role_required(ROLE_OPERATOR))):
    return user

def admin_required(user: dict = Depends(role_required(ROLE_ADMIN))):
    return user