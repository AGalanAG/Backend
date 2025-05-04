# auth_endpoints.py - Router FastAPI para endpoints de autenticación
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import datetime
from typing import List, Optional

from auth_db import AuthDB, ROLE_VIEWER, ROLE_OPERATOR, ROLE_ADMIN
from auth_utils import create_access_token
from auth_middleware import get_current_user, admin_required

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
)

auth_db = AuthDB()

class LoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    role: int
    created_at: str
    last_login: Optional[str] = None

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: int

class UpdateUserRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    role: Optional[int] = None

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = auth_db.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    
    access_token = create_access_token(data={"sub": str(user["id"])})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: dict = Depends(get_current_user)):
    return user

@router.get("/users", response_model=List[UserResponse])
async def list_users(user: dict = Depends(admin_required)):
    return auth_db.list_users()

@router.post("/users", response_model=UserResponse)
async def create_user(request: CreateUserRequest, user: dict = Depends(admin_required)):
    new_user = auth_db.create_user(request.username, request.password, request.role)
    if not new_user:
        raise HTTPException(status_code=400, detail="El nombre de usuario ya existe")
    return new_user

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, request: UpdateUserRequest, current_user: dict = Depends(admin_required)):
    # No permitir cambiar el rol del admin
    if user_id == 1 and request.role is not None and request.role != ROLE_ADMIN:
        raise HTTPException(status_code=400, detail="No se puede cambiar el rol del administrador")
    
    success = auth_db.update_user(user_id, request.username, request.password, request.role)
    if not success:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Obtener usuario actualizado
    updated_user = auth_db.get_user_by_id(user_id)
    return updated_user

@router.delete("/users/{user_id}")
async def delete_user(user_id: int, current_user: dict = Depends(admin_required)):
    # No permitir eliminar al admin
    if user_id == 1:
        raise HTTPException(status_code=400, detail="No se puede eliminar al usuario administrador")
    
    success = auth_db.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return {"success": True}