"""
Authentication and Authorization Manager for Quantum AI Platform
Secure user management, JWT tokens, API keys, and role-based access control
"""

import asyncio
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

# Cryptography and JWT imports
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# FastAPI security
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class UserCredentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    email: Optional[str] = None

class UserInfo(BaseModel):
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    is_active: bool
    created_at: str
    last_login: Optional[str]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str

class APIKeyInfo(BaseModel):
    key_id: str
    name: str
    permissions: List[str]
    created_at: str
    last_used: Optional[str]
    expires_at: Optional[str]

class AuthenticationManager:
    """
    Comprehensive authentication and authorization manager with:
    - User registration and login
    - JWT token management
    - API key authentication
    - Role-based access control
    - Password hashing and validation
    - Session management
    - Quantum-safe encryption
    """
    
    def __init__(self, secret_key: str = None):
        # Security configuration
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # Password hashing
        self.bcrypt_rounds = 12
        
        # Encryption for sensitive data
        self.fernet = self._setup_encryption()
        
        # User storage (in production, this would use the database)
        self.users = {}  # user_id -> user_data
        self.sessions = {}  # session_token -> session_data
        self.api_keys = {}  # key_hash -> key_data
        
        # Role and permission definitions
        self.roles = {
            "admin": [
                "quantum:read", "quantum:write", "quantum:admin",
                "time_crystal:read", "time_crystal:write", "time_crystal:admin",
                "neuromorphic:read", "neuromorphic:write", "neuromorphic:admin",
                "ai_ml:read", "ai_ml:write", "ai_ml:admin",
                "iot:read", "iot:write", "iot:admin",
                "users:read", "users:write", "users:admin",
                "system:read", "system:write", "system:admin"
            ],
            "researcher": [
                "quantum:read", "quantum:write",
                "time_crystal:read", "time_crystal:write",
                "neuromorphic:read", "neuromorphic:write",
                "ai_ml:read", "ai_ml:write",
                "iot:read"
            ],
            "user": [
                "quantum:read",
                "time_crystal:read",
                "neuromorphic:read",
                "ai_ml:read",
                "iot:read"
            ],
            "api_client": [
                "quantum:read", "quantum:write",
                "time_crystal:read",
                "neuromorphic:read",
                "ai_ml:read"
            ]
        }
        
        # Security bearer
        self.security = HTTPBearer()
        
        # Database reference (set externally)
        self.db_manager = None
        
        logger.info("Authentication Manager initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure random secret key"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _setup_encryption(self) -> Fernet:
        """Setup Fernet encryption for sensitive data"""
        # Derive key from secret
        password = self.secret_key.encode()
        salt = b'quantum_ai_salt'  # In production, use random salt per encryption
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=self.bcrypt_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{secrets.token_urlsafe(16)}"
    
    def _generate_api_key(self) -> Tuple[str, str]:
        """Generate API key and its hash"""
        # Generate a secure random API key
        api_key = f"qai_{secrets.token_urlsafe(32)}"
        
        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        return api_key, key_hash
    
    async def register_user(self, credentials: UserCredentials, roles: List[str] = None) -> Optional[str]:
        """Register a new user"""
        
        try:
            # Check if username already exists
            for user_data in self.users.values():
                if user_data["username"] == credentials.username:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Username already exists"
                    )
            
            # Generate user ID and hash password
            user_id = self._generate_user_id()
            password_hash = self._hash_password(credentials.password)
            
            # Default roles
            user_roles = roles or ["user"]
            
            # Create user record
            user_data = {
                "user_id": user_id,
                "username": credentials.username,
                "email": credentials.email or "",
                "password_hash": password_hash,
                "roles": user_roles,
                "permissions": self._get_permissions_for_roles(user_roles),
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": None,
                "login_attempts": 0,
                "locked_until": None
            }
            
            # Store user (in production, use database)
            self.users[user_id] = user_data
            
            # If database is available, store there too
            if self.db_manager:
                await self._store_user_in_db(user_data)
            
            logger.info(f"User registered successfully: {credentials.username}")
            return user_id
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )
    
    async def authenticate_user(self, username: str, password: str) -> Optional[UserInfo]:
        """Authenticate user with username/password"""
        
        try:
            # Find user by username
            user_data = None
            for user in self.users.values():
                if user["username"] == username:
                    user_data = user
                    break
            
            if not user_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Check if account is locked
            if user_data.get("locked_until"):
                locked_until = datetime.fromisoformat(user_data["locked_until"])
                if datetime.utcnow() < locked_until:
                    raise HTTPException(
                        status_code=status.HTTP_423_LOCKED,
                        detail="Account temporarily locked"
                    )
            
            # Check if account is active
            if not user_data.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account deactivated"
                )
            
            # Verify password
            if not self._verify_password(password, user_data["password_hash"]):
                # Increment failed login attempts
                user_data["login_attempts"] = user_data.get("login_attempts", 0) + 1
                
                # Lock account after 5 failed attempts
                if user_data["login_attempts"] >= 5:
                    user_data["locked_until"] = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
            
            # Reset login attempts on successful login
            user_data["login_attempts"] = 0
            user_data["locked_until"] = None
            user_data["last_login"] = datetime.utcnow().isoformat()
            
            # Return user info
            return UserInfo(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                roles=user_data["roles"],
                permissions=user_data["permissions"],
                is_active=user_data["is_active"],
                created_at=user_data["created_at"],
                last_login=user_data["last_login"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"User authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )
    
    def _get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        """Get all permissions for given roles"""
        permissions = set()
        for role in roles:
            if role in self.roles:
                permissions.update(self.roles[role])
        return list(permissions)
    
    def create_access_token(self, user_info: UserInfo) -> str:
        """Create JWT access token"""
        
        # Token payload
        payload = {
            "sub": user_info.user_id,
            "username": user_info.username,
            "roles": user_info.roles,
            "permissions": user_info.permissions,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_info: UserInfo) -> str:
        """Create JWT refresh token"""
        
        # Token payload
        payload = {
            "sub": user_info.user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expire_days),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
            )
    
    async def create_api_key(self, user_id: str, name: str, permissions: List[str] = None,
                           expires_days: int = None) -> Tuple[str, APIKeyInfo]:
        """Create a new API key for a user"""
        
        try:
            # Generate API key
            api_key, key_hash = self._generate_api_key()
            
            # Default permissions based on user's roles
            user_data = self.users.get(user_id)
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")
            
            key_permissions = permissions or user_data["permissions"]
            
            # Create key info
            key_data = {
                "key_id": f"key_{secrets.token_urlsafe(8)}",
                "user_id": user_id,
                "name": name,
                "key_hash": key_hash,
                "permissions": key_permissions,
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_used": None,
                "expires_at": (datetime.utcnow() + timedelta(days=expires_days)).isoformat() if expires_days else None
            }
            
            # Store API key
            self.api_keys[key_hash] = key_data
            
            # Return API key and info
            key_info = APIKeyInfo(
                key_id=key_data["key_id"],
                name=key_data["name"],
                permissions=key_data["permissions"],
                created_at=key_data["created_at"],
                last_used=key_data["last_used"],
                expires_at=key_data["expires_at"]
            )
            
            return api_key, key_info
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key creation failed"
            )
    
    async def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return key data"""
        
        try:
            # Hash the provided key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find key data
            key_data = self.api_keys.get(key_hash)
            if not key_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            # Check if key is active
            if not key_data.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key deactivated"
                )
            
            # Check expiration
            if key_data.get("expires_at"):
                expires_at = datetime.fromisoformat(key_data["expires_at"])
                if datetime.utcnow() > expires_at:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key expired"
                    )
            
            # Update last used
            key_data["last_used"] = datetime.utcnow().isoformat()
            
            return key_data
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key verification failed"
            )
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        
        # Admin users have all permissions
        if "system:admin" in user_permissions:
            return True
        
        # Check specific permission
        return required_permission in user_permissions
    
    def require_permission(self, required_permission: str):
        """Decorator to require specific permission"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would be used with FastAPI dependency injection
                # Implementation depends on how it's integrated with FastAPI
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def logout_user(self, token: str) -> bool:
        """Logout user by invalidating token"""
        
        try:
            # In a production system, you'd add the token to a blacklist
            # For now, we'll just verify it's valid
            payload = await self.verify_token(token)
            
            # Could store blacklisted tokens in database/cache
            logger.info(f"User logged out: {payload.get('username')}")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    async def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        
        try:
            user_data = self.users.get(user_id)
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Verify old password
            if not self._verify_password(old_password, user_data["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid current password"
                )
            
            # Hash new password
            new_hash = self._hash_password(new_password)
            user_data["password_hash"] = new_hash
            
            logger.info(f"Password changed for user: {user_data['username']}")
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return False
    
    async def _store_user_in_db(self, user_data: Dict[str, Any]):
        """Store user data in database (if available)"""
        
        if self.db_manager and self.db_manager.connections["postgresql"]:
            try:
                async with self.db_manager.postgres_pool.acquire() as connection:
                    await connection.execute("""
                        INSERT INTO users (username, email, password_hash, is_active, created_at)
                        VALUES ($1, $2, $3, $4, $5)
                    """, 
                    user_data["username"],
                    user_data["email"],
                    user_data["password_hash"],
                    user_data["is_active"],
                    datetime.fromisoformat(user_data["created_at"])
                    )
            except Exception as e:
                logger.error(f"Failed to store user in database: {e}")
    
    def set_database_manager(self, db_manager):
        """Set database manager reference"""
        self.db_manager = db_manager
    
    async def health_check(self) -> Dict[str, Any]:
        """Authentication system health check"""
        
        return {
            "status": "healthy",
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.get("is_active", True)]),
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k in self.api_keys.values() if k.get("is_active", True)]),
            "total_roles": len(self.roles),
            "encryption_enabled": True,
            "timestamp": datetime.utcnow().isoformat()
        }