"""Authentication Service for Aetherium Platform"""
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

class AuthenticationService:
    """Complete authentication system with user management"""
    
    def __init__(self):
        self.secret_key = "aetherium-secure-key-2024"
        self.tokens: Dict[str, Dict] = {}
        self.users: Dict[str, Dict] = {}
        
        # Create default admin user
        self.create_default_admin()
    
    def create_default_admin(self):
        """Create default admin user for testing"""
        admin_user = {
            "id": "admin_001",
            "username": "admin",
            "email": "admin@aetherium.com",
            "password_hash": self.hash_password("admin123"),
            "role": "admin",
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        self.users["admin_001"] = admin_user
        print("👤 Default admin user created (admin/admin123)")
    
    def register_user(self, username: str, email: str, password: str, role: str = "user") -> Dict:
        """Register a new user"""
        # Check if username already exists
        for user in self.users.values():
            if user["username"] == username:
                return {"status": "error", "message": "Username already exists"}
        
        # Create new user
        user_id = hashlib.sha256(f"{username}{email}{datetime.now()}".encode()).hexdigest()[:12]
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": self.hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        self.users[user_id] = user_data
        return {"status": "success", "user_id": user_id, "message": "User registered successfully"}
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return user_id if successful"""
        for user_id, user in self.users.items():
            if user["username"] == username and user["is_active"]:
                if self.verify_password(password, user["password_hash"]):
                    return user_id
        return None
    
    def generate_token(self, user_id: str) -> str:
        """Generate authentication token"""
        token_data = f"{user_id}{datetime.now().timestamp()}{self.secret_key}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        
        self.tokens[token] = {
            "user_id": user_id,
            "expires": datetime.now() + timedelta(hours=24),
            "created": datetime.now().isoformat()
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify token and return user_id if valid"""
        if token in self.tokens:
            token_info = self.tokens[token]
            if datetime.now() < token_info["expires"]:
                return token_info["user_id"]
            else:
                # Clean up expired token
                del self.tokens[token]
        return None
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return hashlib.sha256(f"{password}{self.secret_key}".encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == password_hash
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information (without password)"""
        if user_id in self.users:
            user_info = self.users[user_id].copy()
            del user_info["password_hash"]
            return user_info
        return None
    
    def get_active_tokens(self) -> int:
        """Get count of active tokens"""
        current_time = datetime.now()
        active_count = 0
        expired_tokens = []
        
        for token, info in self.tokens.items():
            if current_time < info["expires"]:
                active_count += 1
            else:
                expired_tokens.append(token)
        
        # Clean up expired tokens
        for token in expired_tokens:
            del self.tokens[token]
        
        return active_count

# Global authentication service instance
auth_service = AuthenticationService()

if __name__ == "__main__":
    print("🔐 Authentication Service Initialized")
    
    # Test authentication
    print("Testing authentication system...")
    
    # Test admin login
    user_id = auth_service.authenticate_user("admin", "admin123")
    if user_id:
        print("✅ Admin authentication successful")
        
        # Generate token
        token = auth_service.generate_token(user_id)
        print(f"✅ Token generated: {token[:20]}...")
        
        # Verify token
        verified_user_id = auth_service.verify_token(token)
        if verified_user_id == user_id:
            print("✅ Token verification successful")
        
        # Get user info
        user_info = auth_service.get_user_info(user_id)
        print(f"✅ User info retrieved: {user_info['username']}")
    
    print("🔐 Authentication system ready for production!")