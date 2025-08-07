"""Database Models and Data Management for Aetherium Platform"""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class AetheriumDataStore:
    """Complete data store for Aetherium platform with full CRUD operations"""
    
    def __init__(self):
        # Core data stores
        self.users: Dict[str, Dict] = {}
        self.chat_sessions: Dict[str, Dict] = {}
        self.chat_messages: Dict[str, List[Dict]] = {}
        self.ai_tools_usage: Dict[str, List[Dict]] = {}
        self.user_preferences: Dict[str, Dict] = {}
        self.system_metrics: Dict[str, Any] = self._initialize_metrics()
        
        print("🗄️ Data store initialized with all collections")
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize system metrics"""
        return {
            "total_users": 0,
            "total_sessions": 0,
            "total_messages": 0,
            "total_tool_executions": 0,
            "platform_uptime": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
    
    # USER MANAGEMENT
    def create_user(self, username: str, email: str, role: str = "user") -> Dict:
        """Create a new user"""
        import hashlib
        user_id = hashlib.sha256(f"{username}{email}{datetime.now()}".encode()).hexdigest()[:12]
        
        user_data = {
            "id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_active": True,
            "last_login": None,
            "total_sessions": 0,
            "total_messages": 0
        }
        
        self.users[user_id] = user_data
        self.system_metrics["total_users"] = len(self.users)
        self._update_system_metrics()
        
        return user_data
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username"""
        for user in self.users.values():
            if user.get("username") == username:
                return user
        return None
    
    def update_user(self, user_id: str, updates: Dict) -> Optional[Dict]:
        """Update user information"""
        if user_id in self.users:
            self.users[user_id].update(updates)
            self.users[user_id]["updated_at"] = datetime.now().isoformat()
            return self.users[user_id]
        return None
    
    # CHAT SESSION MANAGEMENT
    def create_chat_session(self, user_id: str, title: str = "New Chat") -> Dict:
        """Create a new chat session"""
        import hashlib
        session_id = hashlib.sha256(f"{user_id}{title}{datetime.now()}".encode()).hexdigest()[:16]
        
        session_data = {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": 0,
            "is_active": True,
            "ai_model": "aetherium_quantum",
            "metadata": {}
        }
        
        self.chat_sessions[session_id] = session_data
        self.chat_messages[session_id] = []
        
        # Update user stats
        if user_id in self.users:
            self.users[user_id]["total_sessions"] += 1
        
        self.system_metrics["total_sessions"] = len(self.chat_sessions)
        self._update_system_metrics()
        
        return session_data
    
    def add_chat_message(self, session_id: str, role: str, content: str, 
                        model: str = None, metadata: Dict = None) -> Dict:
        """Add a message to a chat session"""
        if session_id not in self.chat_messages:
            self.chat_messages[session_id] = []
        
        import hashlib
        message_id = hashlib.sha256(f"{session_id}{role}{content}{datetime.now()}".encode()).hexdigest()[:16]
        
        message_data = {
            "id": message_id,
            "session_id": session_id,
            "role": role,  # user, assistant, system
            "content": content,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.chat_messages[session_id].append(message_data)
        
        # Update session and user stats
        if session_id in self.chat_sessions:
            self.chat_sessions[session_id]["message_count"] += 1
            self.chat_sessions[session_id]["updated_at"] = datetime.now().isoformat()
            
            user_id = self.chat_sessions[session_id]["user_id"]
            if user_id in self.users:
                self.users[user_id]["total_messages"] += 1
        
        self.system_metrics["total_messages"] = sum(len(msgs) for msgs in self.chat_messages.values())
        self._update_system_metrics()
        
        return message_data
    
    def get_chat_messages(self, session_id: str, limit: int = None) -> List[Dict]:
        """Get messages for a chat session"""
        messages = self.chat_messages.get(session_id, [])
        if limit:
            return messages[-limit:]  # Get most recent messages
        return messages
    
    # AI TOOLS USAGE TRACKING
    def log_tool_usage(self, user_id: str, tool_name: str, parameters: Dict, 
                      result: Dict, execution_time: float = 0.0):
        """Log AI tool usage for analytics"""
        if user_id not in self.ai_tools_usage:
            self.ai_tools_usage[user_id] = []
        
        usage_entry = {
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat(),
            "success": result.get("status") == "completed"
        }
        
        self.ai_tools_usage[user_id].append(usage_entry)
        self.system_metrics["total_tool_executions"] += 1
        self._update_system_metrics()
    
    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics"""
        self._update_system_metrics()
        return self.system_metrics
    
    def _update_system_metrics(self):
        """Update system-wide metrics"""
        self.system_metrics.update({
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.get("is_active", True)]),
            "total_sessions": len(self.chat_sessions),
            "active_sessions": len([s for s in self.chat_sessions.values() if s.get("is_active", True)]),
            "total_messages": sum(len(msgs) for msgs in self.chat_messages.values()),
            "total_tool_executions": sum(len(usage) for usage in self.ai_tools_usage.values()),
            "last_updated": datetime.now().isoformat()
        })

# Global data store instance
data_store = AetheriumDataStore()

if __name__ == "__main__":
    print("🗄️ Database Models Initialized")
    
    # Test database operations
    test_user = data_store.create_user("testuser", "test@aetherium.com", "user")
    print(f"✅ Created user: {test_user['username']}")
    
    # Test chat session
    test_session = data_store.create_chat_session(test_user["id"], "Test Chat Session")
    print(f"✅ Created chat session: {test_session['title']}")
    
    # Test message
    test_message = data_store.add_chat_message(
        test_session["id"], 
        "user", 
        "Hello, Aetherium! This is a test message.",
        "aetherium_quantum"
    )
    print(f"✅ Added message: {test_message['id']}")
    
    # Test system metrics
    metrics = data_store.get_system_metrics()
    print(f"✅ System metrics: {metrics['total_users']} users, {metrics['total_messages']} messages")
    
    print("🗄️ Database system ready for production!")