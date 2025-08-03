"""
Aetherium AI Productivity Suite Manager
Central orchestration for all AI-powered productivity tools and features
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from datetime import datetime

from ..config.config_manager import ConfigManager
from ..security.auth_manager import AuthenticationManager

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of AI productivity tools"""
    RESEARCH = "research"
    CREATIVE = "creative"
    CONTENT = "content"
    BUSINESS = "business"
    TRANSLATION = "translation"
    DEVELOPMENT = "development"
    EXPERIMENTAL = "experimental"


class AIProductivitySuiteManager:
    """
    Central manager for all AI productivity suite features.
    Coordinates between different AI service categories and manages user context.
    """
    
    def __init__(self, config_manager: ConfigManager, auth_manager: AuthenticationManager):
        self.config_manager = config_manager
        self.auth_manager = auth_manager
        self.services = {}
        self.user_contexts = {}
        self.usage_analytics = {}
        
        # Initialize service categories
        self._initialize_services()
        
        logger.info("AI Productivity Suite Manager initialized")
    
    def _initialize_services(self):
        """Initialize all AI service categories"""
        try:
            # Import and initialize all service managers
            from .services.research_service import ResearchService
            from .services.creative_service import CreativeService
            from .services.content_service import ContentService
            from .services.business_service import BusinessService
            from .services.translation_service import TranslationService
            from .services.development_service import DevelopmentService
            from .services.experimental_service import ExperimentalService
            
            self.services = {
                ToolCategory.RESEARCH: ResearchService(self.config_manager),
                ToolCategory.CREATIVE: CreativeService(self.config_manager),
                ToolCategory.CONTENT: ContentService(self.config_manager),
                ToolCategory.BUSINESS: BusinessService(self.config_manager),
                ToolCategory.TRANSLATION: TranslationService(self.config_manager),
                ToolCategory.DEVELOPMENT: DevelopmentService(self.config_manager),
                ToolCategory.EXPERIMENTAL: ExperimentalService(self.config_manager)
            }
            
            logger.info(f"Initialized {len(self.services)} AI service categories")
            
        except ImportError as e:
            logger.warning(f"Some AI services not yet implemented: {e}")
            # Initialize with placeholder services for development
            self._initialize_placeholder_services()
    
    def _initialize_placeholder_services(self):
        """Initialize placeholder services for development"""
        from .services.base_service import BaseAIService
        
        for category in ToolCategory:
            self.services[category] = BaseAIService(self.config_manager, category.value)
    
    async def route_request(self, 
                          category: str, 
                          tool: str, 
                          request: Dict[str, Any],
                          user_id: str,
                          auth_token: str) -> Dict[str, Any]:
        """
        Route AI productivity suite requests to appropriate services
        
        Args:
            category: Tool category (research, creative, content, etc.)
            tool: Specific tool name within category
            request: Request parameters and data
            user_id: User identifier
            auth_token: Authentication token
            
        Returns:
            Dict containing the tool response and metadata
        """
        try:
            # Validate authentication
            if not await self.auth_manager.validate_token(auth_token):
                raise ValueError("Invalid authentication token")
            
            # Get user permissions
            user_permissions = await self.auth_manager.get_user_permissions(user_id)
            
            # Check if user has access to this tool category
            if not self._check_tool_access(category, user_permissions):
                raise ValueError(f"Insufficient permissions for {category} tools")
            
            # Get or create user context
            user_context = await self._get_user_context(user_id)
            
            # Route to appropriate service
            category_enum = ToolCategory(category.lower())
            service = self.services.get(category_enum)
            
            if not service:
                raise ValueError(f"Unknown tool category: {category}")
            
            # Execute the tool request
            result = await service.execute_tool(tool, request, user_context)
            
            # Update usage analytics
            await self._update_usage_analytics(user_id, category, tool, result)
            
            # Update user context with result
            await self._update_user_context(user_id, category, tool, request, result)
            
            return {
                "success": True,
                "category": category,
                "tool": tool,
                "result": result,
                "timestamp": datetime.utcnow().isoformat(),
                "user_context_updated": True
            }
            
        except Exception as e:
            logger.error(f"Error routing AI suite request: {e}")
            return {
                "success": False,
                "error": str(e),
                "category": category,
                "tool": tool,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _check_tool_access(self, category: str, permissions: List[str]) -> bool:
        """Check if user has access to specific tool category"""
        # Basic permission mapping - can be enhanced
        permission_map = {
            "research": ["ai_suite_research", "ai_suite_all"],
            "creative": ["ai_suite_creative", "ai_suite_all"],
            "content": ["ai_suite_content", "ai_suite_all"],
            "business": ["ai_suite_business", "ai_suite_all"],
            "translation": ["ai_suite_translation", "ai_suite_all"],
            "development": ["ai_suite_development", "ai_suite_all"],
            "experimental": ["ai_suite_experimental", "ai_suite_all"]
        }
        
        required_permissions = permission_map.get(category.lower(), [])
        return any(perm in permissions for perm in required_permissions)
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get or create user context for personalized AI interactions"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                "user_id": user_id,
                "preferences": {},
                "history": [],
                "active_projects": [],
                "frequently_used_tools": {},
                "created_at": datetime.utcnow().isoformat(),
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return self.user_contexts[user_id]
    
    async def _update_user_context(self, 
                                 user_id: str, 
                                 category: str, 
                                 tool: str, 
                                 request: Dict[str, Any], 
                                 result: Dict[str, Any]):
        """Update user context with recent activity"""
        context = self.user_contexts.get(user_id, {})
        
        # Add to history
        if "history" not in context:
            context["history"] = []
        
        context["history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "tool": tool,
            "request_summary": self._summarize_request(request),
            "result_summary": self._summarize_result(result)
        })
        
        # Keep only last 100 history items
        context["history"] = context["history"][-100:]
        
        # Update frequently used tools
        if "frequently_used_tools" not in context:
            context["frequently_used_tools"] = {}
        
        tool_key = f"{category}:{tool}"
        context["frequently_used_tools"][tool_key] = context["frequently_used_tools"].get(tool_key, 0) + 1
        
        # Update timestamp
        context["last_updated"] = datetime.utcnow().isoformat()
        
        self.user_contexts[user_id] = context
    
    def _summarize_request(self, request: Dict[str, Any]) -> str:
        """Create a summary of the request for context storage"""
        # Simple summarization - can be enhanced with AI
        if "query" in request:
            return request["query"][:100] + "..." if len(request["query"]) > 100 else request["query"]
        elif "text" in request:
            return request["text"][:100] + "..." if len(request["text"]) > 100 else request["text"]
        else:
            return f"Request with {len(request)} parameters"
    
    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """Create a summary of the result for context storage"""
        if "output" in result:
            output = str(result["output"])
            return output[:100] + "..." if len(output) > 100 else output
        elif "generated_content" in result:
            content = str(result["generated_content"])
            return content[:100] + "..." if len(content) > 100 else content
        else:
            return f"Result with {len(result)} fields"
    
    async def _update_usage_analytics(self, 
                                    user_id: str, 
                                    category: str, 
                                    tool: str, 
                                    result: Dict[str, Any]):
        """Update usage analytics for monitoring and optimization"""
        timestamp = datetime.utcnow()
        date_key = timestamp.strftime("%Y-%m-%d")
        
        if date_key not in self.usage_analytics:
            self.usage_analytics[date_key] = {
                "total_requests": 0,
                "categories": {},
                "tools": {},
                "users": set()
            }
        
        daily_stats = self.usage_analytics[date_key]
        daily_stats["total_requests"] += 1
        daily_stats["users"].add(user_id)
        
        # Category stats
        if category not in daily_stats["categories"]:
            daily_stats["categories"][category] = 0
        daily_stats["categories"][category] += 1
        
        # Tool stats
        tool_key = f"{category}:{tool}"
        if tool_key not in daily_stats["tools"]:
            daily_stats["tools"][tool_key] = 0
        daily_stats["tools"][tool_key] += 1
    
    async def get_user_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard data for user including recent activity and recommendations"""
        context = await self._get_user_context(user_id)
        
        # Get recent activity
        recent_activity = context.get("history", [])[-10:]  # Last 10 activities
        
        # Get frequently used tools
        frequent_tools = dict(sorted(
            context.get("frequently_used_tools", {}).items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])  # Top 5 tools
        
        # Generate recommendations based on usage patterns
        recommendations = await self._generate_tool_recommendations(context)
        
        return {
            "recent_activity": recent_activity,
            "frequently_used_tools": frequent_tools,
            "recommendations": recommendations,
            "total_tools_used": len(context.get("frequently_used_tools", {})),
            "total_sessions": len(context.get("history", []))
        }
    
    async def _generate_tool_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tool recommendations based on user context and usage patterns"""
        recommendations = []
        
        # Simple recommendation logic - can be enhanced with ML
        frequent_tools = context.get("frequently_used_tools", {})
        
        # If user uses research tools, recommend analysis tools
        if any("research:" in tool for tool in frequent_tools):
            recommendations.append({
                "category": "research",
                "tool": "data_visualization",
                "reason": "Enhance your research with data visualization"
            })
        
        # If user uses creative tools, recommend design tools
        if any("creative:" in tool for tool in frequent_tools):
            recommendations.append({
                "category": "creative",
                "tool": "style_transfer",
                "reason": "Try style transfer for your creative projects"
            })
        
        # If user uses content tools, recommend business tools
        if any("content:" in tool for tool in frequent_tools):
            recommendations.append({
                "category": "business",
                "tool": "swot_analysis",
                "reason": "Analyze your content strategy with SWOT analysis"
            })
        
        return recommendations
    
    async def get_available_tools(self) -> Dict[str, List[str]]:
        """Get list of all available tools organized by category"""
        available_tools = {}
        
        for category, service in self.services.items():
            available_tools[category.value] = await service.get_available_tools()
        
        return available_tools
    
    async def get_platform_stats(self) -> Dict[str, Any]:
        """Get overall platform statistics"""
        total_users = len(self.user_contexts)
        total_categories = len(self.services)
        
        # Aggregate usage stats
        total_requests = 0
        for daily_stats in self.usage_analytics.values():
            total_requests += daily_stats["total_requests"]
        
        return {
            "total_users": total_users,
            "total_categories": total_categories,
            "total_requests": total_requests,
            "active_services": list(self.services.keys()),
            "platform_status": "operational"
        }
