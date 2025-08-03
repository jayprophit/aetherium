"""
Base AI Service - Foundation class for all AI productivity suite services
Provides common functionality, interface definitions, and error handling
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from ...config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class BaseAIService(ABC):
    """
    Abstract base class for all AI productivity suite services.
    Provides common functionality and interface definitions.
    """
    
    def __init__(self, config_manager: ConfigManager, service_name: str):
        self.config_manager = config_manager
        self.service_name = service_name
        self.tools = {}
        self.service_config = self._load_service_config()
        
        # Initialize service-specific components
        self._initialize_service()
        
        logger.info(f"Initialized AI service: {service_name}")
    
    def _load_service_config(self) -> Dict[str, Any]:
        """Load service-specific configuration"""
        try:
            ai_suite_config = self.config_manager.get("ai_productivity_suite", {})
            return ai_suite_config.get(self.service_name, {})
        except Exception as e:
            logger.warning(f"Could not load config for {self.service_name}: {e}")
            return {}
    
    def _initialize_service(self):
        """Initialize service-specific components - can be overridden by subclasses"""
        pass
    
    async def execute_tool(self, 
                          tool_name: str, 
                          request: Dict[str, Any], 
                          user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific tool within this service
        
        Args:
            tool_name: Name of the tool to execute
            request: Request parameters and data
            user_context: User context for personalization
            
        Returns:
            Dict containing the tool execution result
        """
        try:
            # Validate tool exists
            if tool_name not in self.tools:
                available_tools = list(self.tools.keys())
                raise ValueError(f"Tool '{tool_name}' not available in {self.service_name}. Available tools: {available_tools}")
            
            # Get tool handler
            tool_handler = self.tools[tool_name]
            
            # Add request metadata
            enriched_request = {
                **request,
                "service": self.service_name,
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat(),
                "user_context": user_context
            }
            
            # Execute tool
            result = await tool_handler(enriched_request)
            
            # Add result metadata
            enriched_result = {
                **result,
                "service": self.service_name,
                "tool": tool_name,
                "execution_time": datetime.utcnow().isoformat(),
                "success": True
            }
            
            return enriched_result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} in {self.service_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "service": self.service_name,
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_available_tools(self) -> List[str]:
        """Get list of available tools in this service"""
        return list(self.tools.keys())
    
    async def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in {self.service_name}")
        
        # Return tool metadata - can be enhanced with more details
        return {
            "name": tool_name,
            "service": self.service_name,
            "description": self._get_tool_description(tool_name),
            "parameters": self._get_tool_parameters(tool_name),
            "examples": self._get_tool_examples(tool_name)
        }
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a tool - can be overridden by subclasses"""
        return f"AI-powered {tool_name} tool in {self.service_name} service"
    
    def _get_tool_parameters(self, tool_name: str) -> Dict[str, Any]:
        """Get parameter specification for a tool - can be overridden by subclasses"""
        return {
            "query": {"type": "string", "description": "Input query or text"},
            "options": {"type": "object", "description": "Additional options"}
        }
    
    def _get_tool_examples(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get usage examples for a tool - can be overridden by subclasses"""
        return [
            {
                "input": {"query": "Example input"},
                "output": {"result": "Example output"}
            }
        ]
    
    def register_tool(self, tool_name: str, tool_handler):
        """Register a tool handler with this service"""
        self.tools[tool_name] = tool_handler
        logger.info(f"Registered tool '{tool_name}' in {self.service_name}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health and availability"""
        try:
            # Basic health check - can be enhanced with service-specific checks
            return {
                "service": self.service_name,
                "status": "healthy",
                "tools_count": len(self.tools),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # Common utility methods for AI service implementations
    
    def _extract_text_from_request(self, request: Dict[str, Any]) -> str:
        """Extract text content from request"""
        for key in ["query", "text", "content", "input"]:
            if key in request and request[key]:
                return str(request[key])
        return ""
    
    def _get_user_preference(self, user_context: Dict[str, Any], preference_key: str, default_value: Any = None) -> Any:
        """Get user preference from context"""
        preferences = user_context.get("preferences", {})
        return preferences.get(preference_key, default_value)
    
    def _format_response(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format service response with consistent structure"""
        response = {
            "generated_content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    async def _call_external_api(self, api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call external API - placeholder for actual implementation"""
        # This would be implemented with actual HTTP client
        logger.info(f"Calling external API: {api_url}")
        return {"placeholder": "external_api_response"}
    
    async def _generate_with_ai_model(self, prompt: str, model_params: Dict[str, Any] = None) -> str:
        """Generate content using AI model - placeholder for actual implementation"""
        # This would be implemented with actual AI model integration
        logger.info(f"Generating content with AI model for prompt: {prompt[:50]}...")
        return f"AI-generated response for: {prompt[:100]}..."


class PlaceholderAIService(BaseAIService):
    """
    Placeholder service implementation for development and testing
    """
    
    def _initialize_service(self):
        """Initialize placeholder tools"""
        # Register placeholder tools
        self.register_tool("placeholder_tool", self._placeholder_tool_handler)
        self.register_tool("echo_tool", self._echo_tool_handler)
        self.register_tool("mock_analysis", self._mock_analysis_handler)
    
    async def _placeholder_tool_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder tool handler"""
        text = self._extract_text_from_request(request)
        
        return self._format_response(
            content=f"Placeholder response for: {text}",
            metadata={
                "tool_type": "placeholder",
                "processed_length": len(text)
            }
        )
    
    async def _echo_tool_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Echo tool for testing"""
        return self._format_response(
            content=request,
            metadata={"tool_type": "echo"}
        )
    
    async def _mock_analysis_handler(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock analysis tool"""
        text = self._extract_text_from_request(request)
        
        # Mock analysis results
        analysis = {
            "word_count": len(text.split()) if text else 0,
            "character_count": len(text),
            "sentiment": "neutral",
            "topics": ["general", "analysis"],
            "confidence": 0.85
        }
        
        return self._format_response(
            content=analysis,
            metadata={"tool_type": "analysis"}
        )
