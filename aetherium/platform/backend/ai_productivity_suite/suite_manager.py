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

logger = logging.getLogger(__name__)


class ServiceCategory(Enum):
    """Categories of AI productivity services"""
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    SHOPPING = "shopping"
    AUTOMATION = "automation"


class AISuiteManager:
    """
    Central manager for all AI productivity suite services.
    Coordinates between different AI service categories and manages user context.
    """
    
    def __init__(self):
        self.services = {}
        self.user_contexts = {}
        self.usage_analytics = {}
        
        # Initialize service categories
        self._initialize_services()
        
        logger.info("AI Productivity Suite Manager initialized")
    
    def _initialize_services(self):
        """Initialize all AI service categories"""
        try:
            # Import and initialize all implemented services
            from .services.communication_service import CommunicationService
            from .services.analysis_service import AnalysisService
            from .services.creative_service import CreativeService
            from .services.shopping_service import ShoppingService
            from .services.automation_service import AutomationService
            
            self.services = {
                ServiceCategory.COMMUNICATION.value: CommunicationService(),
                ServiceCategory.ANALYSIS.value: AnalysisService(),
                ServiceCategory.CREATIVE.value: CreativeService(),
                ServiceCategory.SHOPPING.value: ShoppingService(),
                ServiceCategory.AUTOMATION.value: AutomationService()
            }
            
            logger.info(f"Initialized {len(self.services)} AI service categories")
            
        except ImportError as e:
            logger.error(f"Failed to import AI services: {e}")
            raise
    
    async def get_service(self, service_name: str):
        """Get a specific AI service by name"""
        if service_name not in self.services:
            raise ValueError(f"Service '{service_name}' not found. Available: {list(self.services.keys())}")
        
        return self.services[service_name]
    
    async def list_available_services(self) -> List[str]:
        """List all available AI services"""
        return list(self.services.keys())
    
    async def get_suite_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the AI productivity suite"""
        service_status = {}
        total_tools = 0
        
        for service_name, service in self.services.items():
            # Get service-specific status
            service_tools = []
            if hasattr(service, 'get_available_tools'):
                service_tools = await service.get_available_tools()
            
            service_status[service_name] = {
                "status": "active",
                "tools_count": len(service_tools),
                "tools": service_tools
            }
            total_tools += len(service_tools)
        
        return {
            "suite_status": "operational",
            "total_services": len(self.services),
            "total_tools": total_tools,
            "services": service_status,
            "initialized_at": datetime.now().isoformat()
        }
    
    async def execute_tool(self, service_name: str, tool_name: str, **kwargs) -> Any:
        """Execute a specific tool from a service"""
        service = await self.get_service(service_name)
        
        if not hasattr(service, tool_name):
            raise ValueError(f"Tool '{tool_name}' not found in service '{service_name}'")
        
        tool_method = getattr(service, tool_name)
        
        if not callable(tool_method):
            raise ValueError(f"'{tool_name}' is not a callable tool")
        
        # Execute the tool
        return await tool_method(**kwargs)
    
    async def get_service_tools(self, service_name: str) -> List[str]:
        """Get list of available tools for a specific service"""
        service = await self.get_service(service_name)
        
        # Get all callable methods that don't start with underscore
        tools = []
        for attr_name in dir(service):
            if not attr_name.startswith('_'):
                attr = getattr(service, attr_name)
                if callable(attr) and asyncio.iscoroutinefunction(attr):
                    tools.append(attr_name)
        
        return tools
    
    async def get_tool_info(self, service_name: str, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool"""
        service = await self.get_service(service_name)
        
        if not hasattr(service, tool_name):
            raise ValueError(f"Tool '{tool_name}' not found in service '{service_name}'")
        
        tool_method = getattr(service, tool_name)
        
        return {
            "service": service_name,
            "tool": tool_name,
            "description": tool_method.__doc__ or "No description available",
            "is_async": asyncio.iscoroutinefunction(tool_method),
            "available": True
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        health_status = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for service_name, service in self.services.items():
            try:
                # Basic health check - ensure service is accessible
                tools = await self.get_service_tools(service_name)
                health_status["services"][service_name] = {
                    "status": "healthy",
                    "tools_available": len(tools)
                }
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status
