"""
Aetherium Automation Module
Comprehensive automation suite for browser, desktop, app, and program automation
"""

from .browser_automation import BrowserAutomation, BrowserConfig, AdvancedBrowserAutomation
from .desktop_automation import DesktopAutomation, DesktopConfig, WindowManager, ProcessManager, ImageRecognition, AdvancedDesktopAutomation
from .app_automation import AppAutomation, AppConfig, MobileAppAutomation, WebAppAutomation, DesktopAppAutomation, APIAutomation
from .program_automation import ProgramAutomation, ProgramConfig, ServiceManager, RegistryManager, SoftwareInstaller, ConfigurationManager
from .automation_orchestrator import AutomationOrchestrator, Workflow, WorkflowStep, get_automation_orchestrator

__all__ = [
    # Browser Automation
    'BrowserAutomation',
    'BrowserConfig', 
    'AdvancedBrowserAutomation',
    
    # Desktop Automation
    'DesktopAutomation',
    'DesktopConfig',
    'WindowManager',
    'ProcessManager',
    'ImageRecognition',
    'AdvancedDesktopAutomation',
    
    # App Automation
    'AppAutomation',
    'AppConfig',
    'MobileAppAutomation',
    'WebAppAutomation',
    'DesktopAppAutomation',
    'APIAutomation',
    
    # Program Automation
    'ProgramAutomation',
    'ProgramConfig',
    'ServiceManager',
    'RegistryManager',
    'SoftwareInstaller',
    'ConfigurationManager',
    
    # Orchestration
    'AutomationOrchestrator',
    'Workflow',
    'WorkflowStep',
    'get_automation_orchestrator'
]

__version__ = "1.0.0"
__author__ = "Aetherium Team"
__description__ = "Comprehensive automation suite for all platform layers"