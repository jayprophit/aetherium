"""
Aetherium Automation Orchestrator
Master automation system that orchestrates all automation modules and workflows
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .browser_automation import BrowserAutomation, BrowserConfig
from .desktop_automation import DesktopAutomation, DesktopConfig
from .app_automation import AppAutomation, AppConfig
from .program_automation import ProgramAutomation, ProgramConfig

@dataclass
class WorkflowStep:
    """Individual workflow step"""
    id: str
    type: str  # browser, desktop, app, program, api, custom
    action: str
    parameters: Dict[str, Any]
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 5
    depends_on: List[str] = None
    condition: Optional[str] = None

@dataclass
class Workflow:
    """Automation workflow definition"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    schedule: Optional[str] = None  # cron expression
    enabled: bool = True
    created_at: float = 0
    updated_at: float = 0
    tags: List[str] = None

class AutomationOrchestrator:
    """Master automation orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.browser_automation = BrowserAutomation()
        self.desktop_automation = DesktopAutomation()
        self.app_automation = AppAutomation()
        self.program_automation = ProgramAutomation()
        
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, Dict] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.is_running = False
        self.scheduler_thread = None
        
        # Initialize automation systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all automation systems"""
        try:
            self.logger.info("Initializing automation systems...")
            
            # Initialize browser automation
            self.browser_automation.start_browser()
            
            # Initialize desktop automation
            # Already initialized in constructor
            
            # Initialize app automation
            # Already initialized in constructor
            
            # Initialize program automation
            # Already initialized in constructor
            
            self.logger.info("All automation systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize automation systems: {e}")
    
    def create_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Create new automation workflow"""
        try:
            workflow_id = workflow_data.get('id', f"workflow_{int(time.time())}")
            
            # Convert step dictionaries to WorkflowStep objects
            steps = []
            for step_data in workflow_data.get('steps', []):
                step = WorkflowStep(
                    id=step_data.get('id', f"step_{len(steps)}"),
                    type=step_data['type'],
                    action=step_data['action'],
                    parameters=step_data.get('parameters', {}),
                    timeout=step_data.get('timeout', 60),
                    retry_attempts=step_data.get('retry_attempts', 3),
                    retry_delay=step_data.get('retry_delay', 5),
                    depends_on=step_data.get('depends_on', []),
                    condition=step_data.get('condition')
                )
                steps.append(step)
            
            workflow = Workflow(
                id=workflow_id,
                name=workflow_data['name'],
                description=workflow_data.get('description', ''),
                steps=steps,
                schedule=workflow_data.get('schedule'),
                enabled=workflow_data.get('enabled', True),
                created_at=time.time(),
                updated_at=time.time(),
                tags=workflow_data.get('tags', [])
            )
            
            self.workflows[workflow_id] = workflow
            self.logger.info(f"Created workflow: {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            return False
    
    def update_workflow(self, workflow_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing workflow"""
        try:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(workflow, field):
                    setattr(workflow, field, value)
            
            workflow.updated_at = time.time()
            self.logger.info(f"Updated workflow: {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update workflow: {e}")
            return False
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow"""
        try:
            if workflow_id in self.workflows:
                del self.workflows[workflow_id]
                self.logger.info(f"Deleted workflow: {workflow_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete workflow: {e}")
            return False
    
    async def execute_workflow(self, workflow_id: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute automation workflow"""
        try:
            if workflow_id not in self.workflows:
                return {'success': False, 'error': 'Workflow not found'}
            
            workflow = self.workflows[workflow_id]
            if not workflow.enabled:
                return {'success': False, 'error': 'Workflow is disabled'}
            
            execution_id = f"{workflow_id}_{int(time.time())}"
            
            # Track execution
            self.running_workflows[execution_id] = {
                'workflow_id': workflow_id,
                'start_time': time.time(),
                'status': 'running',
                'current_step': 0,
                'results': {},
                'errors': []
            }
            
            self.logger.info(f"Starting workflow execution: {execution_id}")
            
            # Execute steps
            step_results = {}
            completed_steps = set()
            
            for i, step in enumerate(workflow.steps):
                try:
                    # Check dependencies
                    if step.depends_on:
                        missing_deps = [dep for dep in step.depends_on if dep not in completed_steps]
                        if missing_deps:
                            self.logger.warning(f"Step {step.id} has unmet dependencies: {missing_deps}")
                            continue
                    
                    # Check condition if specified
                    if step.condition:
                        if not self._evaluate_condition(step.condition, step_results):
                            self.logger.info(f"Skipping step {step.id} due to condition: {step.condition}")
                            continue
                    
                    # Update current step
                    self.running_workflows[execution_id]['current_step'] = i
                    
                    # Execute step with retries
                    step_success = False
                    step_result = None
                    
                    for attempt in range(step.retry_attempts):
                        try:
                            step_result = await self._execute_step(step, parameters)
                            step_success = True
                            break
                        except Exception as step_e:
                            self.logger.warning(f"Step {step.id} attempt {attempt + 1} failed: {step_e}")
                            if attempt < step.retry_attempts - 1:
                                await asyncio.sleep(step.retry_delay)
                            else:
                                self.running_workflows[execution_id]['errors'].append(str(step_e))
                    
                    step_results[step.id] = {
                        'success': step_success,
                        'result': step_result,
                        'timestamp': time.time()
                    }
                    
                    if step_success:
                        completed_steps.add(step.id)
                    else:
                        self.logger.error(f"Step {step.id} failed after all retry attempts")
                        break
                    
                except Exception as e:
                    self.logger.error(f"Error executing step {step.id}: {e}")
                    step_results[step.id] = {
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    break
            
            # Update execution status
            execution_success = len(completed_steps) == len(workflow.steps)
            self.running_workflows[execution_id].update({
                'status': 'completed' if execution_success else 'failed',
                'end_time': time.time(),
                'results': step_results
            })
            
            # Add to history
            self.workflow_history.append(dict(self.running_workflows[execution_id]))
            
            # Clean up
            del self.running_workflows[execution_id]
            
            return {
                'success': execution_success,
                'execution_id': execution_id,
                'results': step_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute workflow: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_step(self, step: WorkflowStep, global_params: Dict[str, Any] = None) -> Any:
        """Execute individual workflow step"""
        try:
            # Merge global and step parameters
            params = {}
            if global_params:
                params.update(global_params)
            params.update(step.parameters)
            
            if step.type == 'browser':
                return await self._execute_browser_step(step.action, params)
            elif step.type == 'desktop':
                return await self._execute_desktop_step(step.action, params)
            elif step.type == 'app':
                return await self._execute_app_step(step.action, params)
            elif step.type == 'program':
                return await self._execute_program_step(step.action, params)
            elif step.type == 'api':
                return await self._execute_api_step(step.action, params)
            elif step.type == 'custom':
                return await self._execute_custom_step(step.action, params)
            else:
                raise ValueError(f"Unknown step type: {step.type}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute step {step.id}: {e}")
            raise
    
    async def _execute_browser_step(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute browser automation step"""
        if action == 'navigate':
            return self.browser_automation.navigate_to(params['url'])
        elif action == 'click':
            return self.browser_automation.click_element(params['selector'], params.get('by', 'css'))
        elif action == 'input':
            return self.browser_automation.input_text(params['selector'], params['text'], params.get('by', 'css'))
        elif action == 'scrape':
            return self.browser_automation.scrape_page(params.get('url'))
        elif action == 'screenshot':
            return self.browser_automation.take_screenshot(params.get('filename'))
        else:
            raise ValueError(f"Unknown browser action: {action}")
    
    async def _execute_desktop_step(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute desktop automation step"""
        if action == 'click':
            return self.desktop_automation.click(params.get('x'), params.get('y'))
        elif action == 'type':
            return self.desktop_automation.type_text(params['text'])
        elif action == 'key':
            return self.desktop_automation.press_key(params['key'])
        elif action == 'open_app':
            return self.desktop_automation.open_application(params['app_path'], params.get('args'))
        elif action == 'screenshot':
            return self.desktop_automation.image_recognition.save_screenshot(params['filename'], params.get('region'))
        else:
            raise ValueError(f"Unknown desktop action: {action}")
    
    async def _execute_app_step(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute app automation step"""
        app_name = params.get('app_name')
        
        if action == 'mobile_tap':
            app = self.app_automation.get_mobile_app(app_name)
            return app.tap_element(params['locator'], params.get('by', 'id')) if app else False
        elif action == 'web_get':
            app = self.app_automation.get_web_app(app_name)
            return await app.get_data(params['endpoint'], params.get('params')) if app else {}
        elif action == 'web_post':
            app = self.app_automation.get_web_app(app_name)
            return await app.post_data(params['endpoint'], params.get('data', {})) if app else {}
        else:
            raise ValueError(f"Unknown app action: {action}")
    
    async def _execute_program_step(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute program automation step"""
        if action == 'start':
            config = ProgramConfig(**params.get('config', {}))
            return self.program_automation.start_program(config, params.get('program_id'))
        elif action == 'stop':
            return self.program_automation.stop_program(params['program_id'], params.get('force', False))
        elif action == 'restart':
            return self.program_automation.restart_program(params['program_id'])
        elif action == 'status':
            return self.program_automation.get_program_status(params['program_id'])
        else:
            raise ValueError(f"Unknown program action: {action}")
    
    async def _execute_api_step(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute API call step"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            method = params.get('method', 'GET')
            url = params['url']
            headers = params.get('headers', {})
            data = params.get('data')
            
            async with session.request(method, url, json=data, headers=headers) as response:
                return {
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'data': await response.json() if response.content_type == 'application/json' else await response.text()
                }
    
    async def _execute_custom_step(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute custom step (user-defined function)"""
        # This would allow users to define custom automation functions
        # For now, return placeholder
        return {'custom_action': action, 'params': params}
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate step condition"""
        try:
            # Simple condition evaluation
            # In production, use a proper expression evaluator
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            self.logger.error(f"Failed to evaluate condition: {e}")
            return False
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Workflow]:
        """List all workflows"""
        return list(self.workflows.values())
    
    def get_running_workflows(self) -> Dict[str, Dict]:
        """Get currently running workflows"""
        return dict(self.running_workflows)
    
    def get_workflow_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return self.workflow_history[-limit:]
    
    def save_workflow(self, workflow_id: str, filename: str) -> bool:
        """Save workflow to file"""
        try:
            if workflow_id not in self.workflows:
                return False
            
            workflow = self.workflows[workflow_id]
            with open(filename, 'w') as f:
                json.dump(asdict(workflow), f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow: {e}")
            return False
    
    def load_workflow(self, filename: str) -> bool:
        """Load workflow from file"""
        try:
            with open(filename, 'r') as f:
                workflow_data = json.load(f)
            
            return self.create_workflow(workflow_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load workflow: {e}")
            return False
    
    def start_scheduler(self):
        """Start workflow scheduler for scheduled workflows"""
        if not self.is_running:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            self.logger.info("Workflow scheduler started")
    
    def stop_scheduler(self):
        """Stop workflow scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            self.logger.info("Workflow scheduler stopped")
    
    def _scheduler_loop(self):
        """Scheduler main loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                for workflow in self.workflows.values():
                    if workflow.schedule and workflow.enabled:
                        # Check if workflow should run based on schedule
                        # This would use a proper cron parser in production
                        if self._should_run_workflow(workflow, current_time):
                            asyncio.run(self.execute_workflow(workflow.id))
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
    
    def _should_run_workflow(self, workflow: Workflow, current_time: float) -> bool:
        """Check if workflow should run based on schedule"""
        # Simplified scheduling logic
        # In production, use a proper cron parser like croniter
        return False
    
    def create_predefined_workflows(self):
        """Create predefined automation workflows"""
        
        # System Health Check Workflow
        health_check_workflow = {
            'id': 'system_health_check',
            'name': 'System Health Check',
            'description': 'Automated system health monitoring and reporting',
            'steps': [
                {
                    'id': 'check_disk_space',
                    'type': 'custom',
                    'action': 'check_disk_space',
                    'parameters': {'threshold': 90}
                },
                {
                    'id': 'check_memory',
                    'type': 'custom',
                    'action': 'check_memory',
                    'parameters': {'threshold': 85}
                },
                {
                    'id': 'check_services',
                    'type': 'program',
                    'action': 'check_services',
                    'parameters': {'services': ['aetherium-backend', 'aetherium-frontend']}
                }
            ],
            'schedule': '*/30 * * * *',  # Every 30 minutes
            'enabled': True,
            'tags': ['monitoring', 'health']
        }
        
        # Data Backup Workflow
        backup_workflow = {
            'id': 'data_backup',
            'name': 'Automated Data Backup',
            'description': 'Regular backup of important data and configurations',
            'steps': [
                {
                    'id': 'backup_database',
                    'type': 'custom',
                    'action': 'backup_database',
                    'parameters': {'destination': '/backups/db'}
                },
                {
                    'id': 'backup_configs',
                    'type': 'custom',
                    'action': 'backup_configs',
                    'parameters': {'destination': '/backups/config'}
                },
                {
                    'id': 'cleanup_old_backups',
                    'type': 'custom',
                    'action': 'cleanup_backups',
                    'parameters': {'retention_days': 30}
                }
            ],
            'schedule': '0 2 * * *',  # Daily at 2 AM
            'enabled': True,
            'tags': ['backup', 'maintenance']
        }
        
        # Web Scraping Workflow
        scraping_workflow = {
            'id': 'web_scraping',
            'name': 'Automated Web Scraping',
            'description': 'Regular web scraping for data collection',
            'steps': [
                {
                    'id': 'navigate_to_site',
                    'type': 'browser',
                    'action': 'navigate',
                    'parameters': {'url': 'https://example.com'}
                },
                {
                    'id': 'scrape_data',
                    'type': 'browser',
                    'action': 'scrape',
                    'parameters': {}
                },
                {
                    'id': 'save_data',
                    'type': 'custom',
                    'action': 'save_scraped_data',
                    'parameters': {'format': 'json'}
                }
            ],
            'schedule': '0 * * * *',  # Hourly
            'enabled': False,
            'tags': ['scraping', 'data']
        }
        
        # Create workflows
        self.create_workflow(health_check_workflow)
        self.create_workflow(backup_workflow)
        self.create_workflow(scraping_workflow)
        
        self.logger.info("Created predefined workflows")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_scheduler()
            
            # Stop all automation systems
            self.browser_automation.close_browser()
            self.program_automation.cleanup()
            self.app_automation.close_all()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Automation orchestrator cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

# Global orchestrator instance
automation_orchestrator = None

def get_automation_orchestrator() -> AutomationOrchestrator:
    """Get global automation orchestrator instance"""
    global automation_orchestrator
    if automation_orchestrator is None:
        automation_orchestrator = AutomationOrchestrator()
    return automation_orchestrator

# Example usage and testing
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = AutomationOrchestrator()
    
    # Create sample workflow
    workflow_data = {
        'id': 'sample_workflow',
        'name': 'Sample Automation Workflow',
        'description': 'A sample workflow for testing',
        'steps': [
            {
                'id': 'take_screenshot',
                'type': 'desktop',
                'action': 'screenshot',
                'parameters': {'filename': 'test_screenshot.png'}
            },
            {
                'id': 'open_notepad',
                'type': 'desktop',
                'action': 'open_app',
                'parameters': {'app_path': 'notepad.exe'}
            },
            {
                'id': 'type_message',
                'type': 'desktop',
                'action': 'type',
                'parameters': {'text': 'Hello from Aetherium Automation!'}
            }
        ],
        'enabled': True
    }
    
    # Create and execute workflow
    if orchestrator.create_workflow(workflow_data):
        print("Workflow created successfully")
        
        # Execute workflow
        result = asyncio.run(orchestrator.execute_workflow('sample_workflow'))
        print(f"Workflow execution result: {result}")
    
    # Create predefined workflows
    orchestrator.create_predefined_workflows()
    
    # List workflows
    workflows = orchestrator.list_workflows()
    print(f"Total workflows: {len(workflows)}")
    for workflow in workflows:
        print(f"- {workflow.name} ({workflow.id})")
    
    # Cleanup
    orchestrator.cleanup()