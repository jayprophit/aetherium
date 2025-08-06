"""
Aetherium Program & Software Automation Module
Advanced program control, software automation, and system orchestration
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
import asyncio
import shutil
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import psutil
import win32api
import win32con
import win32gui
import win32process
import win32service
import winreg
from pathlib import Path

@dataclass
class ProgramConfig:
    """Program automation configuration"""
    executable_path: str = ""
    working_directory: str = ""
    arguments: List[str] = None
    environment_variables: Dict[str, str] = None
    start_timeout: int = 30
    stop_timeout: int = 10
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    monitor_interval: int = 5
    log_output: bool = True
    run_as_service: bool = False

class ServiceManager:
    """Windows service management"""
    
    @staticmethod
    def get_service_status(service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of Windows service"""
        try:
            scm_handle = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ENUMERATE_SERVICE)
            service_handle = win32service.OpenService(scm_handle, service_name, win32service.SERVICE_QUERY_STATUS)
            
            status = win32service.QueryServiceStatus(service_handle)
            
            win32service.CloseServiceHandle(service_handle)
            win32service.CloseServiceHandle(scm_handle)
            
            return {
                'service_type': status[0],
                'current_state': status[1],
                'controls_accepted': status[2],
                'win32_exit_code': status[3],
                'service_specific_exit_code': status[4],
                'check_point': status[5],
                'wait_hint': status[6]
            }
            
        except Exception as e:
            logging.error(f"Failed to get service status: {e}")
            return None
    
    @staticmethod
    def start_service(service_name: str) -> bool:
        """Start Windows service"""
        try:
            scm_handle = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
            service_handle = win32service.OpenService(scm_handle, service_name, win32service.SERVICE_START)
            
            win32service.StartService(service_handle, None)
            
            win32service.CloseServiceHandle(service_handle)
            win32service.CloseServiceHandle(scm_handle)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start service: {e}")
            return False
    
    @staticmethod
    def stop_service(service_name: str) -> bool:
        """Stop Windows service"""
        try:
            scm_handle = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
            service_handle = win32service.OpenService(scm_handle, service_name, win32service.SERVICE_STOP)
            
            win32service.ControlService(service_handle, win32service.SERVICE_CONTROL_STOP)
            
            win32service.CloseServiceHandle(service_handle)
            win32service.CloseServiceHandle(scm_handle)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop service: {e}")
            return False

class RegistryManager:
    """Windows Registry management"""
    
    @staticmethod
    def read_registry_key(root_key: int, sub_key: str, value_name: str) -> Optional[Any]:
        """Read value from Windows Registry"""
        try:
            key = winreg.OpenKey(root_key, sub_key)
            value, reg_type = winreg.QueryValueEx(key, value_name)
            winreg.CloseKey(key)
            return value
        except Exception as e:
            logging.error(f"Failed to read registry key: {e}")
            return None
    
    @staticmethod
    def write_registry_key(root_key: int, sub_key: str, value_name: str, value: Any, reg_type: int) -> bool:
        """Write value to Windows Registry"""
        try:
            key = winreg.CreateKey(root_key, sub_key)
            winreg.SetValueEx(key, value_name, 0, reg_type, value)
            winreg.CloseKey(key)
            return True
        except Exception as e:
            logging.error(f"Failed to write registry key: {e}")
            return False
    
    @staticmethod
    def delete_registry_key(root_key: int, sub_key: str, value_name: str = None) -> bool:
        """Delete registry key or value"""
        try:
            key = winreg.OpenKey(root_key, sub_key, 0, winreg.KEY_WRITE)
            
            if value_name:
                winreg.DeleteValue(key, value_name)
            else:
                winreg.DeleteKey(key, "")
            
            winreg.CloseKey(key)
            return True
        except Exception as e:
            logging.error(f"Failed to delete registry key: {e}")
            return False

class ProcessManager:
    """Advanced process management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.managed_processes = {}
        self.monitoring = False
        self.monitor_thread = None
    
    def start_process(self, config: ProgramConfig, process_id: str = None) -> Optional[str]:
        """Start a new process with configuration"""
        try:
            process_id = process_id or f"proc_{int(time.time())}"
            
            # Prepare command
            cmd = [config.executable_path]
            if config.arguments:
                cmd.extend(config.arguments)
            
            # Prepare environment
            env = os.environ.copy()
            if config.environment_variables:
                env.update(config.environment_variables)
            
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=config.working_directory or None,
                env=env,
                stdout=subprocess.PIPE if config.log_output else None,
                stderr=subprocess.PIPE if config.log_output else None,
                stdin=subprocess.PIPE
            )
            
            # Store process info
            self.managed_processes[process_id] = {
                'process': process,
                'config': config,
                'start_time': time.time(),
                'restart_attempts': 0,
                'status': 'running'
            }
            
            self.logger.info(f"Started process {process_id} (PID: {process.pid})")
            return process_id
            
        except Exception as e:
            self.logger.error(f"Failed to start process: {e}")
            return None
    
    def stop_process(self, process_id: str, force: bool = False) -> bool:
        """Stop a managed process"""
        try:
            if process_id not in self.managed_processes:
                return False
            
            process_info = self.managed_processes[process_id]
            process = process_info['process']
            config = process_info['config']
            
            if force:
                process.kill()
            else:
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=config.stop_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            process_info['status'] = 'stopped'
            self.logger.info(f"Stopped process {process_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop process: {e}")
            return False
    
    def restart_process(self, process_id: str) -> bool:
        """Restart a managed process"""
        try:
            if process_id not in self.managed_processes:
                return False
            
            process_info = self.managed_processes[process_id]
            config = process_info['config']
            
            # Stop current process
            self.stop_process(process_id)
            
            # Start new process
            new_process_id = self.start_process(config, process_id)
            return new_process_id is not None
            
        except Exception as e:
            self.logger.error(f"Failed to restart process: {e}")
            return False
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get status of managed process"""
        try:
            if process_id not in self.managed_processes:
                return None
            
            process_info = self.managed_processes[process_id]
            process = process_info['process']
            
            # Check if process is still running
            if process.poll() is None:
                status = 'running'
            else:
                status = 'stopped'
                process_info['status'] = 'stopped'
            
            # Get system process info if available
            try:
                sys_process = psutil.Process(process.pid)
                cpu_percent = sys_process.cpu_percent()
                memory_info = sys_process.memory_info()
            except psutil.NoSuchProcess:
                cpu_percent = 0
                memory_info = None
            
            return {
                'process_id': process_id,
                'pid': process.pid,
                'status': status,
                'start_time': process_info['start_time'],
                'restart_attempts': process_info['restart_attempts'],
                'cpu_percent': cpu_percent,
                'memory_usage': memory_info.rss if memory_info else 0,
                'return_code': process.returncode
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get process status: {e}")
            return None
    
    def list_processes(self) -> List[Dict[str, Any]]:
        """List all managed processes"""
        processes = []
        for process_id in self.managed_processes:
            status = self.get_process_status(process_id)
            if status:
                processes.append(status)
        return processes
    
    def start_monitoring(self):
        """Start monitoring all managed processes"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_processes)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring processes"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_processes(self):
        """Monitor processes for failures and restart if needed"""
        while self.monitoring:
            try:
                for process_id in list(self.managed_processes.keys()):
                    process_info = self.managed_processes[process_id]
                    process = process_info['process']
                    config = process_info['config']
                    
                    # Check if process died
                    if process.poll() is not None and config.restart_on_failure:
                        if process_info['restart_attempts'] < config.max_restart_attempts:
                            self.logger.warning(f"Process {process_id} died, attempting restart")
                            process_info['restart_attempts'] += 1
                            
                            # Restart process
                            self.start_process(config, process_id)
                        else:
                            self.logger.error(f"Process {process_id} exceeded max restart attempts")
                            process_info['status'] = 'failed'
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
    
    def send_signal_to_process(self, process_id: str, signal: str) -> bool:
        """Send signal to managed process"""
        try:
            if process_id not in self.managed_processes:
                return False
            
            process = self.managed_processes[process_id]['process']
            
            if sys.platform == "win32":
                if signal == "CTRL_C":
                    process.send_signal(subprocess.signal.CTRL_C_EVENT)
                elif signal == "CTRL_BREAK":
                    process.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
                else:
                    return False
            else:
                import signal as sig_module
                sig_num = getattr(sig_module, f"SIG{signal.upper()}", None)
                if sig_num:
                    process.send_signal(sig_num)
                else:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send signal: {e}")
            return False

class SoftwareInstaller:
    """Software installation and management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def install_msi(self, msi_path: str, silent: bool = True, log_file: str = None) -> bool:
        """Install MSI package"""
        try:
            cmd = ["msiexec", "/i", msi_path]
            
            if silent:
                cmd.append("/quiet")
            
            if log_file:
                cmd.extend(["/l*v", log_file])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to install MSI: {e}")
            return False
    
    def uninstall_msi(self, product_code: str, silent: bool = True) -> bool:
        """Uninstall MSI package by product code"""
        try:
            cmd = ["msiexec", "/x", product_code]
            
            if silent:
                cmd.append("/quiet")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to uninstall MSI: {e}")
            return False
    
    def install_chocolatey_package(self, package_name: str, version: str = None) -> bool:
        """Install package using Chocolatey"""
        try:
            cmd = ["choco", "install", package_name, "-y"]
            
            if version:
                cmd.extend(["--version", version])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to install Chocolatey package: {e}")
            return False
    
    def install_pip_package(self, package_name: str, version: str = None) -> bool:
        """Install Python package using pip"""
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            
            if version:
                cmd.append(f"{package_name}=={version}")
            else:
                cmd.append(package_name)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Failed to install pip package: {e}")
            return False

class ConfigurationManager:
    """Configuration file management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_config_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    import yaml
                    return yaml.safe_load(f)
                elif file_path.endswith('.ini'):
                    import configparser
                    config = configparser.ConfigParser()
                    config.read(file_path)
                    return {section: dict(config[section]) for section in config.sections()}
                else:
                    # Plain text
                    return {'content': f.read()}
        except Exception as e:
            self.logger.error(f"Failed to read config file: {e}")
            return None
    
    def write_config_file(self, file_path: str, config_data: Dict[str, Any]) -> bool:
        """Write configuration to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(config_data, f, indent=2)
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    import yaml
                    yaml.dump(config_data, f, default_flow_style=False)
                elif file_path.endswith('.ini'):
                    import configparser
                    config = configparser.ConfigParser()
                    for section, values in config_data.items():
                        config[section] = values
                    config.write(f)
                else:
                    # Plain text
                    f.write(config_data.get('content', ''))
            return True
        except Exception as e:
            self.logger.error(f"Failed to write config file: {e}")
            return False
    
    def backup_config_file(self, file_path: str) -> Optional[str]:
        """Create backup of configuration file"""
        try:
            backup_path = f"{file_path}.backup_{int(time.time())}"
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to backup config file: {e}")
            return None

class ProgramAutomation:
    """Main program automation orchestrator"""
    
    def __init__(self):
        self.process_manager = ProcessManager()
        self.service_manager = ServiceManager()
        self.registry_manager = RegistryManager()
        self.installer = SoftwareInstaller()
        self.config_manager = ConfigurationManager()
        self.logger = logging.getLogger(__name__)
        
        # Start process monitoring
        self.process_manager.start_monitoring()
    
    def create_program_config(self, **kwargs) -> ProgramConfig:
        """Create program configuration"""
        return ProgramConfig(**kwargs)
    
    def start_program(self, config: ProgramConfig, program_id: str = None) -> Optional[str]:
        """Start a program with configuration"""
        return self.process_manager.start_process(config, program_id)
    
    def stop_program(self, program_id: str, force: bool = False) -> bool:
        """Stop a program"""
        return self.process_manager.stop_process(program_id, force)
    
    def restart_program(self, program_id: str) -> bool:
        """Restart a program"""
        return self.process_manager.restart_process(program_id)
    
    def get_program_status(self, program_id: str) -> Optional[Dict[str, Any]]:
        """Get program status"""
        return self.process_manager.get_process_status(program_id)
    
    def list_programs(self) -> List[Dict[str, Any]]:
        """List all managed programs"""
        return self.process_manager.list_processes()
    
    def install_software(self, installer_type: str, **kwargs) -> bool:
        """Install software using specified installer"""
        if installer_type == "msi":
            return self.installer.install_msi(**kwargs)
        elif installer_type == "chocolatey":
            return self.installer.install_chocolatey_package(**kwargs)
        elif installer_type == "pip":
            return self.installer.install_pip_package(**kwargs)
        else:
            self.logger.error(f"Unknown installer type: {installer_type}")
            return False
    
    def manage_service(self, service_name: str, action: str) -> bool:
        """Manage Windows service"""
        if action == "start":
            return self.service_manager.start_service(service_name)
        elif action == "stop":
            return self.service_manager.stop_service(service_name)
        elif action == "status":
            status = self.service_manager.get_service_status(service_name)
            return status is not None
        else:
            self.logger.error(f"Unknown service action: {action}")
            return False
    
    def manage_registry(self, action: str, **kwargs) -> Any:
        """Manage Windows Registry"""
        if action == "read":
            return self.registry_manager.read_registry_key(**kwargs)
        elif action == "write":
            return self.registry_manager.write_registry_key(**kwargs)
        elif action == "delete":
            return self.registry_manager.delete_registry_key(**kwargs)
        else:
            self.logger.error(f"Unknown registry action: {action}")
            return None
    
    def manage_config(self, action: str, file_path: str, config_data: Dict = None) -> Any:
        """Manage configuration files"""
        if action == "read":
            return self.config_manager.read_config_file(file_path)
        elif action == "write":
            return self.config_manager.write_config_file(file_path, config_data)
        elif action == "backup":
            return self.config_manager.backup_config_file(file_path)
        else:
            self.logger.error(f"Unknown config action: {action}")
            return None
    
    def create_automation_script(self, script_name: str, operations: List[Dict[str, Any]]) -> str:
        """Generate automation script"""
        script_lines = [
            "#!/usr/bin/env python3",
            "# Generated Aetherium Program Automation Script",
            "",
            "from aetherium.automation.program_automation import ProgramAutomation, ProgramConfig",
            "import time",
            "",
            "def main():",
            "    automation = ProgramAutomation()",
            ""
        ]
        
        for op in operations:
            op_type = op.get('type')
            
            if op_type == 'start_program':
                script_lines.append(f"    # Start program: {op.get('name', 'Unknown')}")
                script_lines.append(f"    config = ProgramConfig(")
                script_lines.append(f"        executable_path='{op.get('executable_path', '')}',")
                script_lines.append(f"        arguments={op.get('arguments', [])},")
                script_lines.append(f"    )")
                script_lines.append(f"    program_id = automation.start_program(config)")
                script_lines.append(f"    print(f'Started program: {{program_id}}')")
                
            elif op_type == 'install_software':
                script_lines.append(f"    # Install software: {op.get('name', 'Unknown')}")
                script_lines.append(f"    result = automation.install_software('{op.get('installer_type')}', **{op.get('kwargs', {})})")
                script_lines.append(f"    print(f'Installation result: {{result}}')")
                
            elif op_type == 'manage_service':
                script_lines.append(f"    # Manage service: {op.get('service_name', 'Unknown')}")
                script_lines.append(f"    result = automation.manage_service('{op.get('service_name')}', '{op.get('action')}')")
                script_lines.append(f"    print(f'Service action result: {{result}}')")
            
            script_lines.append("")
        
        script_lines.extend([
            "",
            "if __name__ == '__main__':",
            "    main()"
        ])
        
        return "\n".join(script_lines)
    
    def execute_batch_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute batch operations"""
        results = []
        
        for i, operation in enumerate(operations):
            try:
                op_type = operation.get('type')
                result = {'operation_index': i, 'type': op_type, 'success': False}
                
                if op_type == 'start_program':
                    config = ProgramConfig(**operation.get('config', {}))
                    program_id = self.start_program(config)
                    result.update({'success': program_id is not None, 'program_id': program_id})
                
                elif op_type == 'stop_program':
                    success = self.stop_program(operation.get('program_id'), operation.get('force', False))
                    result.update({'success': success})
                
                elif op_type == 'install_software':
                    success = self.install_software(**operation.get('kwargs', {}))
                    result.update({'success': success})
                
                elif op_type == 'manage_service':
                    success = self.manage_service(operation.get('service_name'), operation.get('action'))
                    result.update({'success': success})
                
                elif op_type == 'manage_registry':
                    result_data = self.manage_registry(**operation.get('kwargs', {}))
                    result.update({'success': result_data is not None, 'data': result_data})
                
                elif op_type == 'manage_config':
                    result_data = self.manage_config(**operation.get('kwargs', {}))
                    result.update({'success': result_data is not None, 'data': result_data})
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to execute operation {i}: {e}")
                results.append({
                    'operation_index': i, 
                    'type': operation.get('type'), 
                    'success': False, 
                    'error': str(e)
                })
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.process_manager.stop_monitoring()
        
        # Stop all managed processes
        for process_id in list(self.process_manager.managed_processes.keys()):
            self.stop_program(process_id)

# Example usage
if __name__ == "__main__":
    # Initialize program automation
    automation = ProgramAutomation()
    
    # Create program configuration
    config = automation.create_program_config(
        executable_path="notepad.exe",
        arguments=["test.txt"],
        restart_on_failure=True
    )
    
    # Start program
    program_id = automation.start_program(config)
    print(f"Started program: {program_id}")
    
    # Check status
    status = automation.get_program_status(program_id)
    print(f"Program status: {status}")
    
    # Create automation script
    operations = [
        {
            'type': 'start_program',
            'name': 'Notepad',
            'executable_path': 'notepad.exe',
            'arguments': ['readme.txt']
        },
        {
            'type': 'install_software',
            'name': 'Git',
            'installer_type': 'chocolatey',
            'kwargs': {'package_name': 'git'}
        }
    ]
    
    script = automation.create_automation_script("example_automation", operations)
    print("Generated automation script:")
    print(script)