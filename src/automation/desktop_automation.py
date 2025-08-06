"""
Aetherium Desktop Automation Module
Advanced desktop automation with PyAutoGUI, Windows API, and cross-platform support
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pyautogui
import psutil
import win32api
import win32con
import win32gui
import win32process
from PIL import Image, ImageGrab
import cv2
import numpy as np

@dataclass
class DesktopConfig:
    """Desktop automation configuration"""
    screenshot_interval: float = 0.1
    mouse_speed: float = 0.5
    key_delay: float = 0.05
    confidence_threshold: float = 0.8
    failsafe: bool = True
    pause: float = 0.1

class WindowManager:
    """Windows management utilities"""
    
    @staticmethod
    def get_all_windows() -> List[Dict[str, Any]]:
        """Get list of all windows"""
        windows = []
        
        def enum_windows_proc(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text:
                    rect = win32gui.GetWindowRect(hwnd)
                    pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                    
                    try:
                        process = psutil.Process(pid)
                        process_name = process.name()
                    except:
                        process_name = "Unknown"
                    
                    windows.append({
                        'hwnd': hwnd,
                        'title': window_text,
                        'rect': rect,
                        'pid': pid,
                        'process_name': process_name
                    })
            return True
        
        win32gui.EnumWindows(enum_windows_proc, None)
        return windows
    
    @staticmethod
    def find_window_by_title(title: str, partial: bool = True) -> Optional[Dict[str, Any]]:
        """Find window by title"""
        windows = WindowManager.get_all_windows()
        
        for window in windows:
            if partial:
                if title.lower() in window['title'].lower():
                    return window
            else:
                if title.lower() == window['title'].lower():
                    return window
        
        return None
    
    @staticmethod
    def find_window_by_process(process_name: str) -> Optional[Dict[str, Any]]:
        """Find window by process name"""
        windows = WindowManager.get_all_windows()
        
        for window in windows:
            if process_name.lower() in window['process_name'].lower():
                return window
        
        return None
    
    @staticmethod
    def activate_window(hwnd: int) -> bool:
        """Activate window"""
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            logging.error(f"Failed to activate window: {e}")
            return False
    
    @staticmethod
    def resize_window(hwnd: int, width: int, height: int) -> bool:
        """Resize window"""
        try:
            win32gui.SetWindowPos(hwnd, 0, 0, 0, width, height, win32con.SWP_NOMOVE | win32con.SWP_NOZORDER)
            return True
        except Exception as e:
            logging.error(f"Failed to resize window: {e}")
            return False
    
    @staticmethod
    def move_window(hwnd: int, x: int, y: int) -> bool:
        """Move window"""
        try:
            win32gui.SetWindowPos(hwnd, 0, x, y, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_NOZORDER)
            return True
        except Exception as e:
            logging.error(f"Failed to move window: {e}")
            return False
    
    @staticmethod
    def close_window(hwnd: int) -> bool:
        """Close window"""
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as e:
            logging.error(f"Failed to close window: {e}")
            return False

class ProcessManager:
    """Process management utilities"""
    
    @staticmethod
    def get_running_processes() -> List[Dict[str, Any]]:
        """Get list of running processes"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return processes
    
    @staticmethod
    def find_process(name: str) -> Optional[psutil.Process]:
        """Find process by name"""
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if name.lower() in proc.info['name'].lower():
                    return psutil.Process(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return None
    
    @staticmethod
    def start_process(executable: str, args: List[str] = None) -> Optional[subprocess.Popen]:
        """Start a new process"""
        try:
            command = [executable]
            if args:
                command.extend(args)
            
            return subprocess.Popen(command)
        except Exception as e:
            logging.error(f"Failed to start process: {e}")
            return None
    
    @staticmethod
    def kill_process(pid: int) -> bool:
        """Kill process by PID"""
        try:
            process = psutil.Process(pid)
            process.terminate()
            return True
        except Exception as e:
            logging.error(f"Failed to kill process: {e}")
            return False

class ImageRecognition:
    """Image recognition for GUI automation"""
    
    @staticmethod
    def find_image_on_screen(template_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """Find image template on screen"""
        try:
            location = pyautogui.locateOnScreen(template_path, confidence=confidence)
            if location:
                return pyautogui.center(location)
            return None
        except Exception as e:
            logging.error(f"Failed to find image on screen: {e}")
            return None
    
    @staticmethod
    def find_all_images_on_screen(template_path: str, confidence: float = 0.8) -> List[Tuple[int, int]]:
        """Find all instances of image template on screen"""
        try:
            locations = list(pyautogui.locateAllOnScreen(template_path, confidence=confidence))
            return [pyautogui.center(loc) for loc in locations]
        except Exception as e:
            logging.error(f"Failed to find images on screen: {e}")
            return []
    
    @staticmethod
    def take_screenshot(region: Tuple[int, int, int, int] = None) -> Image.Image:
        """Take screenshot of screen or region"""
        try:
            if region:
                return ImageGrab.grab(region)
            else:
                return ImageGrab.grab()
        except Exception as e:
            logging.error(f"Failed to take screenshot: {e}")
            return None
    
    @staticmethod
    def save_screenshot(filename: str, region: Tuple[int, int, int, int] = None) -> bool:
        """Save screenshot to file"""
        try:
            screenshot = ImageRecognition.take_screenshot(region)
            if screenshot:
                screenshot.save(filename)
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to save screenshot: {e}")
            return False

class DesktopAutomation:
    """Main desktop automation class"""
    
    def __init__(self, config: DesktopConfig = None):
        self.config = config or DesktopConfig()
        self.logger = logging.getLogger(__name__)
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = self.config.failsafe
        pyautogui.PAUSE = self.config.pause
        
        self.window_manager = WindowManager()
        self.process_manager = ProcessManager()
        self.image_recognition = ImageRecognition()
    
    # Mouse Operations
    def move_mouse(self, x: int, y: int, duration: float = None) -> bool:
        """Move mouse to coordinates"""
        try:
            duration = duration or self.config.mouse_speed
            pyautogui.moveTo(x, y, duration=duration)
            return True
        except Exception as e:
            self.logger.error(f"Failed to move mouse: {e}")
            return False
    
    def click(self, x: int = None, y: int = None, button: str = 'left', clicks: int = 1) -> bool:
        """Click at coordinates or current position"""
        try:
            if x is not None and y is not None:
                pyautogui.click(x, y, button=button, clicks=clicks)
            else:
                pyautogui.click(button=button, clicks=clicks)
            return True
        except Exception as e:
            self.logger.error(f"Failed to click: {e}")
            return False
    
    def double_click(self, x: int = None, y: int = None) -> bool:
        """Double click at coordinates or current position"""
        return self.click(x, y, clicks=2)
    
    def right_click(self, x: int = None, y: int = None) -> bool:
        """Right click at coordinates or current position"""
        return self.click(x, y, button='right')
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0) -> bool:
        """Drag from start to end coordinates"""
        try:
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration)
            return True
        except Exception as e:
            self.logger.error(f"Failed to drag: {e}")
            return False
    
    def scroll(self, clicks: int, x: int = None, y: int = None) -> bool:
        """Scroll at coordinates or current position"""
        try:
            if x is not None and y is not None:
                pyautogui.scroll(clicks, x=x, y=y)
            else:
                pyautogui.scroll(clicks)
            return True
        except Exception as e:
            self.logger.error(f"Failed to scroll: {e}")
            return False
    
    # Keyboard Operations
    def type_text(self, text: str, interval: float = None) -> bool:
        """Type text"""
        try:
            interval = interval or self.config.key_delay
            pyautogui.write(text, interval=interval)
            return True
        except Exception as e:
            self.logger.error(f"Failed to type text: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a key"""
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to press key: {e}")
            return False
    
    def press_keys(self, keys: List[str]) -> bool:
        """Press multiple keys simultaneously"""
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as e:
            self.logger.error(f"Failed to press keys: {e}")
            return False
    
    def key_down(self, key: str) -> bool:
        """Press and hold key"""
        try:
            pyautogui.keyDown(key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to press key down: {e}")
            return False
    
    def key_up(self, key: str) -> bool:
        """Release key"""
        try:
            pyautogui.keyUp(key)
            return True
        except Exception as e:
            self.logger.error(f"Failed to release key: {e}")
            return False
    
    # Application Control
    def open_application(self, app_path: str, args: List[str] = None) -> bool:
        """Open application"""
        try:
            process = self.process_manager.start_process(app_path, args)
            if process:
                time.sleep(2)  # Wait for app to start
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to open application: {e}")
            return False
    
    def close_application(self, process_name: str) -> bool:
        """Close application by process name"""
        try:
            process = self.process_manager.find_process(process_name)
            if process:
                process.terminate()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to close application: {e}")
            return False
    
    def switch_to_application(self, app_name: str) -> bool:
        """Switch to application window"""
        try:
            window = self.window_manager.find_window_by_title(app_name)
            if not window:
                window = self.window_manager.find_window_by_process(app_name)
            
            if window:
                return self.window_manager.activate_window(window['hwnd'])
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to switch to application: {e}")
            return False
    
    # File Operations
    def open_file(self, file_path: str) -> bool:
        """Open file with default application"""
        try:
            os.startfile(file_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to open file: {e}")
            return False
    
    def create_folder(self, folder_path: str) -> bool:
        """Create folder"""
        try:
            os.makedirs(folder_path, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create folder: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete file: {e}")
            return False
    
    def copy_file(self, source: str, destination: str) -> bool:
        """Copy file"""
        try:
            import shutil
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            self.logger.error(f"Failed to copy file: {e}")
            return False
    
    def move_file(self, source: str, destination: str) -> bool:
        """Move file"""
        try:
            import shutil
            shutil.move(source, destination)
            return True
        except Exception as e:
            self.logger.error(f"Failed to move file: {e}")
            return False
    
    # Image-based Automation
    def click_image(self, image_path: str, confidence: float = None) -> bool:
        """Click on image found on screen"""
        try:
            confidence = confidence or self.config.confidence_threshold
            location = self.image_recognition.find_image_on_screen(image_path, confidence)
            
            if location:
                return self.click(location[0], location[1])
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to click image: {e}")
            return False
    
    def wait_for_image(self, image_path: str, timeout: int = 30, confidence: float = None) -> Optional[Tuple[int, int]]:
        """Wait for image to appear on screen"""
        try:
            confidence = confidence or self.config.confidence_threshold
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                location = self.image_recognition.find_image_on_screen(image_path, confidence)
                if location:
                    return location
                
                time.sleep(self.config.screenshot_interval)
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to wait for image: {e}")
            return None
    
    # System Information
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen resolution"""
        return pyautogui.size()
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()
    
    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """Get pixel color at coordinates"""
        return pyautogui.pixel(x, y)
    
    def get_active_window(self) -> Optional[Dict[str, Any]]:
        """Get currently active window"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                title = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                
                return {
                    'hwnd': hwnd,
                    'title': title,
                    'rect': rect,
                    'pid': pid
                }
        except Exception as e:
            self.logger.error(f"Failed to get active window: {e}")
        
        return None
    
    # Advanced Automation
    def record_actions(self, duration: int = 60) -> List[Dict[str, Any]]:
        """Record user actions for playback"""
        actions = []
        start_time = time.time()
        
        # This would require more complex implementation
        # For now, return empty list
        return actions
    
    def playback_actions(self, actions: List[Dict[str, Any]]) -> bool:
        """Playback recorded actions"""
        try:
            for action in actions:
                action_type = action.get('type')
                
                if action_type == 'click':
                    self.click(action['x'], action['y'])
                elif action_type == 'key':
                    self.press_key(action['key'])
                elif action_type == 'type':
                    self.type_text(action['text'])
                
                time.sleep(action.get('delay', 0.1))
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to playback actions: {e}")
            return False
    
    def create_automation_script(self, script_data: Dict[str, Any]) -> str:
        """Generate automation script from configuration"""
        script_lines = [
            "# Generated Aetherium Desktop Automation Script",
            "from aetherium.automation.desktop_automation import DesktopAutomation",
            "",
            "def main():",
            "    automation = DesktopAutomation()",
            ""
        ]
        
        for step in script_data.get('steps', []):
            step_type = step.get('type')
            
            if step_type == 'click':
                script_lines.append(f"    automation.click({step['x']}, {step['y']})")
            elif step_type == 'type':
                script_lines.append(f"    automation.type_text('{step['text']}')")
            elif step_type == 'key':
                script_lines.append(f"    automation.press_key('{step['key']}')")
            elif step_type == 'wait':
                script_lines.append(f"    time.sleep({step['duration']})")
        
        script_lines.extend([
            "",
            "if __name__ == '__main__':",
            "    main()"
        ])
        
        return "\n".join(script_lines)

# Advanced Features
class AdvancedDesktopAutomation(DesktopAutomation):
    """Advanced desktop automation with AI and ML capabilities"""
    
    def __init__(self, config: DesktopConfig = None):
        super().__init__(config)
        self.ocr_engine = None  # Initialize OCR if needed
    
    def extract_text_from_screen(self, region: Tuple[int, int, int, int] = None) -> str:
        """Extract text from screen using OCR"""
        try:
            # This would require OCR library like pytesseract
            # For now, return placeholder
            return "OCR text extraction not implemented"
        except Exception as e:
            self.logger.error(f"Failed to extract text: {e}")
            return ""
    
    def smart_click(self, text: str, region: Tuple[int, int, int, int] = None) -> bool:
        """Click on text found on screen"""
        try:
            # This would use OCR to find text and click on it
            # For now, return placeholder
            return False
        except Exception as e:
            self.logger.error(f"Failed to smart click: {e}")
            return False
    
    def monitor_screen_changes(self, callback, region: Tuple[int, int, int, int] = None, interval: float = 1.0):
        """Monitor screen for changes and call callback"""
        try:
            last_screenshot = self.image_recognition.take_screenshot(region)
            
            while True:
                time.sleep(interval)
                current_screenshot = self.image_recognition.take_screenshot(region)
                
                # Compare screenshots
                if not self._images_equal(last_screenshot, current_screenshot):
                    callback({
                        'timestamp': time.time(),
                        'region': region,
                        'previous': last_screenshot,
                        'current': current_screenshot
                    })
                    last_screenshot = current_screenshot
                    
        except Exception as e:
            self.logger.error(f"Failed to monitor screen changes: {e}")
    
    def _images_equal(self, img1: Image.Image, img2: Image.Image, threshold: float = 0.95) -> bool:
        """Compare two images for similarity"""
        try:
            # Convert to numpy arrays
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Calculate similarity
            similarity = cv2.matchTemplate(arr1, arr2, cv2.TM_CCOEFF_NORMED)[0][0]
            return similarity > threshold
            
        except Exception:
            return False

# Example usage
if __name__ == "__main__":
    # Basic desktop automation
    automation = DesktopAutomation()
    
    # Get screen size
    width, height = automation.get_screen_size()
    print(f"Screen size: {width}x{height}")
    
    # Get mouse position
    x, y = automation.get_mouse_position()
    print(f"Mouse position: ({x}, {y})")
    
    # Take screenshot
    if automation.image_recognition.save_screenshot("screenshot.png"):
        print("Screenshot saved successfully")