/**
 * INTEGRATED AETHERIUM DASHBOARD
 * Complete integrated dashboard with all services connected
 */

import React, { useState, useRef, useEffect } from 'react';
import { 
  Plus, Settings, User, ChevronDown, Home, MessageSquare, 
  Archive, Clock, HelpCircle, BookOpen, QrCode, X, Menu
} from 'lucide-react';
import { useAppContext } from '../App';
import ChatInterface from './ChatInterface';
import AIToolsPanel from './AIToolsPanel';
import SystemStatusPanel from './SystemStatusPanel';
import { usePreferences } from '../hooks/useAetherium';

const IntegratedAetheriumDashboard: React.FC = () => {
  const { systemHealth, isWebSocketConnected, availableModels } = useAppContext();
  const { preferences, updatePreferences } = usePreferences();
  
  const [showDropdown, setShowDropdown] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('aetherium-quantum-1');
  const [rightPanelView, setRightPanelView] = useState('status');
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [activeLeftTab, setActiveLeftTab] = useState('chats');
  const [activeCenterTab, setActiveCenterTab] = useState('chat');
  const [showMobileSidebar, setShowMobileSidebar] = useState(false);

  const darkMode = preferences.theme === 'dark';

  // Set default model from available models
  useEffect(() => {
    if (availableModels.length > 0 && !availableModels.find(m => m.id === selectedModel)) {
      setSelectedModel(availableModels[0].id);
    }
  }, [availableModels, selectedModel]);

  // Handle tool execution from AI Tools Panel
  const handleToolExecute = (toolId: string, toolName: string) => {
    // This will be handled by the ChatInterface through context
    console.log(`Executing tool: ${toolName} (${toolId})`);
  };

  // Toggle theme
  const toggleTheme = () => {
    updatePreferences({ theme: darkMode ? 'light' : 'dark' });
  };

  // Sidebar tabs configuration
  const leftTabs = [
    { id: 'chats', label: 'Chats', icon: MessageSquare },
    { id: 'tasks', label: 'Tasks', icon: Archive },
    { id: 'history', label: 'History', icon: Clock },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'homepage', label: 'Homepage', icon: Home },
    { id: 'help', label: 'Help', icon: HelpCircle },
    { id: 'knowledge', label: 'Knowledge', icon: BookOpen },
    { id: 'qr-code', label: 'QR Code', icon: QrCode }
  ];

  const centerTabs = [
    { id: 'chat', label: 'Chat' },
    { id: 'tools', label: 'Tools' },
    { id: 'context', label: 'Context' },
    { id: 'styles', label: 'Styles' }
  ];

  const rightPanelOptions = [
    { id: 'status', label: 'System Status' },
    { id: 'preview', label: 'Live Preview' },
    { id: 'actions', label: 'Quick Actions' }
  ];

  // Sample conversations for the sidebar
  const conversations = [
    { id: 1, title: 'Aetherium Platform Development', time: 'Now', pinned: true, type: 'active' },
    { id: 2, title: 'Quantum Computing Research', time: '2 hours ago', pinned: false, type: 'normal' },
    { id: 3, title: 'AI Tool Integration', time: 'Yesterday', pinned: false, type: 'normal' },
    { id: 4, title: 'Time Crystals Implementation', time: '2 days ago', pinned: false, type: 'normal' },
    { id: 5, title: 'Neuromorphic AI Design', time: '3 days ago', pinned: false, type: 'normal' }
  ];

  return (
    <div className={`h-screen flex ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Left Sidebar */}
      <div className={`${showMobileSidebar ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 fixed md:relative z-30 w-80 h-full ${
        darkMode ? 'bg-gray-800' : 'bg-white'
      } border-r ${darkMode ? 'border-gray-700' : 'border-gray-200'} transition-transform duration-300`}>
        
        {/* Header */}
        <div className={`p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg">⚛</span>
              </div>
              <div>
                <h1 className={`font-bold text-lg ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  Aetherium
                </h1>
                <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Quantum AI Platform
                </p>
              </div>
            </div>
            <button
              onClick={() => setShowMobileSidebar(false)}
              className="md:hidden p-1 hover:bg-gray-700 rounded"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className={`border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} p-2`}>
          <div className="grid grid-cols-4 gap-1">
            {leftTabs.slice(0, 4).map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveLeftTab(tab.id)}
                className={`p-3 rounded-lg transition-colors ${
                  activeLeftTab === tab.id
                    ? `${darkMode ? 'bg-purple-600 text-white' : 'bg-purple-500 text-white'}`
                    : `${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'}`
                }`}
                title={tab.label}
              >
                <tab.icon className="w-4 h-4 mx-auto" />
              </button>
            ))}
          </div>
        </div>

        {/* Content based on active tab */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeLeftTab === 'chats' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between mb-4">
                <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  Recent Chats
                </h3>
                <button className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}>
                  <Plus className="w-4 h-4" />
                </button>
              </div>
              {conversations.map(chat => (
                <div key={chat.id} className={`p-3 rounded-lg cursor-pointer transition-colors ${
                  chat.type === 'active' 
                    ? `${darkMode ? 'bg-purple-600 bg-opacity-20 border border-purple-500' : 'bg-purple-50 border border-purple-200'}`
                    : `${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`
                }`}>
                  <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    {chat.title}
                  </div>
                  <div className={`text-xs mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {chat.time}
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {activeLeftTab === 'settings' && (
            <div className="space-y-4">
              <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Settings
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Dark Mode
                  </span>
                  <button
                    onClick={toggleTheme}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      darkMode ? 'bg-purple-600' : 'bg-gray-300'
                    }`}
                  >
                    <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                      darkMode ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Animations
                  </span>
                  <button
                    onClick={() => updatePreferences({ enableAnimations: !preferences.enableAnimations })}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      preferences.enableAnimations ? 'bg-purple-600' : 'bg-gray-300'
                    }`}
                  >
                    <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                      preferences.enableAnimations ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                <div className="flex items-center justify-between">
                  <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                    Quantum Effects
                  </span>
                  <button
                    onClick={() => updatePreferences({ enableQuantumEffects: !preferences.enableQuantumEffects })}
                    className={`w-12 h-6 rounded-full transition-colors ${
                      preferences.enableQuantumEffects ? 'bg-purple-600' : 'bg-gray-300'
                    }`}
                  >
                    <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                      preferences.enableQuantumEffects ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Connection Status */}
        <div className={`p-4 border-t ${darkMode ? 'border-gray-700 bg-gray-750' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className={`h-2 w-2 rounded-full ${isWebSocketConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {isWebSocketConnected ? 'Connected' : 'Offline'}
              </span>
            </div>
            <div className={`h-2 w-2 rounded-full ${
              systemHealth?.status === 'healthy' ? 'bg-green-500' : 
              systemHealth?.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
            }`} />
          </div>
        </div>
      </div>

      {/* Mobile Sidebar Overlay */}
      {showMobileSidebar && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-20 md:hidden"
          onClick={() => setShowMobileSidebar(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Header */}
        <div className={`h-16 ${darkMode ? 'bg-gray-800' : 'bg-white'} border-b ${
          darkMode ? 'border-gray-700' : 'border-gray-200'
        } flex items-center justify-between px-4`}>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setShowMobileSidebar(true)}
              className="md:hidden p-2 hover:bg-gray-700 rounded"
            >
              <Menu className="w-5 h-5" />
            </button>
            
            <div className="flex space-x-1">
              {centerTabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveCenterTab(tab.id)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    activeCenterTab === tab.id
                      ? `${darkMode ? 'bg-purple-600 text-white' : 'bg-purple-500 text-white'}`
                      : `${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'}`
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className={`px-3 py-1 rounded border text-sm ${
                darkMode 
                  ? 'bg-gray-700 border-gray-600 text-white' 
                  : 'bg-white border-gray-300 text-gray-900'
              }`}
            >
              {availableModels.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>

            <button
              onClick={() => setShowRightPanel(!showRightPanel)}
              className={`p-2 rounded-lg transition-colors ${
                darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'
              }`}
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 flex">
          {/* Center Panel */}
          <div className="flex-1 flex flex-col">
            {activeCenterTab === 'chat' && (
              <ChatInterface
                darkMode={darkMode}
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
              />
            )}
            
            {activeCenterTab === 'tools' && (
              <AIToolsPanel
                darkMode={darkMode}
                onToolExecute={handleToolExecute}
              />
            )}
            
            {activeCenterTab === 'context' && (
              <div className={`flex-1 p-8 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
                <div className="text-center">
                  <div className="text-4xl mb-4">📋</div>
                  <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Context Management
                  </h3>
                  <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Coming soon - Advanced context and memory management
                  </p>
                </div>
              </div>
            )}
            
            {activeCenterTab === 'styles' && (
              <div className={`flex-1 p-8 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
                <div className="text-center">
                  <div className="text-4xl mb-4">🎨</div>
                  <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Style Customization
                  </h3>
                  <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Coming soon - Advanced styling and theme customization
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel */}
          {showRightPanel && (
            <div className={`w-80 ${darkMode ? 'bg-gray-800' : 'bg-white'} border-l ${
              darkMode ? 'border-gray-700' : 'border-gray-200'
            }`}>
              <div className={`p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <div className="flex space-x-1">
                  {rightPanelOptions.map(option => (
                    <button
                      key={option.id}
                      onClick={() => setRightPanelView(option.id)}
                      className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                        rightPanelView === option.id
                          ? `${darkMode ? 'bg-purple-600 text-white' : 'bg-purple-500 text-white'}`
                          : `${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-600'}`
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex-1">
                {rightPanelView === 'status' && (
                  <SystemStatusPanel darkMode={darkMode} />
                )}
                
                {rightPanelView === 'preview' && (
                  <div className="p-4">
                    <div className="text-center py-12">
                      <div className="text-4xl mb-4">👁️</div>
                      <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Live Preview
                      </h3>
                      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        Preview your work in real-time
                      </p>
                    </div>
                  </div>
                )}
                
                {rightPanelView === 'actions' && (
                  <div className="p-4">
                    <div className="space-y-3">
                      <h4 className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Quick Actions
                      </h4>
                      {[
                        'Export Chat History',
                        'Import Settings', 
                        'Clear Cache',
                        'Run Diagnostics',
                        'Backup Data'
                      ].map((action, idx) => (
                        <button
                          key={idx}
                          className={`w-full text-left p-3 rounded-lg transition-colors ${
                            darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-700'
                          }`}
                        >
                          <span className="text-sm">{action}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default IntegratedAetheriumDashboard;