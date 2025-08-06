/**
 * AETHERIUM MAIN APP COMPONENT - FINAL INTEGRATED VERSION
 * Complete integration of all services with the new IntegratedAetheriumDashboard
 */

import React, { useEffect, useState } from 'react';
import IntegratedAetheriumDashboard from './components/IntegratedAetheriumDashboard';
import { apiService, SystemHealth } from './services/api';
import { aiModelsService, AIModelConfig } from './services/aiModels';
import { websocketService, ChatSession } from './services/websocket';
import { storageService } from './services/storage';
import initDevTools from './utils/devtools';

// App Context for global state management
export interface AppContextType {
  systemHealth: SystemHealth | null;
  availableModels: AIModelConfig[];
  isWebSocketConnected: boolean;
  currentChatSession: ChatSession | null;
  apiService: typeof apiService;
  aiModelsService: typeof aiModelsService;
  websocketService: typeof websocketService;
  storageService: typeof storageService;
}

export const AppContext = React.createContext<AppContextType | null>(null);

// Main App Component
const App: React.FC = () => {
  // Global state
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [availableModels, setAvailableModels] = useState<AIModelConfig[]>([]);
  const [isWebSocketConnected, setIsWebSocketConnected] = useState(false);
  const [currentChatSession, setCurrentChatSession] = useState<ChatSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize app state
  useEffect(() => {
    initializeApp();
    
    // Initialize development tools in dev mode
    if (import.meta.env.DEV) {
      initDevTools();
    }
  }, []);

  // Monitor WebSocket connection
  useEffect(() => {
    const checkConnection = () => {
      setIsWebSocketConnected(websocketService.isConnected());
    };

    const interval = setInterval(checkConnection, 1000);
    checkConnection();

    return () => clearInterval(interval);
  }, []);

  // Initialize application state
  const initializeApp = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Load system health (optional - don't fail if backend is not running)
      try {
        const health = await apiService.getSystemHealth();
        setSystemHealth(health);
        console.log('✅ System health loaded:', health.status);
      } catch (error) {
        console.warn('⚠️ System health check failed (backend may not be running):', error);
        // Set a default health status for demo purposes
        setSystemHealth({
          status: 'degraded',
          timestamp: new Date().toISOString(),
          version: '1.0.0',
          components: {
            database: 'unknown',
            quantum: 'simulation',
            ai: 'local'
          }
        });
      }

      // Load available AI models
      const models = aiModelsService.getAvailableModels();
      setAvailableModels(models);
      console.log(`✅ ${models.length} AI models loaded`);

      // Set up WebSocket message handlers
      websocketService.onMessage('app-handler', (message) => {
        if (message.type === 'chat') {
          // Handle chat messages
          const session = websocketService.getCurrentSession();
          setCurrentChatSession(session);
        }
      });

      // Initialize chat session if none exists
      const existingSessions = websocketService.getAllSessions();
      if (existingSessions.length === 0) {
        const newSession = websocketService.createNewSession('aetherium-quantum-1');
        setCurrentChatSession(newSession);
      } else {
        setCurrentChatSession(existingSessions[0]);
      }

      // Load user preferences and restore session
      try {
        const savedSessions = storageService.getAllChatSessions();
        if (savedSessions.length > 0) {
          console.log(`✅ Restored ${savedSessions.length} chat sessions from storage`);
        }
      } catch (error) {
        console.warn('⚠️ Failed to restore chat sessions:', error);
      }

      setIsLoading(false);
      console.log('✅ App initialization complete');

      // Dispatch ready event for development tools
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('aetherium-ready'));
      }, 1000);

    } catch (error) {
      console.error('❌ App initialization failed:', error);
      setError(error instanceof Error ? error.message : 'Unknown error');
      setIsLoading(false);
    }
  };

  // Context value
  const contextValue: AppContextType = {
    systemHealth,
    availableModels,
    isWebSocketConnected,
    currentChatSession,
    apiService,
    aiModelsService,
    websocketService,
    storageService,
  };

  // Loading screen
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="text-center">
          <div className="relative">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-500 mb-4 mx-auto"></div>
            <div className="absolute inset-0 rounded-full h-16 w-16 border-r-2 border-blue-500 animate-spin animation-delay-150"></div>
          </div>
          <div className="text-white text-xl font-semibold mb-2">Loading Aetherium Platform...</div>
          <div className="text-gray-300 text-sm">Initializing quantum AI systems...</div>
        </div>
      </div>
    );
  }

  // Error screen
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-red-900 to-red-800">
        <div className="text-center text-white max-w-md px-6">
          <div className="text-6xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold mb-2">Initialization Error</h1>
          <p className="text-red-200 mb-6 text-sm">{error}</p>
          <button
            onClick={initializeApp}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg transition-colors font-medium"
          >
            🔄 Retry Initialization
          </button>
          <div className="mt-4 text-xs text-red-300">
            Note: Some features may work even if the backend is not connected
          </div>
        </div>
      </div>
    );
  }

  // Main application with integrated dashboard
  return (
    <AppContext.Provider value={contextValue}>
      <div className="App h-screen">
        {/* Global connection status indicators (top-right corner) */}
        <div className="fixed top-4 right-4 z-50 flex items-center space-x-2">
          {/* System Health Status */}
          <div
            className={`h-3 w-3 rounded-full shadow-lg ${
              systemHealth?.status === 'healthy'
                ? 'bg-green-500 shadow-green-500/50'
                : systemHealth?.status === 'degraded'
                ? 'bg-yellow-500 shadow-yellow-500/50'
                : 'bg-red-500 shadow-red-500/50'
            }`}
            title={`System Status: ${systemHealth?.status || 'Unknown'}`}
          />
          
          {/* WebSocket Connection Status */}
          <div
            className={`h-3 w-3 rounded-full shadow-lg ${
              isWebSocketConnected 
                ? 'bg-green-500 shadow-green-500/50' 
                : 'bg-red-500 shadow-red-500/50'
            }`}
            title={`WebSocket: ${isWebSocketConnected ? 'Connected' : 'Disconnected'}`}
          />

          {/* AI Models Status */}
          <div
            className={`h-3 w-3 rounded-full shadow-lg ${
              availableModels.length > 0
                ? 'bg-blue-500 shadow-blue-500/50'
                : 'bg-gray-500 shadow-gray-500/50'
            }`}
            title={`AI Models: ${availableModels.length} available`}
          />
        </div>

        {/* Main Integrated Dashboard */}
        <IntegratedAetheriumDashboard />
      </div>
    </AppContext.Provider>
  );
};

// Hook to use App context
export const useAppContext = (): AppContextType => {
  const context = React.useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppContext.Provider');
  }
  return context;
};

export default App;