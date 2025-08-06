/**
 * AETHERIUM MAIN APP COMPONENT
 * Integrates all services with the AetheriumDashboard component
 */

import React, { useEffect, useState } from 'react';
import AetheriumDashboard from './components/AetheriumDashboard';
import { apiService, SystemHealth } from './services/api';
import { aiModelsService, AIModelConfig } from './services/aiModels';
import { websocketService, ChatSession } from './services/websocket';

// App Context for global state management
export interface AppContextType {
  systemHealth: SystemHealth | null;
  availableModels: AIModelConfig[];
  isWebSocketConnected: boolean;
  currentChatSession: ChatSession | null;
  apiService: typeof apiService;
  aiModelsService: typeof aiModelsService;
  websocketService: typeof websocketService;
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

      // Load system health
      try {
        const health = await apiService.getSystemHealth();
        setSystemHealth(health);
        console.log('✅ System health loaded:', health.status);
      } catch (error) {
        console.warn('⚠️ System health check failed:', error);
        // Continue without backend health
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
        const newSession = websocketService.createNewSession('quantum-1');
        setCurrentChatSession(newSession);
      } else {
        setCurrentChatSession(existingSessions[0]);
      }

      setIsLoading(false);
      console.log('✅ App initialization complete');

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
  };

  // Loading screen
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
          <div className="text-white text-lg">Loading Aetherium Platform...</div>
        </div>
      </div>
    );
  }

  // Error screen
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-red-900 to-red-800">
        <div className="text-center text-white">
          <div className="text-6xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold mb-2">Application Error</h1>
          <p className="text-red-200 mb-4">{error}</p>
          <button
            onClick={initializeApp}
            className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded transition-colors"
          >
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  // Main application
  return (
    <AppContext.Provider value={contextValue}>
      <div className="App">
        {/* Global connection status indicator */}
        <div className="fixed top-4 right-4 z-50 flex items-center space-x-2">
          {/* System Health Status */}
          <div
            className={`h-3 w-3 rounded-full ${
              systemHealth?.status === 'healthy'
                ? 'bg-green-500'
                : systemHealth?.status === 'degraded'
                ? 'bg-yellow-500'
                : 'bg-red-500'
            }`}
            title={`System Status: ${systemHealth?.status || 'Unknown'}`}
          />
          
          {/* WebSocket Connection Status */}
          <div
            className={`h-3 w-3 rounded-full ${
              isWebSocketConnected ? 'bg-green-500' : 'bg-red-500'
            }`}
            title={`WebSocket: ${isWebSocketConnected ? 'Connected' : 'Disconnected'}`}
          />
        </div>

        {/* Main Dashboard Component */}
        <AetheriumDashboard />
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