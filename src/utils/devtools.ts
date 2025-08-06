/**
 * AETHERIUM DEVELOPMENT TOOLS
 * Development utilities and debugging tools
 */

import { apiService } from '../services/api';
import { aiModelsService } from '../services/aiModels';
import { websocketService } from '../services/websocket';

// Development tools interface
interface DevTools {
  api: typeof apiService;
  aiModels: typeof aiModelsService;
  websocket: typeof websocketService;
  debug: {
    enableVerboseLogging: () => void;
    disableVerboseLogging: () => void;
    dumpState: () => void;
    testConnections: () => Promise<void>;
    simulateError: (component: string) => void;
    clearStorage: () => void;
  };
  performance: {
    startMeasure: (name: string) => void;
    endMeasure: (name: string) => void;
    getMetrics: () => any;
  };
}

// Performance measurement utilities
const performanceMetrics: Map<string, number> = new Map();

const performance = {
  startMeasure: (name: string) => {
    performanceMetrics.set(name, Date.now());
    console.time(`📊 ${name}`);
  },
  
  endMeasure: (name: string) => {
    const startTime = performanceMetrics.get(name);
    if (startTime) {
      const duration = Date.now() - startTime;
      console.timeEnd(`📊 ${name}`);
      console.log(`⏱️ ${name}: ${duration}ms`);
      performanceMetrics.set(`${name}_duration`, duration);
    }
  },
  
  getMetrics: () => {
    return Object.fromEntries(performanceMetrics.entries());
  }
};

// Debug utilities
const debug = {
  enableVerboseLogging: () => {
    localStorage.setItem('aetherium_debug', 'true');
    console.log('🔍 Verbose logging enabled');
  },
  
  disableVerboseLogging: () => {
    localStorage.removeItem('aetherium_debug');
    console.log('🔇 Verbose logging disabled');
  },
  
  dumpState: () => {
    const state = {
      timestamp: new Date().toISOString(),
      websocketConnected: websocketService.isConnected(),
      websocketState: websocketService.getConnectionState(),
      chatSessions: websocketService.getAllSessions().length,
      availableModels: aiModelsService.getAvailableModels().length,
      localStorage: { ...localStorage },
      performance: performance.getMetrics(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      }
    };
    
    console.group('🔍 Aetherium State Dump');
    console.table(state);
    console.log('📋 State JSON:', JSON.stringify(state, null, 2));
    console.groupEnd();
    
    return state;
  },
  
  testConnections: async () => {
    console.group('🧪 Testing Connections');
    
    try {
      performance.startMeasure('API Health Check');
      const health = await apiService.getSystemHealth();
      performance.endMeasure('API Health Check');
      console.log('✅ API Connection:', health.status);
    } catch (error) {
      console.error('❌ API Connection failed:', error);
    }
    
    try {
      const wsState = websocketService.getConnectionState();
      console.log(`🔌 WebSocket State: ${wsState}`);
      
      if (wsState !== 'connected') {
        console.log('🔄 Attempting WebSocket connection...');
        const connected = await websocketService.connect();
        console.log(connected ? '✅ WebSocket Connected' : '❌ WebSocket Failed');
      }
    } catch (error) {
      console.error('❌ WebSocket test failed:', error);
    }
    
    console.groupEnd();
  },
  
  simulateError: (component: string) => {
    console.warn(`🧪 Simulating error in ${component}`);
    
    switch (component) {
      case 'websocket':
        websocketService.disconnect();
        break;
      case 'storage':
        localStorage.clear();
        break;
      default:
        throw new Error(`Simulated error in ${component}`);
    }
  },
  
  clearStorage: () => {
    const confirm = window.confirm('⚠️ This will clear all local storage. Continue?');
    if (confirm) {
      localStorage.clear();
      sessionStorage.clear();
      console.log('🧹 Local storage cleared');
      window.location.reload();
    }
  }
};

// Initialize development tools
export function initDevTools(): void {
  if (import.meta.env.DEV) {
    const devTools: DevTools = {
      api: apiService,
      aiModels: aiModelsService,
      websocket: websocketService,
      debug,
      performance
    };
    
    (window as any).AetheriumDevTools = devTools;
    
    console.log('🛠️ Aetherium Development Tools loaded');
    console.log('📋 Access via: AetheriumDevTools');
    
    if (localStorage.getItem('aetherium_debug') === 'true') {
      debug.enableVerboseLogging();
    }
    
    performance.startMeasure('App Initialize');
    
    window.addEventListener('aetherium-ready', () => {
      performance.endMeasure('App Initialize');
      console.log('🎉 Aetherium fully loaded and ready!');
    });
  }
}

export { performance, debug };
export default initDevTools;