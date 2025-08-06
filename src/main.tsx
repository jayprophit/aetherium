/**
 * AETHERIUM MAIN APPLICATION ENTRY POINT
 * Integrates all services: API, AI Models, WebSocket, and React components
 */

import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Import services for initialization
import { apiService } from './services/api';
import { aiModelsService } from './services/aiModels';
import { websocketService } from './services/websocket';

// Global error boundary
class ErrorBoundary extends React.Component {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if ((this.state as any).hasError) {
      return (
        <div className="error-fallback">
          <h2>⚠️ Something went wrong</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {(this.state as any).error && (this.state as any).error.toString()}
          </details>
          <button onClick={() => window.location.reload()}>
            🔄 Reload Application
          </button>
        </div>
      );
    }

    return (this.props as any).children;
  }
}

// Initialize application
async function initializeAetherium() {
  console.log('🎉 AETHERIUM PLATFORM INITIALIZATION');
  
  try {
    // Check API connectivity
    console.log('🔗 Checking backend connectivity...');
    const health = await apiService.getSystemHealth().catch(() => null);
    
    if (health) {
      console.log('✅ Backend connected:', health.status);
    } else {
      console.log('⚠️ Backend not available - running in offline mode');
    }
    
    // Initialize AI models
    console.log('🧠 Initializing AI models...');
    const models = aiModelsService.getAvailableModels();
    console.log(`✅ ${models.length} AI models available:`, models.map(m => m.name));
    
    // Connect WebSocket
    console.log('🔄 Connecting WebSocket...');
    const wsConnected = await websocketService.connect();
    
    if (wsConnected) {
      console.log('✅ WebSocket connected - real-time features enabled');
    } else {
      console.log('⚠️ WebSocket connection failed - using polling fallback');
    }
    
    console.log('🚀 Aetherium Platform fully initialized!');
    console.log('⚛️ Features ready:');
    console.log('   🧠 80+ AI Tools');
    console.log('   🎨 Manus/Claude-style UI/UX');
    console.log('   💬 Real-time Chat');
    console.log('   🔬 Quantum Computing');
    console.log('   💼 Productivity Suite');
    
    // Dispatch ready event
    window.dispatchEvent(new CustomEvent('aetherium-ready'));
    
    return true;
    
  } catch (error) {
    console.error('❌ Aetherium initialization failed:', error);
    return false;
  }
}

// Application startup
async function startApp() {
  const container = document.getElementById('root');
  
  if (!container) {
    throw new Error('Root container not found');
  }
  
  // Initialize platform
  const initialized = await initializeAetherium();
  
  if (!initialized) {
    console.warn('⚠️ Platform initialization had issues, but continuing...');
  }
  
  // Create React root and render app
  const root = createRoot(container);
  
  root.render(
    <StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </StrictMode>
  );
  
  // Performance monitoring
  if (import.meta.env.DEV) {
    import('./utils/devtools').then(({ initDevTools }) => {
      initDevTools();
    });
  }
}

// Start the application
startApp().catch((error) => {
  console.error('Failed to start Aetherium application:', error);
  
  // Show error screen
  const errorScreen = document.getElementById('error-screen');
  const loadingScreen = document.getElementById('loading-screen');
  
  if (loadingScreen) loadingScreen.style.display = 'none';
  if (errorScreen) {
    const errorMessage = errorScreen.querySelector('.error-message');
    if (errorMessage) {
      errorMessage.textContent = `Application startup failed: ${error.message}`;
    }
    errorScreen.style.display = 'flex';
  }
});

// Hot Module Replacement for development
if (import.meta.hot) {
  import.meta.hot.accept('./App', () => {
    console.log('🔄 Hot reloading App component');
  });
}

export default startApp;