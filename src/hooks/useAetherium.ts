/**
 * AETHERIUM PLATFORM HOOKS
 * Custom React hooks for Aetherium platform functionality
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAppContext } from '../App';
import { storageService, ChatMessage, ChatSession, UserPreferences } from '../services/storage';

// ==========================================
// CHAT MANAGEMENT HOOK
// ==========================================

export function useChat(sessionId?: string) {
  const { websocketService, apiService, currentChatSession } = useAppContext();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const activeSessionId = sessionId || currentChatSession?.id || 'default';

  // Load messages on session change
  useEffect(() => {
    const savedMessages = storageService.getChatMessages(activeSessionId);
    setMessages(savedMessages);
  }, [activeSessionId]);

  // Send message function
  const sendMessage = useCallback(async (content: string, model?: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      content: content.trim(),
      role: 'user',
      timestamp: new Date().toISOString(),
      sessionId: activeSessionId,
      metadata: { model: model || 'aetherium-quantum-1' }
    };

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);
    storageService.saveChatMessage(userMessage);
    setIsLoading(true);
    setError(null);

    try {
      // Send via WebSocket if connected, otherwise API
      if (websocketService.isConnected()) {
        await websocketService.sendChatMessage(activeSessionId, content, model);
      } else {
        const response = await apiService.sendChatMessage({
          message: content,
          sessionId: activeSessionId,
          model: model || 'aetherium-quantum-1'
        });

        const assistantMessage: ChatMessage = {
          id: response.id,
          content: response.content,
          role: 'assistant',
          timestamp: response.timestamp,
          sessionId: activeSessionId,
          metadata: response.metadata
        };

        setMessages(prev => [...prev, assistantMessage]);
        storageService.saveChatMessage(assistantMessage);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      
      const errorChatMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        content: `Sorry, I encountered an error: ${errorMessage}`,
        role: 'assistant',
        timestamp: new Date().toISOString(),
        sessionId: activeSessionId
      };
      
      setMessages(prev => [...prev, errorChatMessage]);
      storageService.saveChatMessage(errorChatMessage);
    } finally {
      setIsLoading(false);
    }
  }, [activeSessionId, isLoading, websocketService, apiService]);

  // Clear messages
  const clearMessages = useCallback(() => {
    setMessages([]);
    storageService.deleteChatMessages(activeSessionId);
  }, [activeSessionId]);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    sessionId: activeSessionId
  };
}

// ==========================================
// USER PREFERENCES HOOK
// ==========================================

export function usePreferences() {
  const [preferences, setPreferencesState] = useState<UserPreferences>(() => 
    storageService.getUserPreferences()
  );

  const updatePreferences = useCallback((updates: Partial<UserPreferences>) => {
    const newPreferences = { ...preferences, ...updates };
    setPreferencesState(newPreferences);
    storageService.saveUserPreferences(newPreferences);
  }, [preferences]);

  return {
    preferences,
    updatePreferences
  };
}

// ==========================================
// AI TOOLS HOOK
// ==========================================

export function useAITools() {
  const { apiService } = useAppContext();
  const [availableTools, setAvailableTools] = useState<any[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);

  // Load available tools
  useEffect(() => {
    const loadTools = async () => {
      try {
        const tools = await apiService.getAvailableTools();
        setAvailableTools(tools);
      } catch (error) {
        console.error('Failed to load AI tools:', error);
      }
    };
    
    loadTools();
  }, [apiService]);

  // Execute tool
  const executeTool = useCallback(async (toolId: string, parameters: any = {}) => {
    setIsExecuting(true);
    
    try {
      const result = await apiService.executeTool(toolId, parameters);
      return result;
    } catch (error) {
      console.error(`Tool execution failed for ${toolId}:`, error);
      throw error;
    } finally {
      setIsExecuting(false);
    }
  }, [apiService]);

  return {
    availableTools,
    isExecuting,
    executeTool
  };
}

// ==========================================
// WEBSOCKET CONNECTION HOOK
// ==========================================

export function useWebSocketConnection() {
  const { websocketService } = useAppContext();
  const [connectionState, setConnectionState] = useState<string>('disconnected');
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    // Monitor connection state
    const checkConnection = () => {
      const state = websocketService.getConnectionState();
      setConnectionState(state);
    };

    // Set up message handler
    const handleMessage = (message: any) => {
      setLastMessage(message);
    };

    websocketService.onMessage('connection-hook', handleMessage);
    intervalId = setInterval(checkConnection, 3000);
    checkConnection(); // Initial check

    return () => {
      clearInterval(intervalId);
      websocketService.removeMessageHandler('connection-hook');
    };
  }, [websocketService]);

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    lastMessage
  };
}

// Export all hooks
export default {
  useChat,
  usePreferences,
  useAITools,
  useWebSocketConnection
};