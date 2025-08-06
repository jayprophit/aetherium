/**
 * AETHERIUM WEBSOCKET SERVICE
 * Real-time WebSocket integration for live chat and system updates
 */

import { ChatMessage } from './api';
import { aiModelsService } from './aiModels';

export interface WebSocketMessage {
  type: 'chat' | 'system' | 'tool_result' | 'thinking' | 'status';
  data: any;
  timestamp: string;
  id?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  model: string;
  createdAt: string;
  updatedAt: string;
}

export interface SystemStatusUpdate {
  component: string;
  status: 'online' | 'offline' | 'degraded';
  message?: string;
  timestamp: string;
}

// WebSocket Chat Service
class WebSocketService {
  private websocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  private isConnecting = false;
  private messageHandlers: Map<string, (message: WebSocketMessage) => void> = new Map();
  private chatSessions: Map<string, ChatSession> = new Map();
  private currentSessionId: string | null = null;

  constructor() {
    this.loadChatSessions();
  }

  // Initialize WebSocket connection
  async connect(): Promise<boolean> {
    if (this.isConnecting || (this.websocket && this.websocket.readyState === WebSocket.OPEN)) {
      return true;
    }

    this.isConnecting = true;
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    
    try {
      console.log('🔄 Connecting to Aetherium WebSocket...');
      
      this.websocket = new WebSocket(`${wsUrl}/ws/chat`);
      
      this.websocket.onopen = () => {
        console.log('✅ WebSocket connected to Aetherium backend');
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.sendMessage({
          type: 'system',
          data: { action: 'join', timestamp: new Date().toISOString() },
          timestamp: new Date().toISOString()
        });
      };

      this.websocket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onclose = (event) => {
        console.log(`WebSocket disconnected: ${event.code} - ${event.reason}`);
        this.isConnecting = false;
        this.websocket = null;
        
        // Auto-reconnect if not intentionally closed
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => {
            this.reconnectAttempts++;
            console.log(`🔄 Reconnecting... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            this.connect();
          }, this.reconnectDelay * this.reconnectAttempts);
        }
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.isConnecting = false;
      };

      return new Promise((resolve) => {
        const checkConnection = () => {
          if (this.websocket?.readyState === WebSocket.OPEN) {
            resolve(true);
          } else if (this.websocket?.readyState === WebSocket.CLOSED) {
            resolve(false);
          } else {
            setTimeout(checkConnection, 100);
          }
        };
        checkConnection();
      });

    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.isConnecting = false;
      return false;
    }
  }

  // Disconnect WebSocket
  disconnect(): void {
    if (this.websocket) {
      this.websocket.close(1000, 'Client disconnect');
      this.websocket = null;
    }
  }

  // Send message through WebSocket
  private sendMessage(message: WebSocketMessage): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  // Handle incoming WebSocket messages
  private handleMessage(message: WebSocketMessage): void {
    // Notify registered handlers
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        console.error('Message handler error:', error);
      }
    });

    // Handle specific message types
    switch (message.type) {
      case 'chat':
        this.handleChatMessage(message.data as ChatMessage);
        break;
      case 'system':
        this.handleSystemMessage(message.data);
        break;
      case 'thinking':
        this.handleThinkingMessage(message.data);
        break;
      case 'tool_result':
        this.handleToolResult(message.data);
        break;
      case 'status':
        this.handleStatusUpdate(message.data as SystemStatusUpdate);
        break;
    }
  }

  // Handle chat messages
  private handleChatMessage(chatMessage: ChatMessage): void {
    if (this.currentSessionId) {
      const session = this.chatSessions.get(this.currentSessionId);
      if (session) {
        session.messages.push(chatMessage);
        session.updatedAt = new Date().toISOString();
        this.saveChatSessions();
      }
    }
  }

  // Handle system messages
  private handleSystemMessage(data: any): void {
    console.log('System message:', data);
  }

  // Handle AI thinking process
  private handleThinkingMessage(data: any): void {
    // Update UI with AI thinking process
    const event = new CustomEvent('ai-thinking', { detail: data });
    window.dispatchEvent(event);
  }

  // Handle tool execution results
  private handleToolResult(data: any): void {
    const event = new CustomEvent('tool-result', { detail: data });
    window.dispatchEvent(event);
  }

  // Handle system status updates
  private handleStatusUpdate(status: SystemStatusUpdate): void {
    const event = new CustomEvent('system-status', { detail: status });
    window.dispatchEvent(event);
  }

  // Register message handler
  onMessage(handlerId: string, handler: (message: WebSocketMessage) => void): void {
    this.messageHandlers.set(handlerId, handler);
  }

  // Unregister message handler
  offMessage(handlerId: string): void {
    this.messageHandlers.delete(handlerId);
  }

  // Send chat message
  async sendChatMessage(
    message: string, 
    modelId: string = 'quantum-1',
    sessionId?: string
  ): Promise<void> {
    // Create or get session
    const session = sessionId ? 
      this.getSession(sessionId) : 
      this.createNewSession(modelId);

    // Add user message to session
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      content: message,
      role: 'user',
      timestamp: new Date().toISOString()
    };

    session.messages.push(userMessage);
    this.currentSessionId = session.id;

    // Send message through WebSocket
    this.sendMessage({
      type: 'chat',
      data: {
        message,
        modelId,
        sessionId: session.id,
        context: session.messages.slice(-10) // Send last 10 messages for context
      },
      timestamp: new Date().toISOString(),
      id: userMessage.id
    });

    // Show AI thinking
    this.sendMessage({
      type: 'thinking',
      data: {
        sessionId: session.id,
        status: 'processing',
        model: modelId
      },
      timestamp: new Date().toISOString()
    });

    this.saveChatSessions();
  }

  // Execute AI tool
  async executeTool(
    toolId: string, 
    parameters: any,
    sessionId?: string
  ): Promise<void> {
    const session = sessionId ? 
      this.getSession(sessionId) : 
      this.getCurrentSession();

    if (!session) {
      throw new Error('No active session');
    }

    this.sendMessage({
      type: 'tool_result',
      data: {
        toolId,
        parameters,
        sessionId: session.id
      },
      timestamp: new Date().toISOString()
    });
  }

  // Create new chat session
  createNewSession(modelId: string = 'quantum-1'): ChatSession {
    const session: ChatSession = {
      id: crypto.randomUUID(),
      title: `Chat ${new Date().toLocaleString()}`,
      messages: [],
      model: modelId,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    this.chatSessions.set(session.id, session);
    this.currentSessionId = session.id;
    this.saveChatSessions();

    return session;
  }

  // Get session by ID
  getSession(sessionId: string): ChatSession {
    let session = this.chatSessions.get(sessionId);
    if (!session) {
      session = this.createNewSession();
    }
    return session;
  }

  // Get current session
  getCurrentSession(): ChatSession | null {
    return this.currentSessionId ? this.getSession(this.currentSessionId) : null;
  }

  // Get all sessions
  getAllSessions(): ChatSession[] {
    return Array.from(this.chatSessions.values())
      .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
  }

  // Switch to session
  switchToSession(sessionId: string): void {
    this.currentSessionId = sessionId;
  }

  // Delete session
  deleteSession(sessionId: string): void {
    this.chatSessions.delete(sessionId);
    if (this.currentSessionId === sessionId) {
      this.currentSessionId = null;
    }
    this.saveChatSessions();
  }

  // Update session title
  updateSessionTitle(sessionId: string, title: string): void {
    const session = this.chatSessions.get(sessionId);
    if (session) {
      session.title = title;
      session.updatedAt = new Date().toISOString();
      this.saveChatSessions();
    }
  }

  // Save chat sessions to local storage
  private saveChatSessions(): void {
    try {
      const sessions = Object.fromEntries(this.chatSessions.entries());
      localStorage.setItem('aetherium_chat_sessions', JSON.stringify(sessions));
    } catch (error) {
      console.error('Failed to save chat sessions:', error);
    }
  }

  // Load chat sessions from local storage
  private loadChatSessions(): void {
    try {
      const stored = localStorage.getItem('aetherium_chat_sessions');
      if (stored) {
        const sessions = JSON.parse(stored);
        Object.entries(sessions).forEach(([id, session]) => {
          this.chatSessions.set(id, session as ChatSession);
        });
      }
    } catch (error) {
      console.error('Failed to load chat sessions:', error);
    }
  }

  // Get connection status
  isConnected(): boolean {
    return this.websocket?.readyState === WebSocket.OPEN;
  }

  // Get connection state
  getConnectionState(): string {
    if (!this.websocket) return 'disconnected';
    
    switch (this.websocket.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'disconnected';
      default: return 'unknown';
    }
  }

  // Stream AI response (for streaming responses)
  streamResponse(sessionId: string, onChunk: (chunk: string) => void): void {
    const handler = (message: WebSocketMessage) => {
      if (message.type === 'chat' && message.data.sessionId === sessionId) {
        if (message.data.streaming) {
          onChunk(message.data.chunk);
        }
      }
    };

    const handlerId = `stream_${sessionId}`;
    this.onMessage(handlerId, handler);

    // Clean up after streaming completes
    setTimeout(() => {
      this.offMessage(handlerId);
    }, 30000); // 30 second timeout
  }
}

// Create singleton instance
export const websocketService = new WebSocketService();

// React hooks for WebSocket integration
export const useWebSocket = () => {
  return websocketService;
};

export const useChat = () => {
  const ws = websocketService;
  
  return {
    sendMessage: ws.sendChatMessage.bind(ws),
    createSession: ws.createNewSession.bind(ws),
    getSessions: ws.getAllSessions.bind(ws),
    getCurrentSession: ws.getCurrentSession.bind(ws),
    switchSession: ws.switchToSession.bind(ws),
    deleteSession: ws.deleteSession.bind(ws),
    updateTitle: ws.updateSessionTitle.bind(ws),
    isConnected: ws.isConnected.bind(ws),
    connect: ws.connect.bind(ws),
    disconnect: ws.disconnect.bind(ws)
  };
};

export default websocketService;