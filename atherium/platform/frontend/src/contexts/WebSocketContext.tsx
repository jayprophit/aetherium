import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useAuth } from './AuthContext';

interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

interface WebSocketContextType {
  socket: WebSocket | null;
  isConnected: boolean;
  messages: WebSocketMessage[];
  sendMessage: (type: string, data: any) => void;
  subscribe: (type: string, callback: (data: any) => void) => () => void;
  clearMessages: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

interface WebSocketProviderProps {
  children: React.ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const { isAuthenticated } = useAuth();
  const subscriptionsRef = useRef<Map<string, ((data: any) => void)[]>>(new Map());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  useEffect(() => {
    if (isAuthenticated) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [isAuthenticated]);

  const connectWebSocket = () => {
    try {
      const wsUrl = `ws://localhost:8000/ws`;
      const token = localStorage.getItem('auth_token');
      
      const newSocket = new WebSocket(`${wsUrl}?token=${token}`);
      
      newSocket.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setSocket(newSocket);
        reconnectAttempts.current = 0;
      };

      newSocket.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setMessages(prev => [...prev.slice(-99), message]); // Keep last 100 messages
          
          // Notify subscribers
          const subscribers = subscriptionsRef.current.get(message.type) || [];
          subscribers.forEach(callback => callback(message.data));
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      newSocket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setSocket(null);
        
        // Attempt to reconnect if not a normal closure
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.pow(2, reconnectAttempts.current) * 1000; // Exponential backoff
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++;
            connectWebSocket();
          }, delay);
        }
      };

      newSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setIsConnected(false);
    }
  };

  const disconnectWebSocket = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (socket) {
      socket.close(1000, 'User logged out');
      setSocket(null);
    }
    
    setIsConnected(false);
    reconnectAttempts.current = 0;
  };

  const sendMessage = (type: string, data: any) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      const message = {
        type,
        data,
        timestamp: new Date().toISOString()
      };
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', { type, data });
    }
  };

  const subscribe = (type: string, callback: (data: any) => void) => {
    const currentSubscribers = subscriptionsRef.current.get(type) || [];
    subscriptionsRef.current.set(type, [...currentSubscribers, callback]);
    
    // Return unsubscribe function
    return () => {
      const subscribers = subscriptionsRef.current.get(type) || [];
      const updatedSubscribers = subscribers.filter(cb => cb !== callback);
      if (updatedSubscribers.length > 0) {
        subscriptionsRef.current.set(type, updatedSubscribers);
      } else {
        subscriptionsRef.current.delete(type);
      }
    };
  };

  const clearMessages = () => {
    setMessages([]);
  };

  const value = {
    socket,
    isConnected,
    messages,
    sendMessage,
    subscribe,
    clearMessages,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};