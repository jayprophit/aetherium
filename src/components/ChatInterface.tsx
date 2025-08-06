/**
 * AETHERIUM CHAT INTERFACE COMPONENT
 * Real-time chat interface with full service integration
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, Paperclip, MoreHorizontal } from 'lucide-react';
import { useAppContext } from '../App';
import { useChat } from '../hooks/useAetherium';

interface ChatInterfaceProps {
  darkMode: boolean;
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  darkMode,
  selectedModel,
  onModelChange
}) => {
  const { availableModels, isWebSocketConnected } = useAppContext();
  const { messages, isLoading, error, sendMessage } = useChat();
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Handle send message
  const handleSendMessage = async () => {
    if (inputMessage.trim() && !isLoading) {
      await sendMessage(inputMessage, selectedModel);
      setInputMessage('');
      inputRef.current?.focus();
    }
  };

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Connection Status */}
      <div className={`px-4 py-2 border-b ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className={`h-2 w-2 rounded-full ${isWebSocketConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              {isWebSocketConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          {/* Model Selection */}
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className={`text-xs px-2 py-1 rounded border ${
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
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🔮</div>
            <h3 className={`text-xl font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              Welcome to Aetherium AI
            </h3>
            <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Your quantum-enhanced AI productivity platform is ready
            </p>
            <div className="mt-4 text-sm text-gray-500">
              Try asking: "Help me analyze data" or "Create a presentation"
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-lg p-3 ${
                  msg.role === 'user' 
                    ? `${darkMode ? 'bg-purple-600' : 'bg-purple-500'} text-white`
                    : msg.role === 'system'
                    ? `${darkMode ? 'bg-blue-600' : 'bg-blue-500'} text-white`
                    : `${darkMode ? 'bg-gray-700' : 'bg-gray-200'} ${darkMode ? 'text-white' : 'text-gray-900'}`
                }`}>
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                  <div className="text-xs mt-2 opacity-70">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                    {msg.metadata?.model && (
                      <span className="ml-2">• {msg.metadata.model}</span>
                    )}
                    {msg.metadata?.executionTime && (
                      <span className="ml-2">• {msg.metadata.executionTime}ms</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {/* Loading indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-200'} rounded-lg p-3`}>
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-500"></div>
                    <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      AI is thinking...
                    </span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        
        {/* Error display */}
        {error && (
          <div className={`p-3 rounded-lg border ${darkMode ? 'bg-red-900 border-red-700 text-red-200' : 'bg-red-50 border-red-200 text-red-800'}`}>
            <div className="flex items-center space-x-2">
              <span className="text-sm">⚠️ {error}</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className={`p-4 border-t ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'}`}>
        <div className="flex items-center space-x-3">
          <button 
            className={`p-2 rounded-lg transition-colors ${
              darkMode ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-200 text-gray-600 hover:text-gray-900'
            }`}
          >
            <Paperclip className="w-4 h-4" />
          </button>
          
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask Aetherium AI anything..."
              disabled={isLoading}
              className={`w-full px-4 py-3 rounded-lg border resize-none transition-colors ${
                darkMode 
                  ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:border-purple-500' 
                  : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-purple-500'
              } focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50`}
            />
          </div>
          
          <button 
            className={`p-2 rounded-lg transition-colors ${
              darkMode ? 'hover:bg-gray-700 text-gray-400 hover:text-white' : 'hover:bg-gray-200 text-gray-600 hover:text-gray-900'
            }`}
          >
            <Mic className="w-4 h-4" />
          </button>
          
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              inputMessage.trim() && !isLoading
                ? 'bg-purple-600 hover:bg-purple-700 text-white'
                : `${darkMode ? 'bg-gray-700 text-gray-500' : 'bg-gray-200 text-gray-400'} cursor-not-allowed`
            }`}
          >
            {isLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        
        <div className={`text-xs mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;