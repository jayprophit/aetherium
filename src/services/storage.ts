/**
 * AETHERIUM STORAGE SERVICE
 * Real-time persistence and storage management
 */

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
  sessionId: string;
  metadata?: {
    model?: string;
    tokens?: number;
    executionTime?: number;
    toolUsed?: string;
  };
}

export interface ChatSession {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  model: string;
  metadata?: {
    tags?: string[];
    archived?: boolean;
    starred?: boolean;
  };
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  language: string;
  defaultModel: string;
  enableAnimations: boolean;
  enableQuantumEffects: boolean;
  sidebarWidth: number;
  autoSave: boolean;
  enableVoiceInput: boolean;
  enableNotifications: boolean;
}

export interface FileMetadata {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: string;
  sessionId?: string;
  tags?: string[];
  description?: string;
}

class StorageService {
  private storagePrefix: string;
  private maxChatHistory: number;
  private autoSaveInterval: number;
  private autoSaveTimer?: NodeJS.Timeout;

  constructor() {
    this.storagePrefix = import.meta.env.REACT_APP_LOCAL_STORAGE_PREFIX || 'aetherium_';
    this.maxChatHistory = parseInt(import.meta.env.REACT_APP_CHAT_HISTORY_LIMIT || '1000');
    this.autoSaveInterval = parseInt(import.meta.env.REACT_APP_AUTO_SAVE_INTERVAL || '5000');
    
    this.initializeAutoSave();
  }

  // ==========================================
  // CHAT PERSISTENCE
  // ==========================================

  /**
   * Save chat message to persistent storage
   */
  saveChatMessage(message: ChatMessage): void {
    try {
      const messages = this.getChatMessages(message.sessionId);
      messages.push(message);
      
      // Limit message history
      if (messages.length > this.maxChatHistory) {
        messages.splice(0, messages.length - this.maxChatHistory);
      }
      
      this.setItem(`chat_messages_${message.sessionId}`, messages);
      this.updateSessionTimestamp(message.sessionId);
      
      console.log(`💾 Chat message saved: ${message.id}`);
    } catch (error) {
      console.error('Failed to save chat message:', error);
    }
  }

  /**
   * Get all chat messages for a session
   */
  getChatMessages(sessionId: string): ChatMessage[] {
    try {
      return this.getItem(`chat_messages_${sessionId}`, []);
    } catch (error) {
      console.error('Failed to get chat messages:', error);
      return [];
    }
  }

  /**
   * Delete chat messages for a session
   */
  deleteChatMessages(sessionId: string): void {
    try {
      this.removeItem(`chat_messages_${sessionId}`);
      console.log(`🗑️ Chat messages deleted for session: ${sessionId}`);
    } catch (error) {
      console.error('Failed to delete chat messages:', error);
    }
  }

  /**
   * Search chat messages
   */
  searchChatMessages(query: string, sessionId?: string): ChatMessage[] {
    try {
      const sessions = sessionId ? [sessionId] : this.getAllChatSessions().map(s => s.id);
      const results: ChatMessage[] = [];
      
      for (const sid of sessions) {
        const messages = this.getChatMessages(sid);
        const matching = messages.filter(msg => 
          msg.content.toLowerCase().includes(query.toLowerCase())
        );
        results.push(...matching);
      }
      
      return results.sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    } catch (error) {
      console.error('Failed to search chat messages:', error);
      return [];
    }
  }

  // ==========================================
  // SESSION MANAGEMENT
  // ==========================================

  /**
   * Save chat session
   */
  saveChatSession(session: ChatSession): void {
    try {
      const sessions = this.getAllChatSessions();
      const existingIndex = sessions.findIndex(s => s.id === session.id);
      
      if (existingIndex >= 0) {
        sessions[existingIndex] = { ...session, updatedAt: new Date().toISOString() };
      } else {
        sessions.push(session);
      }
      
      this.setItem('chat_sessions', sessions);
      console.log(`💾 Chat session saved: ${session.id}`);
    } catch (error) {
      console.error('Failed to save chat session:', error);
    }
  }

  /**
   * Get all chat sessions
   */
  getAllChatSessions(): ChatSession[] {
    try {
      return this.getItem('chat_sessions', []);
    } catch (error) {
      console.error('Failed to get chat sessions:', error);
      return [];
    }
  }

  /**
   * Delete chat session
   */
  deleteChatSession(sessionId: string): void {
    try {
      const sessions = this.getAllChatSessions().filter(s => s.id !== sessionId);
      this.setItem('chat_sessions', sessions);
      this.deleteChatMessages(sessionId);
      console.log(`🗑️ Chat session deleted: ${sessionId}`);
    } catch (error) {
      console.error('Failed to delete chat session:', error);
    }
  }

  /**
   * Update session timestamp
   */
  private updateSessionTimestamp(sessionId: string): void {
    const sessions = this.getAllChatSessions();
    const sessionIndex = sessions.findIndex(s => s.id === sessionId);
    
    if (sessionIndex >= 0) {
      sessions[sessionIndex].updatedAt = new Date().toISOString();
      sessions[sessionIndex].messageCount = this.getChatMessages(sessionId).length;
      this.setItem('chat_sessions', sessions);
    }
  }

  // ==========================================
  // USER PREFERENCES
  // ==========================================

  /**
   * Save user preferences
   */
  saveUserPreferences(preferences: Partial<UserPreferences>): void {
    try {
      const current = this.getUserPreferences();
      const updated = { ...current, ...preferences };
      this.setItem('user_preferences', updated);
      console.log('💾 User preferences saved');
    } catch (error) {
      console.error('Failed to save user preferences:', error);
    }
  }

  /**
   * Get user preferences with defaults
   */
  getUserPreferences(): UserPreferences {
    try {
      return this.getItem('user_preferences', {
        theme: 'dark',
        language: 'en',
        defaultModel: 'aetherium-quantum-1',
        enableAnimations: true,
        enableQuantumEffects: true,
        sidebarWidth: 280,
        autoSave: true,
        enableVoiceInput: false,
        enableNotifications: true
      });
    } catch (error) {
      console.error('Failed to get user preferences:', error);
      return {
        theme: 'dark',
        language: 'en',
        defaultModel: 'aetherium-quantum-1',
        enableAnimations: true,
        enableQuantumEffects: true,
        sidebarWidth: 280,
        autoSave: true,
        enableVoiceInput: false,
        enableNotifications: true
      };
    }
  }

  // ==========================================
  // FILE MANAGEMENT
  // ==========================================

  /**
   * Save file metadata
   */
  saveFileMetadata(metadata: FileMetadata): void {
    try {
      const files = this.getAllFileMetadata();
      files.push(metadata);
      this.setItem('file_metadata', files);
      console.log(`💾 File metadata saved: ${metadata.name}`);
    } catch (error) {
      console.error('Failed to save file metadata:', error);
    }
  }

  /**
   * Get all file metadata
   */
  getAllFileMetadata(): FileMetadata[] {
    try {
      return this.getItem('file_metadata', []);
    } catch (error) {
      console.error('Failed to get file metadata:', error);
      return [];
    }
  }

  /**
   * Delete file metadata
   */
  deleteFileMetadata(fileId: string): void {
    try {
      const files = this.getAllFileMetadata().filter(f => f.id !== fileId);
      this.setItem('file_metadata', files);
      console.log(`🗑️ File metadata deleted: ${fileId}`);
    } catch (error) {
      console.error('Failed to delete file metadata:', error);
    }
  }

  // ==========================================
  // AUTO-SAVE & SYNC
  // ==========================================

  /**
   * Initialize auto-save functionality
   */
  private initializeAutoSave(): void {
    const autoSaveEnabled = import.meta.env.REACT_APP_AUTO_SAVE_INTERVAL !== '0';
    
    if (autoSaveEnabled) {
      this.autoSaveTimer = setInterval(() => {
        this.performAutoSave();
      }, this.autoSaveInterval);
      
      console.log(`💾 Auto-save initialized (${this.autoSaveInterval}ms interval)`);
    }
  }

  /**
   * Perform auto-save operations
   */
  private performAutoSave(): void {
    try {
      // Trigger custom auto-save event for components to respond to
      window.dispatchEvent(new CustomEvent('aetherium-autosave'));
    } catch (error) {
      console.error('Auto-save failed:', error);
    }
  }

  /**
   * Export all data for backup
   */
  exportData(): object {
    try {
      const data = {
        sessions: this.getAllChatSessions(),
        preferences: this.getUserPreferences(),
        files: this.getAllFileMetadata(),
        exportedAt: new Date().toISOString(),
        version: '1.0.0'
      };
      
      // Add chat messages for each session
      const messagesData: { [key: string]: ChatMessage[] } = {};
      for (const session of data.sessions) {
        messagesData[session.id] = this.getChatMessages(session.id);
      }
      
      return { ...data, messages: messagesData };
    } catch (error) {
      console.error('Failed to export data:', error);
      return {};
    }
  }

  /**
   * Import data from backup
   */
  importData(data: any): boolean {
    try {
      if (data.sessions) {
        this.setItem('chat_sessions', data.sessions);
      }
      
      if (data.preferences) {
        this.setItem('user_preferences', data.preferences);
      }
      
      if (data.files) {
        this.setItem('file_metadata', data.files);
      }
      
      if (data.messages) {
        for (const [sessionId, messages] of Object.entries(data.messages)) {
          this.setItem(`chat_messages_${sessionId}`, messages);
        }
      }
      
      console.log('💾 Data import completed successfully');
      return true;
    } catch (error) {
      console.error('Failed to import data:', error);
      return false;
    }
  }

  /**
   * Clear all stored data
   */
  clearAllData(): void {
    try {
      const keys = Object.keys(localStorage).filter(key => 
        key.startsWith(this.storagePrefix)
      );
      
      for (const key of keys) {
        localStorage.removeItem(key);
      }
      
      console.log('🧹 All storage data cleared');
    } catch (error) {
      console.error('Failed to clear storage data:', error);
    }
  }

  // ==========================================
  // UTILITY METHODS
  // ==========================================

  /**
   * Set item in storage with prefix
   */
  private setItem(key: string, value: any): void {
    localStorage.setItem(`${this.storagePrefix}${key}`, JSON.stringify(value));
  }

  /**
   * Get item from storage with prefix
   */
  private getItem<T>(key: string, defaultValue: T): T {
    try {
      const item = localStorage.getItem(`${this.storagePrefix}${key}`);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`Failed to parse stored item: ${key}`, error);
      return defaultValue;
    }
  }

  /**
   * Remove item from storage
   */
  private removeItem(key: string): void {
    localStorage.removeItem(`${this.storagePrefix}${key}`);
  }

  /**
   * Get storage usage statistics
   */
  getStorageStats(): {
    totalSize: number;
    itemCount: number;
    sessions: number;
    messages: number;
    files: number;
  } {
    try {
      let totalSize = 0;
      let itemCount = 0;
      
      const keys = Object.keys(localStorage).filter(key => 
        key.startsWith(this.storagePrefix)
      );
      
      for (const key of keys) {
        const value = localStorage.getItem(key);
        if (value) {
          totalSize += value.length;
          itemCount++;
        }
      }
      
      return {
        totalSize,
        itemCount,
        sessions: this.getAllChatSessions().length,
        messages: keys.filter(k => k.includes('chat_messages_')).length,
        files: this.getAllFileMetadata().length
      };
    } catch (error) {
      console.error('Failed to get storage stats:', error);
      return {
        totalSize: 0,
        itemCount: 0,
        sessions: 0,
        messages: 0,
        files: 0
      };
    }
  }

  /**
   * Cleanup old data
   */
  cleanup(): void {
    try {
      const sessions = this.getAllChatSessions();
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - 30); // Keep last 30 days
      
      let cleaned = 0;
      for (const session of sessions) {
        if (new Date(session.updatedAt) < cutoffDate && !session.metadata?.starred) {
          this.deleteChatSession(session.id);
          cleaned++;
        }
      }
      
      console.log(`🧹 Cleaned up ${cleaned} old chat sessions`);
    } catch (error) {
      console.error('Failed to cleanup storage:', error);
    }
  }

  /**
   * Dispose of the service
   */
  dispose(): void {
    if (this.autoSaveTimer) {
      clearInterval(this.autoSaveTimer);
      this.autoSaveTimer = undefined;
    }
  }
}

// Create and export singleton instance
export const storageService = new StorageService();

// Export types
export type { ChatMessage, ChatSession, UserPreferences, FileMetadata };
export default storageService;