/**
 * AETHERIUM API SERVICE
 * Frontend-Backend Integration Layer
 * Connects React frontend to FastAPI backend
 */

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

// Types
export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
  thinking?: string;
  toolUsed?: string;
}

export interface AITool {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  endpoint: string;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    quantum_computer: boolean;
    time_crystal_engine: boolean;
    neuromorphic_processor: boolean;
    database: boolean;
    authentication: boolean;
  };
  uptime: string;
  timestamp: string;
}

export interface SystemMetrics {
  quantum_metrics: any;
  time_crystal_metrics: any;
  neuromorphic_metrics: any;
  ai_ml_metrics: any;
  iot_metrics: any;
  system_metrics: any;
  timestamp: string;
}

// API Service Class
class APIService {
  private baseUrl: string;
  private wsUrl: string;
  private token: string | null = null;
  private websocket: WebSocket | null = null;

  constructor() {
    this.baseUrl = API_BASE_URL;
    this.wsUrl = WS_BASE_URL;
  }

  // Authentication
  setAuthToken(token: string) {
    this.token = token;
    localStorage.setItem('aetherium_token', token);
  }

  getAuthToken(): string | null {
    if (!this.token) {
      this.token = localStorage.getItem('aetherium_token');
    }
    return this.token;
  }

  clearAuthToken() {
    this.token = null;
    localStorage.removeItem('aetherium_token');
  }

  // HTTP Request Helper
  private async makeRequest<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const token = this.getAuthToken();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // System Health & Metrics
  async getSystemHealth(): Promise<SystemHealth> {
    return this.makeRequest<SystemHealth>('/health');
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    return this.makeRequest<SystemMetrics>('/metrics');
  }

  // AI Chat Integration
  async sendChatMessage(message: string, toolId?: string): Promise<ChatMessage> {
    const endpoint = toolId ? `/api/tools/${toolId}/chat` : '/api/chat';
    
    return this.makeRequest<ChatMessage>(endpoint, {
      method: 'POST',
      body: JSON.stringify({ 
        message,
        timestamp: new Date().toISOString()
      }),
    });
  }

  async getChatHistory(limit: number = 50): Promise<ChatMessage[]> {
    return this.makeRequest<ChatMessage[]>(`/api/chat/history?limit=${limit}`);
  }

  // AI Tools
  async getAvailableTools(): Promise<AITool[]> {
    return this.makeRequest<AITool[]>('/api/tools');
  }

  async executeTool(toolId: string, parameters: any): Promise<any> {
    return this.makeRequest(`/api/tools/${toolId}/execute`, {
      method: 'POST',
      body: JSON.stringify(parameters),
    });
  }

  // Quantum Computing API
  async runQuantumCircuit(circuit: any): Promise<any> {
    return this.makeRequest('/api/quantum/execute', {
      method: 'POST',
      body: JSON.stringify(circuit),
    });
  }

  async getQuantumMetrics(): Promise<any> {
    return this.makeRequest('/api/quantum/metrics');
  }

  // Time Crystals API
  async getTimeCrystalStatus(): Promise<any> {
    return this.makeRequest('/api/time-crystals/status');
  }

  async synchronizeTimeCrystals(): Promise<any> {
    return this.makeRequest('/api/time-crystals/synchronize', {
      method: 'POST',
    });
  }

  // Neuromorphic Processing API
  async processNeuromorphic(data: any): Promise<any> {
    return this.makeRequest('/api/neuromorphic/process', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // AI Productivity Suite
  async getProductivityTools(): Promise<AITool[]> {
    return this.makeRequest<AITool[]>('/api/productivity/tools');
  }

  async executeProductivityTool(toolName: string, parameters: any): Promise<any> {
    return this.makeRequest(`/api/productivity/${toolName}`, {
      method: 'POST',
      body: JSON.stringify(parameters),
    });
  }

  // WebSocket Connection for Real-time Chat
  connectWebSocket(onMessage: (message: ChatMessage) => void): void {
    if (this.websocket) {
      this.websocket.close();
    }

    const wsUrl = `${this.wsUrl}/ws/chat`;
    this.websocket = new WebSocket(wsUrl);

    this.websocket.onopen = () => {
      console.log('✅ WebSocket connected to Aetherium backend');
    };

    this.websocket.onmessage = (event) => {
      try {
        const message: ChatMessage = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.websocket.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      // Auto-reconnect after 3 seconds
      setTimeout(() => this.connectWebSocket(onMessage), 3000);
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  sendWebSocketMessage(message: ChatMessage): void {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      console.error('WebSocket not connected');
    }
  }

  disconnectWebSocket(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  // File Upload/Download
  async uploadFile(file: File, category?: string): Promise<{ url: string; id: string }> {
    const formData = new FormData();
    formData.append('file', file);
    if (category) {
      formData.append('category', category);
    }

    const token = this.getAuthToken();
    const headers: HeadersInit = {};
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${this.baseUrl}/api/files/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`File upload failed: ${response.status}`);
    }

    return response.json();
  }

  async downloadFile(fileId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/files/${fileId}`, {
      headers: this.getAuthToken() ? {
        'Authorization': `Bearer ${this.getAuthToken()}`
      } : {},
    });

    if (!response.ok) {
      throw new Error(`File download failed: ${response.status}`);
    }

    return response.blob();
  }

  // External AI Models Integration
  async connectExternalAI(provider: 'openai' | 'claude' | 'gemini', apiKey: string): Promise<boolean> {
    return this.makeRequest('/api/ai/connect', {
      method: 'POST',
      body: JSON.stringify({ provider, apiKey }),
    });
  }

  async getExternalAIStatus(): Promise<{ [provider: string]: boolean }> {
    return this.makeRequest('/api/ai/status');
  }
}

// Create singleton instance
export const apiService = new APIService();

// React Hooks for API Integration
export const useAPI = () => {
  return apiService;
};

// Error Handling Utilities
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public endpoint?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// Real AI Model Integration Helpers
export const AI_MODELS = {
  QUANTUM_1: 'quantum-1',
  NEURAL_3: 'neural-3', 
  CRYSTAL_2: 'crystal-2',
  OPENAI_GPT4: 'openai-gpt4',
  CLAUDE_3: 'claude-3',
  GEMINI_PRO: 'gemini-pro'
} as const;

export type AIModel = typeof AI_MODELS[keyof typeof AI_MODELS];

export default apiService;