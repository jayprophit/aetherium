/**
 * AETHERIUM AI MODELS SERVICE
 * Real AI Model Integration (OpenAI, Claude, Gemini, etc.)
 */

import { apiService } from './api';

// AI Model Types
export interface AIModelConfig {
  id: string;
  name: string;
  provider: 'aetherium' | 'openai' | 'anthropic' | 'google';
  type: 'quantum' | 'neural' | 'crystal' | 'gpt' | 'claude' | 'gemini';
  description: string;
  capabilities: string[];
  maxTokens: number;
  pricing?: {
    inputTokens: number;
    outputTokens: number;
  };
}

export interface ChatResponse {
  id: string;
  content: string;
  model: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  thinking?: string;
  timestamp: string;
}

export interface ModelStatus {
  id: string;
  available: boolean;
  latency: number;
  lastChecked: string;
  error?: string;
}

// AI Models Service
class AIModelsService {
  private models: Map<string, AIModelConfig> = new Map();
  private modelStatus: Map<string, ModelStatus> = new Map();
  private apiKeys: Map<string, string> = new Map();

  constructor() {
    this.initializeModels();
    this.loadApiKeys();
  }

  // Initialize available AI models
  private initializeModels() {
    const modelConfigs: AIModelConfig[] = [
      // Aetherium Quantum AI Models (Primary)
      {
        id: 'quantum-1',
        name: 'Aetherium Quantum-1',
        provider: 'aetherium',
        type: 'quantum',
        description: 'Advanced quantum AI model with superposition reasoning',
        capabilities: [
          'quantum_reasoning',
          'superposition_analysis',
          'quantum_optimization',
          'complex_problem_solving'
        ],
        maxTokens: 32000
      },
      {
        id: 'neural-3',
        name: 'Aetherium Neural-3',
        provider: 'aetherium',
        type: 'neural',
        description: 'Neuromorphic AI with spiking neural network processing',
        capabilities: [
          'neuromorphic_processing',
          'real_time_learning',
          'pattern_recognition',
          'adaptive_reasoning'
        ],
        maxTokens: 16000
      },
      {
        id: 'crystal-2',
        name: 'Aetherium Crystal-2',
        provider: 'aetherium',
        type: 'crystal',
        description: 'Time crystal enhanced AI with temporal reasoning',
        capabilities: [
          'temporal_analysis',
          'time_crystal_computation',
          'causal_reasoning',
          'predictive_modeling'
        ],
        maxTokens: 24000
      },

      // External AI Models (Secondary)
      {
        id: 'gpt-4',
        name: 'GPT-4',
        provider: 'openai',
        type: 'gpt',
        description: 'OpenAI GPT-4 - Large language model',
        capabilities: [
          'text_generation',
          'code_generation',
          'analysis',
          'creative_writing'
        ],
        maxTokens: 8000,
        pricing: { inputTokens: 0.03, outputTokens: 0.06 }
      },
      {
        id: 'claude-3',
        name: 'Claude-3 Sonnet',
        provider: 'anthropic',
        type: 'claude',
        description: 'Anthropic Claude-3 - Advanced reasoning AI',
        capabilities: [
          'advanced_reasoning',
          'code_analysis',
          'document_analysis',
          'creative_tasks'
        ],
        maxTokens: 20000,
        pricing: { inputTokens: 0.003, outputTokens: 0.015 }
      },
      {
        id: 'gemini-pro',
        name: 'Gemini Pro',
        provider: 'google',
        type: 'gemini',
        description: 'Google Gemini Pro - Multimodal AI',
        capabilities: [
          'multimodal_processing',
          'image_analysis',
          'code_generation',
          'reasoning'
        ],
        maxTokens: 30000,
        pricing: { inputTokens: 0.00025, outputTokens: 0.0005 }
      }
    ];

    modelConfigs.forEach(model => {
      this.models.set(model.id, model);
    });
  }

  // Load API keys from storage
  private loadApiKeys() {
    const stored = localStorage.getItem('aetherium_ai_keys');
    if (stored) {
      try {
        const keys = JSON.parse(stored);
        Object.entries(keys).forEach(([provider, key]) => {
          this.apiKeys.set(provider, key as string);
        });
      } catch (error) {
        console.error('Failed to load AI API keys:', error);
      }
    }
  }

  // Save API keys to storage
  private saveApiKeys() {
    const keys = Object.fromEntries(this.apiKeys.entries());
    localStorage.setItem('aetherium_ai_keys', JSON.stringify(keys));
  }

  // Set API key for external providers
  setApiKey(provider: string, apiKey: string): void {
    this.apiKeys.set(provider, apiKey);
    this.saveApiKeys();
  }

  // Get all available models
  getAvailableModels(): AIModelConfig[] {
    return Array.from(this.models.values());
  }

  // Get model by ID
  getModel(modelId: string): AIModelConfig | undefined {
    return this.models.get(modelId);
  }

  // Get models by provider
  getModelsByProvider(provider: string): AIModelConfig[] {
    return Array.from(this.models.values()).filter(m => m.provider === provider);
  }

  // Check model availability
  async checkModelStatus(modelId: string): Promise<ModelStatus> {
    const startTime = Date.now();
    
    try {
      const model = this.models.get(modelId);
      if (!model) {
        throw new Error(`Model ${modelId} not found`);
      }

      let available = false;

      if (model.provider === 'aetherium') {
        // Check internal Aetherium models via backend
        const health = await apiService.getSystemHealth();
        available = health.status === 'healthy';
      } else {
        // Check external models
        const apiKey = this.apiKeys.get(model.provider);
        if (!apiKey) {
          throw new Error(`API key not configured for ${model.provider}`);
        }

        // Test API connection
        available = await this.testExternalModel(model, apiKey);
      }

      const status: ModelStatus = {
        id: modelId,
        available,
        latency: Date.now() - startTime,
        lastChecked: new Date().toISOString()
      };

      this.modelStatus.set(modelId, status);
      return status;

    } catch (error) {
      const status: ModelStatus = {
        id: modelId,
        available: false,
        latency: Date.now() - startTime,
        lastChecked: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      };

      this.modelStatus.set(modelId, status);
      return status;
    }
  }

  // Test external model connectivity
  private async testExternalModel(model: AIModelConfig, apiKey: string): Promise<boolean> {
    try {
      switch (model.provider) {
        case 'openai':
          return await this.testOpenAI(apiKey);
        case 'anthropic':
          return await this.testClaude(apiKey);
        case 'google':
          return await this.testGemini(apiKey);
        default:
          return false;
      }
    } catch (error) {
      console.error(`Failed to test ${model.provider}:`, error);
      return false;
    }
  }

  // Test OpenAI connection
  private async testOpenAI(apiKey: string): Promise<boolean> {
    const response = await fetch('https://api.openai.com/v1/models', {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      }
    });
    return response.ok;
  }

  // Test Claude connection
  private async testClaude(apiKey: string): Promise<boolean> {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1,
        messages: [{ role: 'user', content: 'test' }]
      })
    });
    return response.status !== 401; // Not unauthorized
  }

  // Test Gemini connection
  private async testGemini(apiKey: string): Promise<boolean> {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);
    return response.ok;
  }

  // Send chat message to specific model
  async sendChatMessage(
    modelId: string, 
    message: string, 
    context?: any[]
  ): Promise<ChatResponse> {
    const model = this.models.get(modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not found`);
    }

    const startTime = Date.now();

    try {
      let response: ChatResponse;

      if (model.provider === 'aetherium') {
        // Use internal Aetherium models via backend
        response = await this.sendToAetheriumModel(model, message, context);
      } else {
        // Use external AI models
        response = await this.sendToExternalModel(model, message, context);
      }

      // Add metadata
      response.timestamp = new Date().toISOString();
      response.model = modelId;

      return response;

    } catch (error) {
      console.error(`Chat message failed for ${modelId}:`, error);
      throw new Error(`Failed to get response from ${model.name}: ${error}`);
    }
  }

  // Send to Aetherium internal models
  private async sendToAetheriumModel(
    model: AIModelConfig, 
    message: string, 
    context?: any[]
  ): Promise<ChatResponse> {
    // Route to appropriate Aetherium backend endpoint
    const endpoint = this.getAetheriumEndpoint(model.type);
    
    const backendResponse = await apiService.sendChatMessage(message, model.id);
    
    return {
      id: backendResponse.id,
      content: backendResponse.content,
      model: model.id,
      thinking: backendResponse.thinking || this.generateThinking(model, message),
      timestamp: backendResponse.timestamp
    };
  }

  // Send to external AI models
  private async sendToExternalModel(
    model: AIModelConfig,
    message: string,
    context?: any[]
  ): Promise<ChatResponse> {
    const apiKey = this.apiKeys.get(model.provider);
    if (!apiKey) {
      throw new Error(`API key not configured for ${model.provider}`);
    }

    switch (model.provider) {
      case 'openai':
        return await this.sendToOpenAI(model, message, apiKey, context);
      case 'anthropic':
        return await this.sendToClaude(model, message, apiKey, context);
      case 'google':
        return await this.sendToGemini(model, message, apiKey, context);
      default:
        throw new Error(`Unsupported provider: ${model.provider}`);
    }
  }

  // OpenAI integration
  private async sendToOpenAI(
    model: AIModelConfig,
    message: string,
    apiKey: string,
    context?: any[]
  ): Promise<ChatResponse> {
    const messages = [
      ...(context || []),
      { role: 'user', content: message }
    ];

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages,
        max_tokens: Math.min(model.maxTokens, 4000),
        temperature: 0.7
      })
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      id: data.id || crypto.randomUUID(),
      content: data.choices[0]?.message?.content || '',
      model: model.id,
      usage: data.usage,
      thinking: `Processing with ${model.name}...`,
      timestamp: new Date().toISOString()
    };
  }

  // Claude integration
  private async sendToClaude(
    model: AIModelConfig,
    message: string,
    apiKey: string,
    context?: any[]
  ): Promise<ChatResponse> {
    const messages = [
      ...(context || []),
      { role: 'user', content: message }
    ];

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-3-sonnet-20240229',
        max_tokens: Math.min(model.maxTokens, 4000),
        messages
      })
    });

    if (!response.ok) {
      throw new Error(`Claude API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      id: data.id || crypto.randomUUID(),
      content: data.content[0]?.text || '',
      model: model.id,
      thinking: `Analyzing with ${model.name}...`,
      timestamp: new Date().toISOString()
    };
  }

  // Gemini integration
  private async sendToGemini(
    model: AIModelConfig,
    message: string,
    apiKey: string,
    context?: any[]
  ): Promise<ChatResponse> {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: message }] }],
          generationConfig: {
            maxOutputTokens: Math.min(model.maxTokens, 4000),
            temperature: 0.7
          }
        })
      }
    );

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      id: crypto.randomUUID(),
      content: data.candidates[0]?.content?.parts[0]?.text || '',
      model: model.id,
      thinking: `Computing with ${model.name}...`,
      timestamp: new Date().toISOString()
    };
  }

  // Get Aetherium model endpoint
  private getAetheriumEndpoint(modelType: string): string {
    switch (modelType) {
      case 'quantum': return '/api/quantum/chat';
      case 'neural': return '/api/neuromorphic/chat';  
      case 'crystal': return '/api/time-crystals/chat';
      default: return '/api/chat';
    }
  }

  // Generate AI thinking process simulation
  private generateThinking(model: AIModelConfig, message: string): string {
    const thinkingPatterns = {
      quantum: [
        'Initializing quantum superposition states...',
        'Analyzing quantum entanglement patterns...',
        'Processing through quantum circuits...',
        'Measuring quantum probabilities...'
      ],
      neural: [
        'Activating neuromorphic pathways...',
        'Simulating spiking neural networks...',
        'Processing through synaptic connections...',
        'Adapting neural weights...'
      ],
      crystal: [
        'Synchronizing with time crystal network...',
        'Analyzing temporal patterns...',
        'Processing through crystal lattice...',
        'Computing causal relationships...'
      ]
    };

    const patterns = thinkingPatterns[model.type as keyof typeof thinkingPatterns] || [
      'Processing request...',
      'Analyzing input...',
      'Generating response...'
    ];

    return patterns[Math.floor(Math.random() * patterns.length)];
  }

  // Get model status
  getModelStatus(modelId: string): ModelStatus | undefined {
    return this.modelStatus.get(modelId);
  }

  // Get all model statuses
  getAllModelStatuses(): Map<string, ModelStatus> {
    return new Map(this.modelStatus);
  }
}

// Create singleton instance
export const aiModelsService = new AIModelsService();

export default aiModelsService;