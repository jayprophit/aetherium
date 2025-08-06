/**
 * AETHERIUM API SERVICE TESTS
 * Test suite for API service functionality
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { apiService } from './api';
import { createMockResponse } from '../test/setup';

describe('API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn();
  });

  describe('getSystemHealth', () => {
    it('should return system health data', async () => {
      const mockHealth = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        components: {
          database: 'healthy',
          quantum: 'healthy',
          ai: 'healthy'
        }
      };

      (global.fetch as any).mockResolvedValue(createMockResponse(mockHealth));

      const result = await apiService.getSystemHealth();
      
      expect(result).toEqual(mockHealth);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/health'),
        expect.any(Object)
      );
    });

    it('should handle API errors gracefully', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network error'));

      await expect(apiService.getSystemHealth()).rejects.toThrow('Network error');
    });
  });

  describe('sendChatMessage', () => {
    it('should send chat message successfully', async () => {
      const mockResponse = {
        id: 'msg-123',
        content: 'Hello! How can I help you?',
        role: 'assistant',
        timestamp: new Date().toISOString()
      };

      (global.fetch as any).mockResolvedValue(createMockResponse(mockResponse));

      const result = await apiService.sendChatMessage({
        message: 'Hello',
        sessionId: 'session-123',
        model: 'quantum-1'
      });

      expect(result).toEqual(mockResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/chat/send'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          }),
          body: expect.stringContaining('"message":"Hello"')
        })
      );
    });
  });

  describe('getAvailableTools', () => {
    it('should return list of available tools', async () => {
      const mockTools = [
        {
          id: 'summarizer',
          name: 'AI Summarizer',
          description: 'Summarize text content',
          category: 'analysis',
          icon: 'FileText'
        },
        {
          id: 'translator',
          name: 'Universal Translator',
          description: 'Translate between languages',
          category: 'communication',
          icon: 'Languages'
        }
      ];

      (global.fetch as any).mockResolvedValue(createMockResponse(mockTools));

      const result = await apiService.getAvailableTools();
      
      expect(result).toEqual(mockTools);
      expect(result).toHaveLength(2);
      expect(result[0]).toHaveProperty('id');
      expect(result[0]).toHaveProperty('name');
      expect(result[0]).toHaveProperty('category');
    });
  });

  describe('executeTool', () => {
    it('should execute tool with parameters', async () => {
      const mockResult = {
        success: true,
        result: 'Tool executed successfully',
        executionTime: 150
      };

      (global.fetch as any).mockResolvedValue(createMockResponse(mockResult));

      const result = await apiService.executeTool('summarizer', {
        text: 'Long text to summarize...',
        maxLength: 100
      });

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/tools/summarizer/execute'),
        expect.objectContaining({
          method: 'POST'
        })
      );
    });

    it('should handle tool execution errors', async () => {
      (global.fetch as any).mockResolvedValue(createMockResponse(
        { error: 'Tool not found' },
        404
      ));

      await expect(
        apiService.executeTool('nonexistent', {})
      ).rejects.toThrow('HTTP error! status: 404');
    });
  });
});