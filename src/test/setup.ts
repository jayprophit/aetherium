/**
 * AETHERIUM TEST SETUP
 * Global test configuration and utilities
 */

import { expect, afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';

// Extend expect with React Testing Library matchers
expect.extend(matchers);

// Clean up after each test
afterEach(() => {
  cleanup();
});

// Mock environment variables
vi.mock('import.meta.env', () => ({
  REACT_APP_API_URL: 'http://localhost:8000',
  REACT_APP_WS_URL: 'ws://localhost:8000',
  DEV: true,
  VITEST: true
}));

// Mock WebSocket
global.WebSocket = vi.fn(() => ({
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: WebSocket.OPEN,
  CONNECTING: WebSocket.CONNECTING,
  OPEN: WebSocket.OPEN,
  CLOSING: WebSocket.CLOSING,
  CLOSED: WebSocket.CLOSED,
})) as any;

// Mock fetch for API calls
global.fetch = vi.fn();

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
global.localStorage = localStorageMock as any;

// Mock sessionStorage
const sessionStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
global.sessionStorage = sessionStorageMock as any;

// Mock window methods
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Console helpers for tests
export const mockConsole = {
  log: vi.spyOn(console, 'log').mockImplementation(() => {}),
  error: vi.spyOn(console, 'error').mockImplementation(() => {}),
  warn: vi.spyOn(console, 'warn').mockImplementation(() => {}),
};

// Test utilities
export const createMockResponse = (data: any, status = 200) => {
  return Promise.resolve({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  } as Response);
};

export const mockApiService = {
  getSystemHealth: vi.fn(),
  getSystemMetrics: vi.fn(),
  sendChatMessage: vi.fn(),
  getChatHistory: vi.fn(),
  getAvailableTools: vi.fn(),
  executeTool: vi.fn(),
};

export const mockWebSocketService = {
  connect: vi.fn(),
  disconnect: vi.fn(),
  sendChatMessage: vi.fn(),
  createNewSession: vi.fn(),
  getAllSessions: vi.fn(),
  getCurrentSession: vi.fn(),
  isConnected: vi.fn(),
  getConnectionState: vi.fn(),
};

export const mockAiModelsService = {
  getAvailableModels: vi.fn(),
  getModel: vi.fn(),
  sendChatMessage: vi.fn(),
  checkModelStatus: vi.fn(),
  setApiKey: vi.fn(),
};

// Export for use in tests
export { vi, cleanup };