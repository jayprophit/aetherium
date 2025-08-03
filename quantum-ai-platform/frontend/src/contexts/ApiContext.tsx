import React, { createContext, useContext, useState, useEffect } from 'react';
import axios, { AxiosInstance } from 'axios';

interface ApiContextType {
  api: AxiosInstance;
  isConnected: boolean;
  baseUrl: string;
  setBaseUrl: (url: string) => void;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export const useApi = () => {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

interface ApiProviderProps {
  children: React.ReactNode;
}

export const ApiProvider: React.FC<ApiProviderProps> = ({ children }) => {
  const [baseUrl, setBaseUrl] = useState('http://localhost:8000');
  const [isConnected, setIsConnected] = useState(false);
  const [api, setApi] = useState<AxiosInstance>(() => 
    axios.create({
      baseURL: baseUrl,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  );

  useEffect(() => {
    // Create new axios instance when baseUrl changes
    const newApi = axios.create({
      baseURL: baseUrl,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for authentication
    newApi.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Add response interceptor for error handling
    newApi.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          // Could trigger logout here
        }
        return Promise.reject(error);
      }
    );

    setApi(newApi);

    // Test connection
    const testConnection = async () => {
      try {
        await newApi.get('/health');
        setIsConnected(true);
      } catch (error) {
        setIsConnected(false);
        console.warn('API connection failed:', error);
      }
    };

    testConnection();
    
    // Set up periodic health checks
    const healthCheckInterval = setInterval(testConnection, 30000); // Every 30 seconds

    return () => clearInterval(healthCheckInterval);
  }, [baseUrl]);

  const value = {
    api,
    isConnected,
    baseUrl,
    setBaseUrl,
  };

  return (
    <ApiContext.Provider value={value}>
      {children}
    </ApiContext.Provider>
  );
};