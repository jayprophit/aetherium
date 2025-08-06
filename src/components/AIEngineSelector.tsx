/**
 * AI ENGINE SELECTOR COMPONENT
 * Allows users to select between internal Aetherium AI and external providers
 */

import React, { useState, useEffect } from 'react';
import { Brain, Zap, Globe, Settings, CheckCircle, AlertCircle, Clock } from 'lucide-react';

interface AIEngine {
  id: string;
  name: string;
  description: string;
  status: 'online' | 'available' | 'needs_api_key' | 'coming_soon';
  primary: boolean;
  models: AIModel[];
}

interface AIModel {
  id: string;
  name: string;
  specialization: string;
}

interface Props {
  selectedEngine: string;
  selectedModel: string;
  onEngineChange: (engine: string, model: string) => void;
  darkMode?: boolean;
}

const AIEngineSelector: React.FC<Props> = ({ 
  selectedEngine, 
  selectedModel, 
  onEngineChange,
  darkMode = false 
}) => {
  const [engines, setEngines] = useState<AIEngine[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    fetchEngines();
  }, []);

  const fetchEngines = async () => {
    try {
      const response = await fetch('/api/ai-engines');
      const data = await response.json();
      setEngines(data.engines || []);
    } catch (error) {
      console.error('Failed to fetch AI engines:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getEngineIcon = (engineId: string) => {
    switch (engineId) {
      case 'aetherium-internal':
        return <Brain className="w-5 h-5" />;
      case 'openai':
        return <Zap className="w-5 h-5" />;
      default:
        return <Globe className="w-5 h-5" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'text-green-500';
      case 'available':
        return 'text-blue-500';
      case 'needs_api_key':
        return 'text-yellow-500';
      case 'coming_soon':
        return 'text-gray-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'available':
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      case 'needs_api_key':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'coming_soon':
        return <Clock className="w-4 h-4 text-gray-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const handleEngineSelect = (engine: AIEngine) => {
    if (engine.status === 'online' || engine.status === 'available') {
      const defaultModel = engine.models[0]?.id || engine.id;
      onEngineChange(engine.id, defaultModel);
    }
  };

  const handleModelSelect = (engineId: string, modelId: string) => {
    onEngineChange(engineId, modelId);
  };

  const selectedEngineData = engines.find(e => e.id === selectedEngine);

  if (isLoading) {
    return (
      <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 rounded mb-2"></div>
          <div className="h-4 bg-gray-300 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
      {/* Quick Engine Selector */}
      <div className="flex items-center space-x-2 mb-4">
        <span className="text-sm font-medium">AI Engine:</span>
        <div className="flex space-x-2">
          {engines.slice(0, 3).map((engine) => (
            <button
              key={engine.id}
              onClick={() => handleEngineSelect(engine)}
              disabled={engine.status === 'coming_soon' || engine.status === 'needs_api_key'}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm transition-all ${
                selectedEngine === engine.id
                  ? darkMode
                    ? 'bg-purple-600 text-white'
                    : 'bg-purple-100 text-purple-800'
                  : engine.status === 'online' || engine.status === 'available'
                    ? darkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                    : 'bg-gray-500/20 text-gray-500 cursor-not-allowed'
              }`}
              title={engine.description}
            >
              {getEngineIcon(engine.id)}
              <span>{engine.name.split(' ')[0]}</span>
              {selectedEngine === engine.id && getStatusIcon(engine.status)}
            </button>
          ))}
        </div>
        
        <button
          onClick={() => setShowDetails(!showDetails)}
          className={`p-2 rounded-lg transition-colors ${
            darkMode 
              ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
              : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
          }`}
          title="Engine Details"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>

      {/* Model Selector for Selected Engine */}
      {selectedEngineData && selectedEngineData.models.length > 1 && (
        <div className="space-y-2">
          <label className="text-sm font-medium">Model:</label>
          <div className="grid grid-cols-1 gap-2">
            {selectedEngineData.models.map((model) => (
              <button
                key={model.id}
                onClick={() => handleModelSelect(selectedEngine, model.id)}
                className={`text-left p-3 rounded-lg transition-all ${
                  selectedModel === model.id
                    ? darkMode
                      ? 'bg-purple-600/20 border border-purple-500 text-purple-300'
                      : 'bg-purple-50 border border-purple-200 text-purple-800'
                    : darkMode
                      ? 'bg-gray-700 hover:bg-gray-600 border border-gray-600'
                      : 'bg-gray-50 hover:bg-gray-100 border border-gray-200'
                }`}
              >
                <div className="font-medium text-sm">{model.name}</div>
                <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  {model.specialization}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Engine List */}
      {showDetails && (
        <div className={`p-4 rounded-lg border ${
          darkMode 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-gray-50 border-gray-200'
        }`}>
          <h4 className="font-semibold mb-3">Available AI Engines</h4>
          <div className="space-y-3">
            {engines.map((engine) => (
              <div
                key={engine.id}
                className={`p-3 rounded-lg border ${
                  selectedEngine === engine.id
                    ? darkMode
                      ? 'border-purple-500 bg-purple-900/20'
                      : 'border-purple-200 bg-purple-50'
                    : darkMode
                      ? 'border-gray-600 bg-gray-700'
                      : 'border-gray-200 bg-white'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    {getEngineIcon(engine.id)}
                    <div>
                      <div className="font-medium flex items-center space-x-2">
                        <span>{engine.name}</span>
                        {engine.primary && (
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            darkMode ? 'bg-purple-900 text-purple-300' : 'bg-purple-100 text-purple-800'
                          }`}>
                            Primary
                          </span>
                        )}
                      </div>
                      <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        {engine.description}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-1">
                    {getStatusIcon(engine.status)}
                    <span className={`text-xs ${getStatusColor(engine.status)}`}>
                      {engine.status.replace('_', ' ')}
                    </span>
                  </div>
                </div>

                {/* Engine Models */}
                {engine.models.length > 0 && (
                  <div className="mt-2">
                    <div className="text-xs font-medium mb-1">Models:</div>
                    <div className="flex flex-wrap gap-1">
                      {engine.models.map((model) => (
                        <span
                          key={model.id}
                          className={`text-xs px-2 py-1 rounded-full ${
                            darkMode 
                              ? 'bg-gray-600 text-gray-300' 
                              : 'bg-gray-200 text-gray-700'
                          }`}
                          title={model.specialization}
                        >
                          {model.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Action Button */}
                {engine.status === 'online' || engine.status === 'available' ? (
                  <button
                    onClick={() => handleEngineSelect(engine)}
                    disabled={selectedEngine === engine.id}
                    className={`mt-2 w-full py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                      selectedEngine === engine.id
                        ? darkMode
                          ? 'bg-purple-600 text-white cursor-default'
                          : 'bg-purple-100 text-purple-800 cursor-default'
                        : darkMode
                          ? 'bg-purple-600 hover:bg-purple-700 text-white'
                          : 'bg-purple-100 hover:bg-purple-200 text-purple-800'
                    }`}
                  >
                    {selectedEngine === engine.id ? 'Selected' : 'Select Engine'}
                  </button>
                ) : (
                  <div className={`mt-2 w-full py-2 px-3 rounded-lg text-sm text-center ${
                    darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-100 text-gray-500'
                  }`}>
                    {engine.status === 'needs_api_key' ? 'API Key Required' : 'Coming Soon'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Current Selection Summary */}
      <div className={`p-3 rounded-lg border ${
        darkMode 
          ? 'bg-gray-800 border-gray-700' 
          : 'bg-blue-50 border-blue-200'
      }`}>
        <div className="text-sm">
          <span className="font-medium">Active: </span>
          <span className={darkMode ? 'text-blue-300' : 'text-blue-700'}>
            {selectedEngineData?.name} - {selectedEngineData?.models.find(m => m.id === selectedModel)?.name}
          </span>
        </div>
      </div>
    </div>
  );
};

export default AIEngineSelector;