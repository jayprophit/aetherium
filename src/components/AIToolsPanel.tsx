/**
 * AETHERIUM AI TOOLS PANEL COMPONENT
 * Interactive AI tools with real execution functionality
 */

import React, { useState } from 'react';
import { 
  Search, BarChart3, FileText, Image, Calculator, Mail, 
  Globe, Video, Code, Brain, Zap, Users, PieChart,
  TrendingUp, MapPin, ShoppingCart, DollarSign, Languages,
  Mic, Camera, Smartphone, Edit3, Lightbulb, Target,
  Wrench, Briefcase, Award, Rocket, Cpu, Shield
} from 'lucide-react';
import { useAITools } from '../hooks/useAetherium';

interface AIToolsPanelProps {
  darkMode: boolean;
  onToolExecute: (toolId: string, toolName: string) => void;
}

export const AIToolsPanel: React.FC<AIToolsPanelProps> = ({
  darkMode,
  onToolExecute
}) => {
  const { availableTools, isExecuting, executeTool } = useAITools();
  const [selectedCategory, setSelectedCategory] = useState('featured');

  // Comprehensive AI tools configuration
  const toolsConfig = [
    // Research & Analysis
    { id: 'wide_research', name: 'Wide Research', icon: Search, category: 'research', color: 'blue' },
    { id: 'data_visualizations', name: 'Data Visualizations', icon: BarChart3, category: 'research', color: 'green' },
    { id: 'market_research', name: 'Market Research', icon: TrendingUp, category: 'research', color: 'purple' },
    { id: 'fact_checker', name: 'Fact Checker', icon: Shield, category: 'research', color: 'red' },
    
    // Productivity & Automation
    { id: 'everything_calculator', name: 'Everything Calculator', icon: Calculator, category: 'productivity', color: 'blue' },
    { id: 'email_generator', name: 'Email Generator', icon: Mail, category: 'productivity', color: 'green' },
    { id: 'trip_planner', name: 'Trip Planner', icon: MapPin, category: 'productivity', color: 'orange' },
    { id: 'essay_outline', name: 'Essay Outline', icon: FileText, category: 'productivity', color: 'purple' },
    
    // Creative & Design
    { id: 'sketch_to_photo', name: 'Sketch to Photo', icon: Image, category: 'creative', color: 'pink' },
    { id: 'video_generator', name: 'Video Generator', icon: Video, category: 'creative', color: 'red' },
    { id: 'voice_generator', name: 'Voice Generator', icon: Mic, category: 'creative', color: 'blue' },
    { id: 'meme_maker', name: 'Meme Maker', icon: Camera, category: 'creative', color: 'yellow' },
    
    // Business & Finance
    { id: 'business_canvas', name: 'Business Canvas', icon: Briefcase, category: 'business', color: 'blue' },
    { id: 'swot_analysis', name: 'SWOT Analysis', icon: Target, category: 'business', color: 'green' },
    { id: 'expense_tracker', name: 'Expense Tracker', icon: DollarSign, category: 'business', color: 'red' },
    { id: 'pc_builder', name: 'PC Builder', icon: Cpu, category: 'business', color: 'purple' },
    
    // Development & Technical
    { id: 'website_builder', name: 'Website Builder', icon: Code, category: 'development', color: 'blue' },
    { id: 'github_deploy', name: 'GitHub Deploy Tool', icon: Globe, category: 'development', color: 'gray' },
    { id: 'extension_builder', name: 'Extension Builder', icon: Wrench, category: 'development', color: 'orange' },
    { id: 'poc_starter', name: 'POC Starter', icon: Rocket, category: 'development', color: 'purple' },
    
    // Communication & Media
    { id: 'translator', name: 'Universal Translator', icon: Languages, category: 'communication', color: 'blue' },
    { id: 'youtube_analysis', name: 'YouTube Viral Analysis', icon: Video, category: 'communication', color: 'red' },
    { id: 'influencer_finder', name: 'Influencer Finder', icon: Users, category: 'communication', color: 'pink' },
    { id: 'sentiment_analyzer', name: 'Sentiment Analyzer', icon: Brain, category: 'communication', color: 'green' },
    
    // AI & Advanced
    { id: 'ai_coach', name: 'AI Coach', icon: Lightbulb, category: 'ai', color: 'gold' },
    { id: 'quantum_simulator', name: 'Quantum Simulator', icon: Zap, category: 'ai', color: 'purple' },
    { id: 'neural_network', name: 'Neural Network', icon: Brain, category: 'ai', color: 'blue' },
    { id: 'time_crystal', name: 'Time Crystal AI', icon: Award, category: 'ai', color: 'cyan' }
  ];

  const categories = [
    { id: 'featured', name: 'Featured', icon: Zap },
    { id: 'research', name: 'Research', icon: Search },
    { id: 'productivity', name: 'Productivity', icon: Target },
    { id: 'creative', name: 'Creative', icon: Image },
    { id: 'business', name: 'Business', icon: Briefcase },
    { id: 'development', name: 'Development', icon: Code },
    { id: 'communication', name: 'Communication', icon: Users },
    { id: 'ai', name: 'AI Advanced', icon: Brain }
  ];

  // Get tools for selected category
  const getToolsForCategory = () => {
    if (selectedCategory === 'featured') {
      // Featured tools - mix from different categories
      return toolsConfig.filter(tool => 
        ['wide_research', 'everything_calculator', 'sketch_to_photo', 'business_canvas', 
         'website_builder', 'translator', 'ai_coach', 'quantum_simulator'].includes(tool.id)
      );
    }
    return toolsConfig.filter(tool => tool.category === selectedCategory);
  };

  const handleToolClick = async (tool: any) => {
    try {
      // Call the parent handler for UI updates
      onToolExecute(tool.id, tool.name);
      
      // Execute the actual tool
      await executeTool(tool.id, {
        context: 'User initiated tool execution',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error(`Failed to execute tool ${tool.name}:`, error);
    }
  };

  const getColorClasses = (color: string) => {
    const colorMap = {
      blue: 'from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700',
      green: 'from-green-500 to-green-600 hover:from-green-600 hover:to-green-700',
      purple: 'from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700',
      red: 'from-red-500 to-red-600 hover:from-red-600 hover:to-red-700',
      orange: 'from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700',
      pink: 'from-pink-500 to-pink-600 hover:from-pink-600 hover:to-pink-700',
      yellow: 'from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700',
      gray: 'from-gray-500 to-gray-600 hover:from-gray-600 hover:to-gray-700',
      gold: 'from-yellow-400 to-orange-500 hover:from-yellow-500 hover:to-orange-600',
      cyan: 'from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600'
    };
    return colorMap[color as keyof typeof colorMap] || colorMap.blue;
  };

  return (
    <div className="h-full flex flex-col">
      {/* Category Tabs */}
      <div className={`border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} p-4`}>
        <div className="flex flex-wrap gap-2">
          {categories.map(category => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedCategory === category.id
                  ? `${darkMode ? 'bg-purple-600 text-white' : 'bg-purple-500 text-white'}`
                  : `${darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`
              }`}
            >
              <category.icon className="w-4 h-4" />
              <span>{category.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Tools Grid */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {getToolsForCategory().map(tool => (
            <button
              key={tool.id}
              onClick={() => handleToolClick(tool)}
              disabled={isExecuting}
              className={`group relative bg-gradient-to-r ${getColorClasses(tool.color)} text-white p-4 rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              <div className="flex flex-col items-center text-center space-y-2">
                <div className="p-3 bg-white bg-opacity-20 rounded-lg group-hover:bg-opacity-30 transition-colors">
                  <tool.icon className="w-6 h-6" />
                </div>
                <div className="font-medium text-sm leading-tight">{tool.name}</div>
              </div>
              
              {/* Loading overlay */}
              {isExecuting && (
                <div className="absolute inset-0 bg-black bg-opacity-50 rounded-xl flex items-center justify-center">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                </div>
              )}
            </button>
          ))}
        </div>

        {/* Empty state */}
        {getToolsForCategory().length === 0 && (
          <div className="text-center py-12">
            <div className="text-4xl mb-4">🔧</div>
            <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
              No tools in this category
            </h3>
            <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
              Try selecting a different category
            </p>
          </div>
        )}
      </div>

      {/* Tool Count & Status */}
      <div className={`border-t ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'} p-3`}>
        <div className="flex items-center justify-between text-sm">
          <span className={`${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            {getToolsForCategory().length} tools available
          </span>
          {isExecuting && (
            <span className="text-purple-500 font-medium">
              🔄 Executing tool...
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIToolsPanel;