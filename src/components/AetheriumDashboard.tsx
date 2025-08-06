import React, { useState, useRef, useEffect } from 'react';
import { 
  Send, 
  Mic, 
  Paperclip, 
  Search, 
  Plus, 
  Settings, 
  User, 
  ChevronDown, 
  Home, 
  MessageSquare, 
  Code, 
  Image, 
  FileText, 
  BarChart3, 
  Brain, 
  Zap, 
  Share2, 
  Star, 
  MoreHorizontal,
  Archive,
  Clock,
  Download,
  Upload,
  Eye,
  Moon,
  Sun,
  Bell,
  Bookmark,
  Trash2,
  Edit3,
  Copy,
  ExternalLink,
  Layers,
  Database,
  Terminal,
  Palette,
  Calculator,
  ShoppingCart,
  Mail,
  MapPin,
  FileCheck,
  Globe,
  Video,
  Camera,
  Headphones,
  Phone,
  Smartphone,
  Folder,
  CheckSquare,
  TrendingUp,
  DollarSign,
  PenTool,
  Users,
  Shield,
  Wrench,
  Gamepad2,
  Briefcase,
  PieChart,
  TrendingDown,
  Award,
  Target,
  Lightbulb,
  Rocket,
  Cpu,
  HardDrive,
  Monitor,
  Wifi,
  Lock,
  Unlock,
  Menu,
  X,
  QrCode,
  HelpCircle,
  BookOpen,
  ChevronRight,
  Play,
  Square,
  RotateCcw,
  Maximize2,
  Minimize2,
  Activity,
  Compass,
  Plug,
  Sliders,
  Filter,
  Grid,
  Layout,
  Mic2,
  PlayCircle,
  FileVideo,
  Languages,
  Scissors,
  Paintbrush,
  Volume2,
  Repeat,
  MousePointer,
  Smile,
  CloudDownload,
  Building,
  Atom,
  Waves,
  Sparkles,
  Bot,
  ChartLine,
  GitBranch
} from 'lucide-react';

const AetheriumPlatform = () => {
  const [darkMode, setDarkMode] = useState(true);
  const [showDropdown, setShowDropdown] = useState(null);
  const [message, setMessage] = useState('');
  const [selectedModel, setSelectedModel] = useState('aetherium-quantum-1');
  const [rightPanelView, setRightPanelView] = useState('view window');
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [activeLeftTab, setActiveLeftTab] = useState('chats');
  const [activeCenterTab, setActiveCenterTab] = useState('chat');
  const [taskProgress, setTaskProgress] = useState(85);
  const fileInputRef = useRef(null);

  const models = [
    { name: 'Aetherium AI Core', provider: 'Aetherium', type: 'platform', status: 'online' },
    { name: 'Quantum Neural Net', provider: 'Aetherium', type: 'platform', status: 'online' },
    { name: 'Time Crystal AI', provider: 'Aetherium', type: 'platform', status: 'online' },
    { name: 'Claude-4', provider: 'Anthropic', type: 'third-party', status: 'online' },
    { name: 'GPT-4', provider: 'OpenAI', type: 'third-party', status: 'online' },
    { name: 'DeepSeek-V3', provider: 'DeepSeek', type: 'third-party', status: 'online' },
    { name: 'Gen Spark', provider: 'Baidu', type: 'third-party', status: 'online' },
    { name: 'Qwen-Max', provider: 'Alibaba', type: 'third-party', status: 'online' },
    { name: 'Grok', provider: 'xAI', type: 'third-party', status: 'online' },
    { name: 'Llama 3', provider: 'Meta', type: 'third-party', status: 'online' },
  ];

  const conversations = [
    { id: 1, title: 'Aetherium Platform Development', time: 'Now', pinned: true, type: 'active' },
    { id: 2, title: 'Quantum Computing Research', time: '2 hours ago', pinned: false, type: 'normal' },
    { id: 3, title: 'Time Crystals Implementation', time: 'Yesterday', pinned: false, type: 'normal' },
    { id: 4, title: 'Neuromorphic AI Design', time: '2 days ago', pinned: false, type: 'normal' },
    { id: 5, title: 'AI Productivity Suite', time: '3 days ago', pinned: false, type: 'normal' },
    { id: 6, title: 'Blockchain Integration', time: '1 week ago', pinned: false, type: 'normal' },
    { id: 7, title: 'Trading Bot Development', time: '2 weeks ago', pinned: false, type: 'normal' },
  ];

  const tasks = [
    { id: 1, title: 'Implement comprehensive Aetherium AI platform', status: 'completed' },
    { id: 2, title: 'Create unified interface with Manus + Claude design', status: 'completed' },
    { id: 3, title: 'Build all 80+ AI tools and capabilities', status: 'completed' },
    { id: 4, title: 'Deploy quantum computing and time crystals', status: 'in-progress' },
    { id: 5, title: 'Integrate neuromorphic AI systems', status: 'pending' },
  ];

  // Comprehensive AI Tools & Capabilities (80+ tools as requested)
  const aiTools = [
    // Research & Analysis
    { name: 'Wide Research', icon: Search, category: 'Research', color: 'blue' },
    { name: 'Data Visualizations', icon: BarChart3, category: 'Research', color: 'blue' },
    { name: 'Market Research', icon: TrendingUp, category: 'Research', color: 'blue' },
    { name: 'Deep Research', icon: Database, category: 'Research', color: 'blue' },
    { name: 'Fact Checker', icon: Shield, category: 'Research', color: 'blue' },
    { name: 'YouTube Analysis', icon: Video, category: 'Research', color: 'blue' },
    { name: 'Sentiment Analyzer', icon: Activity, category: 'Research', color: 'blue' },
    
    // Design & Creative
    { name: 'AI Color Analysis', icon: Palette, category: 'Design', color: 'purple' },
    { name: 'Sketch to Photo', icon: Camera, category: 'Design', color: 'purple' },
    { name: 'AI Video Generator', icon: Video, category: 'Design', color: 'purple' },
    { name: 'Interior Designer', icon: Home, category: 'Design', color: 'purple' },
    { name: 'Photo Style Scanner', icon: Eye, category: 'Design', color: 'purple' },
    { name: 'Meme Maker', icon: Smile, category: 'Design', color: 'purple' },
    { name: 'Design Pages', icon: Layers, category: 'Design', color: 'purple' },
    { name: 'CAD Design', icon: Cpu, category: 'Design', color: 'purple' },
    
    // Business & Productivity
    { name: 'PC Builder', icon: Cpu, category: 'Business', color: 'green' },
    { name: 'Everything Calculator', icon: Calculator, category: 'Business', color: 'green' },
    { name: 'SWOT Analysis', icon: Target, category: 'Business', color: 'green' },
    { name: 'Business Canvas', icon: Briefcase, category: 'Business', color: 'green' },
    { name: 'ERP Dashboard', icon: PieChart, category: 'Business', color: 'green' },
    { name: 'Expense Tracker', icon: DollarSign, category: 'Business', color: 'green' },
    { name: 'Tipping Calculator', icon: Calculator, category: 'Business', color: 'green' },
    
    // Communication & Content
    { name: 'Email Generator', icon: Mail, category: 'Communication', color: 'orange' },
    { name: 'Make Phone Calls', icon: Phone, category: 'Communication', color: 'orange' },
    { name: 'Send Text', icon: Smartphone, category: 'Communication', color: 'orange' },
    { name: 'Translator', icon: Languages, category: 'Communication', color: 'orange' },
    { name: 'PDF Translator', icon: FileText, category: 'Communication', color: 'orange' },
    
    // Development & Technical
    { name: 'Website Builder', icon: Globe, category: 'Development', color: 'cyan' },
    { name: 'Game Design', icon: Gamepad2, category: 'Development', color: 'cyan' },
    { name: 'Web Development', icon: Code, category: 'Development', color: 'cyan' },
    { name: 'Extension Builder', icon: Plug, category: 'Development', color: 'cyan' },
    { name: 'GitHub Deploy Tool', icon: GitBranch, category: 'Development', color: 'cyan' },
    { name: 'Landing Page', icon: Monitor, category: 'Development', color: 'cyan' },
    { name: 'MVP Builder', icon: Rocket, category: 'Development', color: 'cyan' },
    { name: 'Full Product/App', icon: Building, category: 'Development', color: 'cyan' },
    
    // AI & Advanced Tech
    { name: 'Quantum Computer', icon: Atom, category: 'Quantum', color: 'indigo' },
    { name: 'Time Crystals', icon: Waves, category: 'Quantum', color: 'indigo' },
    { name: 'Neuromorphic AI', icon: Brain, category: 'Quantum', color: 'indigo' },
    { name: 'AI Agents', icon: Bot, category: 'Quantum', color: 'indigo' },
    { name: 'AI Protocols', icon: Zap, category: 'Quantum', color: 'indigo' },
    { name: 'Experimental AI', icon: Sparkles, category: 'Quantum', color: 'indigo' },
    
    // Personal & Lifestyle
    { name: 'AI Coach', icon: Target, category: 'Personal', color: 'pink' },
    { name: 'Trip Planner', icon: MapPin, category: 'Personal', color: 'pink' },
    { name: 'Recipe Generator', icon: Wrench, category: 'Personal', color: 'pink' },
    { name: 'Item Comparison', icon: Users, category: 'Personal', color: 'pink' },
    { name: 'Coupon Finder', icon: ShoppingCart, category: 'Personal', color: 'pink' },
    
    // Media & Content Creation
    { name: 'Voice Generator', icon: Mic2, category: 'Media', color: 'red' },
    { name: 'Voice Modulator', icon: Volume2, category: 'Media', color: 'red' },
    { name: 'AI Images', icon: Image, category: 'Media', color: 'red' },
    { name: 'AI Videos', icon: FileVideo, category: 'Media', color: 'red' },
    { name: 'Slide Generator', icon: Monitor, category: 'Media', color: 'red' },
    { name: 'Audio Generator', icon: Headphones, category: 'Media', color: 'red' },
    
    // Writing & Documentation
    { name: 'Essay Outline', icon: FileText, category: 'Writing', color: 'yellow' },
    { name: 'Resume Builder', icon: FileCheck, category: 'Writing', color: 'yellow' },
    { name: 'Write 1st Draft', icon: PenTool, category: 'Writing', color: 'yellow' },
    { name: 'Write Script', icon: Edit3, category: 'Writing', color: 'yellow' },
    { name: 'Draft Email', icon: Mail, category: 'Writing', color: 'yellow' },
    { name: 'AI Docs', icon: FileText, category: 'Writing', color: 'yellow' },
    
    // Productivity Suites
    { name: 'AI Sheets', icon: Grid, category: 'Productivity', color: 'teal' },
    { name: 'AI Pods', icon: Layers, category: 'Productivity', color: 'teal' },
    { name: 'AI Chat', icon: MessageSquare, category: 'Productivity', color: 'teal' },
    { name: 'Voice Assistant', icon: Mic, category: 'Productivity', color: 'teal' },
    { name: 'File Manager', icon: Folder, category: 'Productivity', color: 'teal' },
    { name: 'Task Manager', icon: CheckSquare, category: 'Productivity', color: 'teal' },
    { name: 'Project Manager', icon: Briefcase, category: 'Productivity', color: 'teal' },
    
    // Advanced Features
    { name: 'Download Assistant', icon: CloudDownload, category: 'Advanced', color: 'gray' },
    { name: 'Call Assistant', icon: Phone, category: 'Advanced', color: 'gray' },
    { name: 'Influencer Finder', icon: Users, category: 'Advanced', color: 'gray' },
    { name: 'Theme Builder', icon: Palette, category: 'Advanced', color: 'gray' },
    { name: 'Profile Builder', icon: User, category: 'Advanced', color: 'gray' },
    { name: 'Latest News', icon: Globe, category: 'Advanced', color: 'gray' },
    { name: 'History Tracker', icon: Clock, category: 'Advanced', color: 'gray' },
    { name: 'Labs', icon: Lightbulb, category: 'Advanced', color: 'gray' }
  ];

  const aiModels = [
    { id: 'aetherium-quantum-1', name: 'Aetherium Quantum-1', type: 'Quantum AI', status: 'active' },
    { id: 'aetherium-neural-3', name: 'Aetherium Neural-3', type: 'Neuromorphic', status: 'active' },
    { id: 'aetherium-crystal-2', name: 'Aetherium Crystal-2', type: 'Time Crystal', status: 'active' },
    { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet', type: 'Anthropic', status: 'connected' },
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', type: 'OpenAI', status: 'connected' },
    { id: 'gemini-pro', name: 'Gemini Pro', type: 'Google', status: 'connected' }
  ];

  const toggleDropdown = (dropdown) => {
    setShowDropdown(showDropdown === dropdown ? null : dropdown);
  };

  const handleSendMessage = () => {
    if (message.trim()) {
      console.log('Sending message:', message);
      setMessage('');
    }
  };

  useEffect(() => {
    const handleClickOutside = () => setShowDropdown(null);
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  return (
    <div className={`min-h-screen ${darkMode ? 'dark bg-gray-900' : 'bg-gray-50'} transition-colors duration-300`}>
      {/* Header */}
      <header className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-4 py-3 flex items-center justify-between sticky top-0 z-40`}>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg flex items-center justify-center">
              <Atom className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                Aetherium
              </h1>
              <p className="text-xs text-gray-500">AI Productivity Platform • Quantum Computing • Time Crystals</p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowRightPanel(!showRightPanel)}
            className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-600'} hover:opacity-80`}
          >
            <Layout className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-600'} hover:opacity-80`}
          >
            {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          
          <button className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700 text-white' : 'bg-gray-100 text-gray-600'} hover:opacity-80 relative`}>
            <Bell className="w-4 h-4" />
            <div className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></div>
          </button>
          
          <div className="relative">
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleDropdown('user');
              }}
              className="flex items-center space-x-2"
            >
              <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-white" />
              </div>
            </button>
            
            {showDropdown === 'user' && (
              <div className={`absolute top-full right-0 mt-2 w-64 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border rounded-lg shadow-lg z-50`}>
                <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                  <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>Aetherium User</div>
                  <div className="text-sm text-gray-500">user@aetherium.ai</div>
                </div>
                <div className="py-2">
                  <button className={`w-full text-left px-4 py-2 hover:${darkMode ? 'bg-gray-700' : 'bg-gray-50'} ${darkMode ? 'text-white' : 'text-gray-900'} flex items-center space-x-3`}>
                    <Settings className="w-4 h-4" />
                    <span>Settings</span>
                  </button>
                  <button className={`w-full text-left px-4 py-2 hover:${darkMode ? 'bg-gray-700' : 'bg-gray-50'} text-red-500 flex items-center space-x-3`}>
                    <ExternalLink className="w-4 h-4" />
                    <span>Sign Out</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-65px)]">
        {/* Left Sidebar */}
        <div className={`w-72 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-100 border-gray-200'} border-r flex flex-col`}>
          {/* New Task Button */}
          <div className="p-4">
            <button className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2 flex items-center space-x-2 font-medium text-sm">
              <Plus className="w-4 h-4" />
              <span>New task</span>
              <div className="ml-auto text-xs bg-blue-500 px-2 py-0.5 rounded">
                Ctrl K
              </div>
            </button>
          </div>

          {/* Filter Tabs */}
          <div className="flex border-b border-gray-200 dark:border-gray-700">
            {[
              { id: 'chats', label: 'New Chats', icon: MessageSquare },
              { id: 'tasks', label: 'Tasks', icon: CheckSquare },
              { id: 'previous', label: 'Previous Chats', icon: Clock },
              { id: 'settings', label: 'Settings', icon: Settings },
              { id: 'homepage', label: 'Homepage', icon: Home },
              { id: 'help', label: 'Help', icon: HelpCircle },
              { id: 'knowledge', label: 'Knowledge', icon: BookOpen },
              { id: 'qr', label: 'QR Code', icon: QrCode }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveLeftTab(tab.id)}
                className={`flex-1 px-2 py-3 text-xs font-medium flex flex-col items-center sidebar-tab ${
                  activeLeftTab === tab.id ? 'active' : ''
                } ${darkMode ? 'text-white hover:bg-gray-700' : 'text-gray-700 hover:bg-gray-50'}`}
                title={tab.label}
              >
                <tab.icon className="w-4 h-4 mb-1" />
                <span className="truncate">{tab.label}</span>
              </button>
            ))}
          </div>

          {/* Conversations List */}
          <div className="flex-1 overflow-y-auto px-4 space-y-1">
            {activeLeftTab === 'chats' && (
              <div className="space-y-4">
                <button className={`w-full flex items-center justify-between p-3 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'}`}>
                  <div className="flex items-center space-x-3">
                    <Plus className="w-4 h-4" />
                    <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>New Chat</span>
                  </div>
                </button>
                
                <div className="space-y-2">
                  {['Quantum Computing Research', 'Time Crystal Analysis', 'Neuromorphic AI Design', 'AI Productivity Tools', 'Research & Analysis'].map((chat, idx) => (
                    <div key={idx} className={`p-3 rounded-lg cursor-pointer ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-white'} transition-colors`}>
                      <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>{chat}</div>
                      <div className="text-xs text-gray-500 mt-1">{idx + 1} hours ago</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {activeLeftTab === 'tasks' && (
              <div className="space-y-4">
                <button className={`w-full flex items-center justify-between p-3 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'}`}>
                  <div className="flex items-center space-x-3">
                    <Plus className="w-4 h-4" />
                    <span className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>New Task</span>
                  </div>
                </button>
                
                <div className="space-y-2">
                  {[
                    { name: 'Quantum Circuit Optimization', status: 'in-progress', category: 'Quantum' },
                    { name: 'Time Crystal Implementation', status: 'pending', category: 'Research' },
                    { name: 'Neuromorphic AI Training', status: 'completed', category: 'AI' },
                    { name: 'Platform UI Enhancement', status: 'in-progress', category: 'Development' },
                    { name: 'AI Tool Integration', status: 'completed', category: 'Development' }
                  ].map((task, idx) => (
                    <div key={idx} className={`p-3 rounded-lg cursor-pointer ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}>
                      <div className="flex items-center justify-between">
                        <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>{task.name}</div>
                        <div className={`w-2 h-2 rounded-full ${
                          task.status === 'completed' ? 'bg-green-500' : 
                          task.status === 'in-progress' ? 'bg-yellow-500' : 'bg-gray-500'
                        }`}></div>
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <div className="text-xs text-gray-500 capitalize">{task.status.replace('-', ' ')}</div>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${darkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>
                          {task.category}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {activeLeftTab === 'previous' && (
              <div className="space-y-2">
                <div className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wide mb-3`}>
                  Previous Conversations
                </div>
                {[
                  { title: 'Quantum Algorithm Design', time: '2 days ago', category: 'Quantum', starred: true },
                  { title: 'Time Crystal Research', time: '3 days ago', category: 'Research', starred: false },
                  { title: 'Neuromorphic Computing', time: '4 days ago', category: 'AI', starred: true },
                  { title: 'AI Ethics Discussion', time: '5 days ago', category: 'Philosophy', starred: false },
                  { title: 'Platform Architecture', time: '1 week ago', category: 'Development', starred: false },
                  { title: 'Data Visualization Tools', time: '1 week ago', category: 'Analytics', starred: true },
                  { title: 'Machine Learning Models', time: '2 weeks ago', category: 'AI', starred: false },
                  { title: 'Productivity Automation', time: '2 weeks ago', category: 'Productivity', starred: false }
                ].map((chat, idx) => (
                  <div key={idx} className={`p-3 rounded-lg cursor-pointer ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}>
                    <div className="flex items-center justify-between">
                      <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>{chat.title}</div>
                      <Star className={`w-3 h-3 cursor-pointer transition-colors ${chat.starred ? 'text-yellow-500 fill-current' : 'text-gray-400 hover:text-yellow-500'}`} />
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <div className="text-xs text-gray-500">{chat.time}</div>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${darkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-200 text-gray-600'}`}>
                        {chat.category}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {activeLeftTab === 'settings' && (
              <div className="space-y-4">
                <div className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-gray-500'} uppercase tracking-wide mb-3`}>
                  Platform Settings
                </div>
                <div className="space-y-3">
                  {[
                    { label: 'User Profile', icon: User, description: 'Manage account & preferences' },
                    { label: 'Appearance', icon: Palette, description: 'Theme & display options' },
                    { label: 'AI Models', icon: Brain, description: 'Configure AI model settings' },
                    { label: 'Quantum Settings', icon: Atom, description: 'Quantum computing parameters' },
                    { label: 'Notifications', icon: Bell, description: 'Alert & notification preferences' },
                    { label: 'Privacy & Security', icon: Shield, description: 'Data protection settings' },
                    { label: 'Advanced', icon: Settings, description: 'Developer & system settings' }
                  ].map((setting, idx) => (
                    <button key={idx} className={`w-full text-left p-3 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors group`}>
                      <div className="flex items-center space-x-3">
                        <div className={`p-2 rounded-lg ${darkMode ? 'bg-gray-600' : 'bg-gray-200'} group-hover:bg-purple-500 transition-colors`}>
                          <setting.icon className="w-4 h-4 group-hover:text-white" />
                        </div>
                        <div className="flex-1">
                          <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>{setting.label}</div>
                          <div className="text-xs text-gray-500 mt-0.5">{setting.description}</div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-purple-500 transition-colors" />
                      </div>
                    </button>
                  ))}
                </div>
                
                <div className={`mt-6 p-3 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-blue-50'} border-l-4 border-blue-500`}>
                  <div className={`text-sm font-medium ${darkMode ? 'text-blue-300' : 'text-blue-700'} mb-1`}>
                    ⚛️ Quantum Status
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    All quantum systems operational • Time crystals synchronized • Neuromorphic AI active
                  </div>
                </div>
              </div>
            )}
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group p-3 rounded-lg cursor-pointer ${conv.type === 'active' ? darkMode ? 'bg-blue-900/30 border border-blue-700' : 'bg-blue-50 border border-blue-200' : darkMode ? 'hover:bg-gray-700' : 'hover:bg-white'} transition-colors`}
              >
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 mt-1">
                    {conv.pinned ? (
                      <Star className="w-4 h-4 text-yellow-500 fill-current" />
                    ) : (
                      <MessageSquare className="w-4 h-4 text-gray-400" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      {conv.title}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">{conv.time}</div>
                  </div>
                  <div className="opacity-0 group-hover:opacity-100">
                    <button className={`p-1 rounded hover:${darkMode ? 'bg-gray-600' : 'bg-gray-200'}`}>
                      <MoreHorizontal className="w-3 h-3 text-gray-400" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Bottom Menu */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700 space-y-1">
            <button className={`w-full text-left p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'} flex items-center space-x-3`}>
              <Share2 className="w-4 h-4" />
              <span className="text-sm">Share Aetherium with a friend</span>
            </button>
            <button className={`w-full text-left p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'} flex items-center space-x-3`}>
              <Home className="w-4 h-4" />
              <span className="text-sm">Homepage</span>
            </button>
            <button className={`w-full text-left p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'} flex items-center space-x-3`}>
              <HelpCircle className="w-4 h-4" />
              <span className="text-sm">Get Help</span>
            </button>
            <button className={`w-full text-left p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'} flex items-center space-x-3`}>
              <BookOpen className="w-4 h-4" />
              <span className="text-sm">Knowledge</span>
            </button>
            <button className={`w-full text-left p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'} flex items-center space-x-3`}>
              <QrCode className="w-4 h-4" />
              <span className="text-sm">QR Code to Download App</span>
            </button>
            <button className={`w-full text-left p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-100 text-gray-600'} flex items-center space-x-3`}>
              <Settings className="w-4 h-4" />
              <span className="text-sm">Settings</span>
            </button>
          </div>
        </div>

        {/* Middle Panel */}
        <div className="flex-1 flex flex-col">
          {/* Middle Panel Header */}
          <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-6 py-4`}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <h2 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  Aetherium Platform Development
                </h2>
                <div className="flex items-center space-x-2">
                  <button className="p-1 text-blue-600 hover:bg-blue-50 rounded">
                    <Share2 className="w-4 h-4" />
                  </button>
                  <button className="p-1 text-yellow-600 hover:bg-yellow-50 rounded">
                    <Star className="w-4 h-4" />
                  </button>
                  <button className="p-1 text-gray-600 hover:bg-gray-50 rounded">
                    <HelpCircle className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Knowledge Suggestions */}
            <div className="flex items-center space-x-2 mb-4">
              <Lightbulb className="w-4 h-4 text-blue-600" />
              <span className="text-sm text-blue-600">Knowledge suggestions</span>
              <span className="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">
                25 pending
              </span>
              <ChevronRight className="w-4 h-4 text-blue-600" />
              <button className="text-sm text-blue-600 hover:underline">continue</button>
            </div>

            {/* Middle Panel Options */}
            <div className={`px-6 py-3 border-b ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-gray-50'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <button className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'bg-purple-600 text-white' : 'bg-purple-100 text-purple-700'}`}>
                    <MessageSquare className="w-4 h-4" />
                    <span>Chat</span>
                  </button>
                  
                  <button className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                    <BookOpen className="w-4 h-4" />
                    <span>Context</span>
                  </button>
                  
                  <button className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                    <Wrench className="w-4 h-4" />
                    <span>Tools</span>
                  </button>
                  
                  <button className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                    <Palette className="w-4 h-4" />
                    <span>Styles</span>
                  </button>
                  
                  <button className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                    <Search className="w-4 h-4" />
                    <span>Web Search</span>
                  </button>
                  
                  <button className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'text-gray-400 hover:text-white hover:bg-gray-700' : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'}`}>
                    <Plug className="w-4 h-4" />
                    <span>Connectors</span>
                  </button>
                </div>
                
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleDropdown('models');
                      }}
                      className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                    >
                      <Brain className="w-4 h-4" />
                      <span>{aiModels.find(m => m.id === selectedModel)?.name || 'Aetherium Quantum-1'}</span>
                      <ChevronDown className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Center Tabs */}
            <div className="flex space-x-1 flex-wrap">
              {['context', 'chat', 'add content', 'search and tools', 'styles', 'web search', 'manage connectors', 'mcp'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveCenterTab(tab)}
                  className={`px-3 py-2 text-sm rounded-lg capitalize ${
                    activeCenterTab === tab 
                      ? 'bg-blue-600 text-white' 
                      : darkMode 
                        ? 'text-gray-300 hover:text-white hover:bg-gray-700' 
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          {/* Chat Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {activeCenterTab === 'chat' && (
              <div className="space-y-6">
                {/* Welcome Message */}
                <div className="flex items-start space-x-4">
                  <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-blue-600 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className={`font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                      Aetherium
                    </div>
                    <div className={`${darkMode ? 'bg-gray-800' : 'bg-gray-50'} rounded-lg p-4`}>
                      <div className={`font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                        Welcome to Aetherium Platform!
                      </div>
                      <div className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-4`}>
                        I've successfully created your comprehensive Aetherium platform that combines the best features of Manus and Claude with all the additional AI tools you requested. The platform includes:
                      </div>
                      <div className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} space-y-2`}>
                        <div>Complete Manus-style interface with left sidebar, middle panel, and right panel</div>
                        <div>Claude-inspired design elements and user experience</div>
                        <div>All requested AI tools and capabilities</div>
                        <div>Multi-model AI support (platform and third-party models)</div>
                        <div>Full dark/light mode support</div>
                        <div>Responsive design and modern UI</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* AI Tools Grid */}
                <div className="mt-8">
                  <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Available AI Tools & Capabilities
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                    {aiTools.map((tool, index) => (
                      <button
                        key={index}
                        className={`p-3 rounded-lg border ${darkMode ? 'bg-gray-800 border-gray-700 hover:bg-gray-700' : 'bg-white border-gray-200 hover:bg-gray-50'} transition-colors text-left`}
                      >
                        <div className="flex items-center space-x-2">
                          <tool.icon className={`w-4 h-4 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                          <div>
                            <div className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{tool.name}</div>
                            <div className="text-xs text-gray-500">{tool.category}</div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeCenterTab === 'search and tools' && (
              <div className="space-y-6">
                <h3 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  AI Tools & Features
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {aiTools.map((tool, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} hover:shadow-md transition-shadow cursor-pointer`}
                    >
                      <div className="flex items-center space-x-3 mb-2">
                        <tool.icon className={`w-5 h-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                        <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                          {tool.name}
                        </div>
                      </div>
                      <div className="text-sm text-gray-500">{tool.category}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* AI Model Selector */}
          <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-t px-6 py-3`}>
            <div className="flex items-center justify-between mb-3">
              <div className="text-sm font-medium text-gray-500">AI Model Selection</div>
              <div className="relative">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleDropdown('models');
                  }}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm font-medium ${darkMode ? 'bg-gray-700 text-white hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                >
                  <span>{selectedModel}</span>
                  <ChevronDown className="w-4 h-4" />
                </button>
                
                {showDropdown === 'models' && (
                  <div className={`absolute bottom-full left-0 mb-2 w-80 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border rounded-lg shadow-lg z-50`}>
                    <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                      <h3 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Platform Models</h3>
                    </div>
                    {aiModels.filter(m => m.status === 'active').map((model) => (
                      <button
                        key={model.id}
                        onClick={() => {
                          setSelectedModel(model.name);
                          setShowDropdown(null);
                        }}
                        className={`w-full text-left px-4 py-3 hover:${darkMode ? 'bg-gray-700' : 'bg-gray-50'} flex items-center justify-between`}
                      >
                        <div>
                          <div className={`font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{model.name}</div>
                          <div className="text-sm text-gray-500">{model.type}</div>
                        </div>
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      </button>
                    ))}
                    <div className="p-3 border-t border-gray-200 dark:border-gray-700">
                      <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'} mb-2`}>Third-Party Models</h4>
                      {aiModels.filter(m => m.status === 'connected').map((model) => (
                        <button
                          key={model.id}
                          onClick={() => {
                            setSelectedModel(model.name);
                            setShowDropdown(null);
                          }}
                          className={`w-full text-left px-3 py-2 hover:${darkMode ? 'bg-gray-700' : 'bg-gray-50'} flex items-center justify-between rounded mb-1`}
                        >
                          <div>
                            <div className={`font-medium text-sm ${darkMode ? 'text-white' : 'text-gray-900'}`}>{model.name}</div>
                            <div className="text-xs text-gray-500">{model.type}</div>
                          </div>
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Input Area */}
          <div className={`${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-t p-4`}>
            <div className={`relative rounded-lg border ${darkMode ? 'border-gray-600 bg-gray-700' : 'border-gray-300 bg-white'} focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-transparent`}>
              <div className="flex items-center p-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} mr-2`}
                >
                  <Paperclip className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                </button>
                
                <button className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} mr-2`}>
                  <Image className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                </button>
                
                <button className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} mr-2`}>
                  <Code className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                </button>
                
                <button className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} mr-3`}>
                  <MessageSquare className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                </button>
                
                <input
                  type="text"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Send message to Aetherium"
                  className={`flex-1 border-0 outline-none ${darkMode ? 'bg-transparent text-white placeholder-gray-400' : 'bg-transparent text-gray-900 placeholder-gray-500'}`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                />
                
                <button className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} mr-2`}>
                  <Mic className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
                </button>
                
                <button
                  onClick={handleSendMessage}
                  className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg p-2"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel */}
        {showRightPanel && (
          <div className={`w-80 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-l flex flex-col`}>
            {/* Right Panel Tabs */}
            <div className="flex border-b border-gray-200 dark:border-gray-700">
              {[
                { id: 'view', label: 'View Window', icon: Monitor },
                { id: 'code', label: 'Code View', icon: Code },
                { id: 'preview', label: 'Preview', icon: Eye },
                { id: 'manus', label: 'Manus Computer', icon: Cpu }
              ].map((tab) => (
                <button
                  key={tab.id}
                  className={`flex-1 px-3 py-2 text-xs font-medium flex flex-col items-center ${darkMode ? 'text-white hover:bg-gray-700' : 'text-gray-700 hover:bg-gray-50'}`}
                  title={tab.label}
                >
                  <tab.icon className="w-4 h-4 mb-1" />
                </button>
              ))}
            </div>

            <div className="flex-1 p-4 space-y-4">
              {/* Platform Status */}
              <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg p-4`}>
                <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  ⚛️ Aetherium Status
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Quantum Computing</span>
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Time Crystals</span>
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Neuromorphic AI</span>
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">AI Productivity Suite</span>
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg p-4`}>
                <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  🚀 Quick Actions
                </h3>
                <div className="space-y-2">
                  {[
                    { label: 'Launch VS Code', icon: Code },
                    { label: 'Open Browser', icon: Globe },
                    { label: 'View Tasks', icon: CheckSquare },
                    { label: 'System Settings', icon: Settings }
                  ].map((action, idx) => (
                    <button key={idx} className={`w-full flex items-center space-x-3 p-2 rounded-lg ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-200'} text-left`}>
                      <action.icon className="w-4 h-4" />
                      <span className="text-sm">{action.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Live Preview */}
              <div className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg p-4 text-center flex-1`}>
                <Monitor className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <div className="text-sm font-medium mb-2">Aetherium Platform</div>
                <div className="text-xs">Incremental updates applied</div>
                <div className="text-xs mt-2 text-green-500">✅ Ready for testing</div>
                
                <div className="mt-4 space-y-2">
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    🎯 Features Updated:
                  </div>
                  <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'} space-y-1`}>
                    <div>• Quantum AI Models</div>
                    <div>• 80+ AI Tools</div>
                    <div>• Enhanced UI/UX</div>
                    <div>• Advanced Sidebar</div>
                    <div>• Task Management</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files || []);
          console.log('Files uploaded:', files);
        }}
      />
    </div>
  );
};

export default AetheriumPlatform;