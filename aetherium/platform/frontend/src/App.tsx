import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout, ConfigProvider, theme } from 'antd';
import 'antd/dist/reset.css';
import './App.css';

// Import components
import Sidebar from './components/Sidebar/Sidebar';
import Dashboard from './pages/Dashboard/Dashboard';
import QuantumLab from './pages/QuantumLab/QuantumLab';
import TimeCrystals from './pages/TimeCrystals/TimeCrystals';
import Neuromorphic from './pages/Neuromorphic/Neuromorphic';
import AIOptimization from './pages/AIOptimization/AIOptimization';
import IoTDevices from './pages/IoTDevices/IoTDevices';
import SystemMetrics from './pages/SystemMetrics/SystemMetrics';
import Settings from './pages/Settings/Settings';
import ProductivitySuite from './pages/ProductivitySuite/ProductivitySuite';

// Context providers
import { ApiProvider } from './contexts/ApiContext';
import { AuthProvider } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';

const { Header, Content, Sider } = Layout;

const App: React.FC = () => {
  const [collapsed, setCollapsed] = React.useState(false);

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#1890ff',
          colorBgContainer: '#001529',
          colorText: '#ffffff',
        },
      }}
    >
      <AuthProvider>
        <ApiProvider>
          <WebSocketProvider>
            <Router>
              <Layout style={{ minHeight: '100vh' }}>
                <Sider 
                  collapsible 
                  collapsed={collapsed} 
                  onCollapse={setCollapsed}
                  theme="dark"
                  width={250}
                >
                  <div className="logo">
                    <h2 style={{ color: '#fff', textAlign: 'center', padding: '16px' }}>
                      {collapsed ? 'AEI' : 'Aetherium AI'}
                    </h2>
                  </div>
                  <Sidebar />
                </Sider>
                
                <Layout>
                  <Header 
                    style={{ 
                      padding: '0 24px', 
                      background: '#001529',
                      borderBottom: '1px solid #303030'
                    }}
                  >
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      color: '#fff'
                    }}>
                      <h1 style={{ margin: 0, color: '#fff' }}>
                        Aetherium AI Productivity Platform
                      </h1>
                      <div>
                        {/* Status indicators, user menu, etc. */}
                      </div>
                    </div>
                  </Header>
                  
                  <Content style={{ margin: '24px', overflow: 'auto' }}>
                    <div style={{ 
                      padding: 24, 
                      minHeight: 360, 
                      background: '#1f1f1f',
                      borderRadius: '8px'
                    }}>
                      <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/productivity" element={<ProductivitySuite />} />
                        <Route path="/quantum" element={<QuantumLab />} />
                        <Route path="/time-crystals" element={<TimeCrystals />} />
                        <Route path="/neuromorphic" element={<Neuromorphic />} />
                        <Route path="/ai-optimization" element={<AIOptimization />} />
                        <Route path="/iot" element={<IoTDevices />} />
                        <Route path="/metrics" element={<SystemMetrics />} />
                        <Route path="/settings" element={<Settings />} />
                      </Routes>
                    </div>
                  </Content>
                </Layout>
              </Layout>
            </Router>
          </WebSocketProvider>
        </ApiProvider>
      </AuthProvider>
    </ConfigProvider>
  );
};

export default App;