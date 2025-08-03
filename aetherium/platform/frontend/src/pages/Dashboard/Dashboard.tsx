import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Progress, Alert, Button, Typography, Space } from 'antd';
import {
  ThunderboltOutlined,
  ClockCircleOutlined,
  BranchesOutlined,
  RobotOutlined,
  WifiOutlined,
  BarChartOutlined,
  ExperimentOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const { Title: AntTitle, Text } = Typography;

interface SystemMetrics {
  quantum_computer: {
    status: string;
    fidelity: number;
    circuits_executed: number;
    error_rate: number;
  };
  time_crystal_engine: {
    status: string;
    coherence: number;
    crystals_active: number;
    synchronization_rate: number;
  };
  neuromorphic_processor: {
    status: string;
    spike_rate: number;
    neurons_active: number;
    network_efficiency: number;
  };
  iot_manager: {
    status: string;
    devices_connected: number;
    total_devices: number;
    data_throughput: number;
  };
}

const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      // Simulate API call - in real implementation, this would call the backend
      const mockMetrics: SystemMetrics = {
        quantum_computer: {
          status: 'healthy',
          fidelity: 0.987,
          circuits_executed: 1247,
          error_rate: 0.013
        },
        time_crystal_engine: {
          status: 'healthy',
          coherence: 0.952,
          crystals_active: 8,
          synchronization_rate: 0.996
        },
        neuromorphic_processor: {
          status: 'healthy',
          spike_rate: 147.3,
          neurons_active: 9876,
          network_efficiency: 0.891
        },
        iot_manager: {
          status: 'healthy',
          devices_connected: 23,
          total_devices: 25,
          data_throughput: 1024.5
        }
      };
      
      setMetrics(mockMetrics);
      setLastUpdate(new Date());
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a';
      case 'warning': return '#faad14';
      case 'error': return '#f5222d';
      default: return '#d9d9d9';
    }
  };

  const quantumPerformanceData = {
    labels: ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30'],
    datasets: [
      {
        label: 'Quantum Fidelity',
        data: [0.985, 0.987, 0.989, 0.986, 0.987, 0.988, 0.987],
        borderColor: '#1890ff',
        backgroundColor: 'rgba(24, 144, 255, 0.1)',
        tension: 0.4,
      },
      {
        label: 'Time Crystal Coherence',
        data: [0.948, 0.951, 0.953, 0.950, 0.952, 0.954, 0.952],
        borderColor: '#52c41a',
        backgroundColor: 'rgba(82, 196, 26, 0.1)',
        tension: 0.4,
      },
    ],
  };

  const systemDistribution = {
    labels: ['Quantum Computing', 'Time Crystals', 'Neuromorphic', 'IoT Systems'],
    datasets: [
      {
        data: [30, 25, 25, 20],
        backgroundColor: ['#1890ff', '#52c41a', '#faad14', '#f5222d'],
        borderWidth: 2,
        borderColor: '#1f1f1f',
      },
    ],
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <div>Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <AntTitle level={2} style={{ color: '#fff', marginBottom: 16 }}>
            Quantum AI Platform Dashboard
          </AntTitle>
          <Text type="secondary">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </Text>
        </Col>
      </Row>

      {/* System Status Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ background: '#262626', border: '1px solid #434343' }}>
            <Statistic
              title={<span style={{ color: '#fff' }}>Quantum Computer</span>}
              value={metrics?.quantum_computer.fidelity || 0}
              precision={3}
              prefix={<ThunderboltOutlined style={{ color: getStatusColor(metrics?.quantum_computer.status || '') }} />}
              suffix="fidelity"
              valueStyle={{ color: '#fff' }}
            />
            <Progress 
              percent={(metrics?.quantum_computer.fidelity || 0) * 100} 
              showInfo={false}
              strokeColor={getStatusColor(metrics?.quantum_computer.status || '')}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card style={{ background: '#262626', border: '1px solid #434343' }}>
            <Statistic
              title={<span style={{ color: '#fff' }}>Time Crystals</span>}
              value={metrics?.time_crystal_engine.coherence || 0}
              precision={3}
              prefix={<ClockCircleOutlined style={{ color: getStatusColor(metrics?.time_crystal_engine.status || '') }} />}
              suffix="coherence"
              valueStyle={{ color: '#fff' }}
            />
            <Progress 
              percent={(metrics?.time_crystal_engine.coherence || 0) * 100} 
              showInfo={false}
              strokeColor={getStatusColor(metrics?.time_crystal_engine.status || '')}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card style={{ background: '#262626', border: '1px solid #434343' }}>
            <Statistic
              title={<span style={{ color: '#fff' }}>Neuromorphic</span>}
              value={metrics?.neuromorphic_processor.spike_rate || 0}
              precision={1}
              prefix={<BranchesOutlined style={{ color: getStatusColor(metrics?.neuromorphic_processor.status || '') }} />}
              suffix="Hz"
              valueStyle={{ color: '#fff' }}
            />
            <Progress 
              percent={(metrics?.neuromorphic_processor.network_efficiency || 0) * 100} 
              showInfo={false}
              strokeColor={getStatusColor(metrics?.neuromorphic_processor.status || '')}
            />
          </Card>
        </Col>

        <Col xs={24} sm={12} lg={6}>
          <Card style={{ background: '#262626', border: '1px solid #434343' }}>
            <Statistic
              title={<span style={{ color: '#fff' }}>IoT Devices</span>}
              value={metrics?.iot_manager.devices_connected || 0}
              prefix={<WifiOutlined style={{ color: getStatusColor(metrics?.iot_manager.status || '') }} />}
              suffix={`/ ${metrics?.iot_manager.total_devices || 0}`}
              valueStyle={{ color: '#fff' }}
            />
            <Progress 
              percent={((metrics?.iot_manager.devices_connected || 0) / (metrics?.iot_manager.total_devices || 1)) * 100} 
              showInfo={false}
              strokeColor={getStatusColor(metrics?.iot_manager.status || '')}
            />
          </Card>
        </Col>
      </Row>

      {/* Charts and Detailed Metrics */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={16}>
          <Card 
            title={<span style={{ color: '#fff' }}>System Performance Over Time</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <Line 
              data={quantumPerformanceData} 
              options={{
                responsive: true,
                plugins: {
                  legend: {
                    labels: { color: '#fff' }
                  }
                },
                scales: {
                  x: {
                    ticks: { color: '#fff' },
                    grid: { color: '#434343' }
                  },
                  y: {
                    ticks: { color: '#fff' },
                    grid: { color: '#434343' },
                    min: 0.94,
                    max: 1.0
                  }
                }
              }}
            />
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card 
            title={<span style={{ color: '#fff' }}>System Resource Distribution</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <Doughnut 
              data={systemDistribution}
              options={{
                responsive: true,
                plugins: {
                  legend: {
                    labels: { color: '#fff' }
                  }
                }
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Quick Actions */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card 
            title={<span style={{ color: '#fff' }}>Quick Actions</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <Space wrap>
              <Button type="primary" icon={<ExperimentOutlined />}>
                Run Quantum Circuit
              </Button>
              <Button icon={<ClockCircleOutlined />}>
                Sync Time Crystals
              </Button>
              <Button icon={<BranchesOutlined />}>
                Inject Spike Pattern
              </Button>
              <Button icon={<RobotOutlined />}>
                Start AI Optimization
              </Button>
              <Button icon={<WifiOutlined />}>
                Discover IoT Devices
              </Button>
              <Button icon={<BarChartOutlined />}>
                Generate Report
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* System Alerts */}
      <Row style={{ marginTop: 24 }}>
        <Col span={24}>
          <Alert
            message="System Status: All components operational"
            description="Quantum fidelity: 98.7% • Time crystal coherence: 95.2% • Neuromorphic efficiency: 89.1% • IoT connectivity: 92%"
            type="success"
            showIcon
            style={{ background: '#162312', border: '1px solid #389e0d' }}
          />
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;