import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Button,
  Space,
  Alert,
  Table,
  Tag,
  Slider,
  InputNumber,
  Switch,
  Tabs,
  Timeline,
  notification
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  SyncOutlined,
  FireOutlined
} from '@ant-design/icons';
import { Line, Scatter } from 'react-chartjs-2';
import { useApi } from '../../contexts/ApiContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface TimeCrystal {
  id: string;
  name: string;
  state: 'active' | 'inactive' | 'synchronizing';
  coherence: number;
  frequency: number;
  phase: number;
  entanglement_strength: number;
  created_at: string;
}

interface CrystalMetrics {
  coherence_history: number[];
  synchronization_events: number;
  phase_evolution: number[];
  entanglement_network: any;
}

const TimeCrystals: React.FC = () => {
  const [crystals, setCrystals] = useState<TimeCrystal[]>([]);
  const [metrics, setMetrics] = useState<CrystalMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedCrystal, setSelectedCrystal] = useState<string | null>(null);
  const [simulationParams, setSimulationParams] = useState({
    num_crystals: 8,
    coupling_strength: 0.5,
    temperature: 0.1,
    external_field: 0.3
  });

  const { api } = useApi();
  const { subscribe, sendMessage } = useWebSocket();

  useEffect(() => {
    loadTimeCrystals();
    loadMetrics();

    // Subscribe to real-time updates
    const unsubscribe = subscribe('time_crystal_update', (data) => {
      if (data.crystal_id) {
        setCrystals(prev => prev.map(crystal => 
          crystal.id === data.crystal_id ? { ...crystal, ...data } : crystal
        ));
      }
      if (data.metrics) {
        setMetrics(data.metrics);
      }
    });

    return unsubscribe;
  }, [subscribe]);

  const loadTimeCrystals = async () => {
    try {
      const response = await api.get('/time-crystals/crystals');
      setCrystals(response.data.crystals || []);
    } catch (error) {
      console.error('Failed to load time crystals:', error);
      // Mock data for development
      setCrystals([
        {
          id: 'tc_001',
          name: 'Alpha Crystal',
          state: 'active',
          coherence: 0.95,
          frequency: 2.4,
          phase: 0.25,
          entanglement_strength: 0.87,
          created_at: '2025-01-01T00:00:00Z'
        },
        {
          id: 'tc_002',
          name: 'Beta Crystal',
          state: 'synchronizing',
          coherence: 0.82,
          frequency: 2.4,
          phase: 0.26,
          entanglement_strength: 0.78,
          created_at: '2025-01-01T00:00:00Z'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await api.get('/time-crystals/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
      // Mock data for development
      setMetrics({
        coherence_history: Array.from({ length: 50 }, (_, i) => 0.8 + 0.2 * Math.sin(i * 0.1) + Math.random() * 0.1),
        synchronization_events: 127,
        phase_evolution: Array.from({ length: 50 }, (_, i) => (i * 0.05) % (2 * Math.PI)),
        entanglement_network: {}
      });
    }
  };

  const startSimulation = async () => {
    try {
      setIsSimulating(true);
      const response = await api.post('/time-crystals/simulate', simulationParams);
      notification.success({
        message: 'Time Crystal Simulation Started',
        description: 'Multi-dimensional time crystal simulation is now running.'
      });
    } catch (error) {
      console.error('Failed to start simulation:', error);
      notification.error({
        message: 'Simulation Error',
        description: 'Failed to start time crystal simulation.'
      });
    }
  };

  const stopSimulation = async () => {
    try {
      await api.post('/time-crystals/stop');
      setIsSimulating(false);
      notification.info({
        message: 'Simulation Stopped',
        description: 'Time crystal simulation has been stopped.'
      });
    } catch (error) {
      console.error('Failed to stop simulation:', error);
    }
  };

  const synchronizeCrystals = async () => {
    try {
      await api.post('/time-crystals/synchronize');
      notification.success({
        message: 'Synchronization Initiated',
        description: 'Time crystal network synchronization process started.'
      });
      sendMessage('crystal_sync_request', { timestamp: Date.now() });
    } catch (error) {
      console.error('Failed to synchronize crystals:', error);
    }
  };

  const resetCrystal = async (crystalId: string) => {
    try {
      await api.post(`/time-crystals/reset/${crystalId}`);
      notification.success({
        message: 'Crystal Reset',
        description: `Time crystal ${crystalId} has been reset to initial state.`
      });
      loadTimeCrystals();
    } catch (error) {
      console.error('Failed to reset crystal:', error);
    }
  };

  const coherenceChartData = {
    labels: Array.from({ length: 50 }, (_, i) => i),
    datasets: [
      {
        label: 'Coherence Level',
        data: metrics?.coherence_history || [],
        borderColor: '#52c41a',
        backgroundColor: 'rgba(82, 196, 26, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const phaseChartData = {
    labels: Array.from({ length: 50 }, (_, i) => i),
    datasets: [
      {
        label: 'Phase Evolution',
        data: metrics?.phase_evolution || [],
        borderColor: '#1890ff',
        backgroundColor: 'rgba(24, 144, 255, 0.1)',
        fill: false,
        tension: 0.4
      }
    ]
  };

  const crystalColumns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: TimeCrystal) => (
        <Space>
          <span style={{ fontWeight: 500 }}>{text}</span>
          <Tag color={record.state === 'active' ? 'green' : record.state === 'synchronizing' ? 'blue' : 'default'}>
            {record.state.toUpperCase()}
          </Tag>
        </Space>
      )
    },
    {
      title: 'Coherence',
      dataIndex: 'coherence',
      key: 'coherence',
      render: (value: number) => (
        <Progress 
          percent={Math.round(value * 100)} 
          size="small" 
          strokeColor={value > 0.8 ? '#52c41a' : value > 0.6 ? '#faad14' : '#f5222d'}
        />
      )
    },
    {
      title: 'Frequency (THz)',
      dataIndex: 'frequency',
      key: 'frequency',
      render: (value: number) => `${value.toFixed(2)} THz`
    },
    {
      title: 'Phase',
      dataIndex: 'phase',
      key: 'phase',
      render: (value: number) => `${(value * 360).toFixed(1)}°`
    },
    {
      title: 'Entanglement',
      dataIndex: 'entanglement_strength',
      key: 'entanglement_strength',
      render: (value: number) => (
        <Progress 
          percent={Math.round(value * 100)} 
          size="small" 
          strokeColor="#722ed1"
          format={(percent) => `${percent}%`}
        />
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: TimeCrystal) => (
        <Space>
          <Button 
            size="small" 
            icon={<ReloadOutlined />}
            onClick={() => resetCrystal(record.id)}
          >
            Reset
          </Button>
          <Button 
            size="small" 
            icon={<SettingOutlined />}
            onClick={() => setSelectedCrystal(record.id)}
          >
            Configure
          </Button>
        </Space>
      )
    }
  ];

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#fff'
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#fff' },
        grid: { color: '#434343' }
      },
      y: {
        ticks: { color: '#fff' },
        grid: { color: '#434343' }
      }
    }
  };

  return (
    <div style={{ padding: '24px', background: '#000', minHeight: '100vh' }}>
      <Row gutter={[24, 24]}>
        {/* Header Controls */}
        <Col span={24}>
          <Card title="Time Crystal Control Center" className="time-crystal-glow">
            <Row gutter={[16, 16]} align="middle">
              <Col flex="auto">
                <Space size="large">
                  <Button
                    type="primary"
                    size="large"
                    icon={isSimulating ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                    onClick={isSimulating ? stopSimulation : startSimulation}
                  >
                    {isSimulating ? 'Stop Simulation' : 'Start Simulation'}
                  </Button>
                  <Button
                    icon={<SyncOutlined />}
                    onClick={synchronizeCrystals}
                  >
                    Synchronize Network
                  </Button>
                  <Button
                    icon={<ExperimentOutlined />}
                    onClick={loadMetrics}
                  >
                    Refresh Metrics
                  </Button>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Statistic
                    title="Active Crystals"
                    value={crystals.filter(c => c.state === 'active').length}
                    prefix={<ThunderboltOutlined style={{ color: '#52c41a' }} />}
                  />
                  <Statistic
                    title="Sync Events"
                    value={metrics?.synchronization_events || 0}
                    prefix={<FireOutlined style={{ color: '#faad14' }} />}
                  />
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Status Overview */}
        <Col span={24}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Network Coherence"
                  value={crystals.length > 0 ? (crystals.reduce((sum, c) => sum + c.coherence, 0) / crystals.length * 100) : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Avg Frequency"
                  value={crystals.length > 0 ? (crystals.reduce((sum, c) => sum + c.frequency, 0) / crystals.length) : 0}
                  precision={2}
                  suffix="THz"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Phase Stability"
                  value={85.7}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Entanglement"
                  value={crystals.length > 0 ? (crystals.reduce((sum, c) => sum + c.entanglement_strength, 0) / crystals.length * 100) : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#eb2f96' }}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Main Content Tabs */}
        <Col span={24}>
          <Card>
            <Tabs
              defaultActiveKey="crystals"
              items={[
                {
                  key: 'crystals',
                  label: 'Crystal Network',
                  children: (
                    <Table
                      columns={crystalColumns}
                      dataSource={crystals}
                      rowKey="id"
                      loading={isLoading}
                      pagination={false}
                    />
                  )
                },
                {
                  key: 'metrics',
                  label: 'Performance Metrics',
                  children: (
                    <Row gutter={[24, 24]}>
                      <Col span={12}>
                        <Card title="Coherence Evolution" size="small">
                          <div style={{ height: '300px' }}>
                            <Line data={coherenceChartData} options={chartOptions} />
                          </div>
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="Phase Dynamics" size="small">
                          <div style={{ height: '300px' }}>
                            <Line data={phaseChartData} options={chartOptions} />
                          </div>
                        </Card>
                      </Col>
                    </Row>
                  )
                },
                {
                  key: 'simulation',
                  label: 'Simulation Parameters',
                  children: (
                    <Row gutter={[24, 24]}>
                      <Col span={12}>
                        <Card title="Crystal Network Configuration" size="small">
                          <Space direction="vertical" style={{ width: '100%' }}>
                            <div>
                              <label>Number of Crystals:</label>
                              <InputNumber
                                min={1}
                                max={32}
                                value={simulationParams.num_crystals}
                                onChange={(value) => setSimulationParams(prev => ({ ...prev, num_crystals: value || 8 }))}
                                style={{ width: '100%', marginTop: 8 }}
                              />
                            </div>
                            <div>
                              <label>Coupling Strength:</label>
                              <Slider
                                min={0}
                                max={1}
                                step={0.01}
                                value={simulationParams.coupling_strength}
                                onChange={(value) => setSimulationParams(prev => ({ ...prev, coupling_strength: value }))}
                                tooltip={{ formatter: (value) => `${value?.toFixed(2)}` }}
                              />
                            </div>
                            <div>
                              <label>Temperature (K):</label>
                              <Slider
                                min={0}
                                max={1}
                                step={0.01}
                                value={simulationParams.temperature}
                                onChange={(value) => setSimulationParams(prev => ({ ...prev, temperature: value }))}
                                tooltip={{ formatter: (value) => `${value?.toFixed(2)} K` }}
                              />
                            </div>
                            <div>
                              <label>External Field:</label>
                              <Slider
                                min={0}
                                max={1}
                                step={0.01}
                                value={simulationParams.external_field}
                                onChange={(value) => setSimulationParams(prev => ({ ...prev, external_field: value }))}
                                tooltip={{ formatter: (value) => `${value?.toFixed(2)}` }}
                              />
                            </div>
                          </Space>
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="System Timeline" size="small">
                          <Timeline
                            items={[
                              {
                                color: 'green',
                                children: 'Time crystal network initialized'
                              },
                              {
                                color: 'blue',
                                children: 'Phase synchronization achieved'
                              },
                              {
                                color: 'orange',
                                children: 'Coherence optimization in progress'
                              },
                              {
                                color: 'purple',
                                children: 'Entanglement network stabilized'
                              }
                            ]}
                          />
                        </Card>
                      </Col>
                    </Row>
                  )
                }
              ]}
            />
          </Card>
        </Col>

        {/* Status Alerts */}
        {isSimulating && (
          <Col span={24}>
            <Alert
              message="Time Crystal Simulation Active"
              description="Multi-dimensional time crystal network simulation is currently running. Monitor coherence levels and phase synchronization."
              type="info"
              showIcon
              action={
                <Button size="small" onClick={stopSimulation}>
                  Stop
                </Button>
              }
            />
          </Col>
        )}
      </Row>
    </div>
  );
};

export default TimeCrystals;