import React, { useState, useEffect, useRef } from 'react';
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
  notification,
  Select,
  Divider
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  BrainOutlined,
  FireOutlined,
  NodeIndexOutlined,
  ClusterOutlined
} from '@ant-design/icons';
import { Line, Scatter, Bar } from 'react-chartjs-2';
import { useApi } from '../../contexts/ApiContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface Neuron {
  id: string;
  type: 'LIF' | 'Izhikevich' | 'AdEx' | 'QuantumLIF';
  state: 'active' | 'inactive' | 'refractory';
  membrane_potential: number;
  spike_count: number;
  quantum_coherence?: number;
  synaptic_weights: number[];
  last_spike_time: number;
}

interface NetworkMetrics {
  total_spikes: number;
  average_firing_rate: number;
  network_synchrony: number;
  synaptic_plasticity: number;
  quantum_entanglement?: number;
  spike_train_history: number[][];
  membrane_potential_history: number[][];
}

interface SpikeEvent {
  neuron_id: string;
  timestamp: number;
  amplitude: number;
  post_synaptic_targets: string[];
}

const Neuromorphic: React.FC = () => {
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [metrics, setMetrics] = useState<NetworkMetrics | null>(null);
  const [spikeEvents, setSpikeEvents] = useState<SpikeEvent[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedNeuron, setSelectedNeuron] = useState<string | null>(null);
  const [networkParams, setNetworkParams] = useState({
    num_neurons: 10000,
    connectivity: 0.1,
    learning_rate: 0.01,
    quantum_coupling: 0.5,
    plasticity_enabled: true,
    noise_level: 0.1
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { api } = useApi();
  const { subscribe, sendMessage } = useWebSocket();

  useEffect(() => {
    loadNeurons();
    loadMetrics();
    loadSpikeEvents();

    // Subscribe to real-time updates
    const unsubscribeNeuron = subscribe('neuron_update', (data) => {
      if (data.neuron_id) {
        setNeurons(prev => prev.map(neuron => 
          neuron.id === data.neuron_id ? { ...neuron, ...data } : neuron
        ));
      }
    });

    const unsubscribeSpike = subscribe('spike_event', (data) => {
      setSpikeEvents(prev => [data, ...prev.slice(0, 999)]); // Keep last 1000 events
    });

    const unsubscribeMetrics = subscribe('neuromorphic_metrics', (data) => {
      setMetrics(data);
    });

    return () => {
      unsubscribeNeuron();
      unsubscribeSpike();
      unsubscribeMetrics();
    };
  }, [subscribe]);

  useEffect(() => {
    if (canvasRef.current && spikeEvents.length > 0) {
      drawSpikeVisualization();
    }
  }, [spikeEvents]);

  const loadNeurons = async () => {
    try {
      const response = await api.get('/neuromorphic/neurons');
      setNeurons(response.data.neurons || []);
    } catch (error) {
      console.error('Failed to load neurons:', error);
      // Mock data for development
      const mockNeurons = Array.from({ length: 20 }, (_, i) => ({
        id: `neuron_${i.toString().padStart(3, '0')}`,
        type: ['LIF', 'Izhikevich', 'AdEx', 'QuantumLIF'][Math.floor(Math.random() * 4)] as any,
        state: ['active', 'inactive', 'refractory'][Math.floor(Math.random() * 3)] as any,
        membrane_potential: -70 + Math.random() * 30,
        spike_count: Math.floor(Math.random() * 100),
        quantum_coherence: Math.random(),
        synaptic_weights: Array.from({ length: 10 }, () => Math.random() * 2 - 1),
        last_spike_time: Date.now() - Math.random() * 10000
      }));
      setNeurons(mockNeurons);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await api.get('/neuromorphic/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
      // Mock data for development
      setMetrics({
        total_spikes: 15420,
        average_firing_rate: 23.5,
        network_synchrony: 0.67,
        synaptic_plasticity: 0.82,
        quantum_entanglement: 0.45,
        spike_train_history: Array.from({ length: 100 }, () => 
          Array.from({ length: 20 }, () => Math.random() > 0.8 ? 1 : 0)
        ),
        membrane_potential_history: Array.from({ length: 100 }, () => 
          Array.from({ length: 20 }, () => -70 + Math.random() * 30)
        )
      });
    }
  };

  const loadSpikeEvents = async () => {
    try {
      const response = await api.get('/neuromorphic/spikes');
      setSpikeEvents(response.data.events || []);
    } catch (error) {
      console.error('Failed to load spike events:', error);
      // Mock data for development
      const mockEvents = Array.from({ length: 50 }, (_, i) => ({
        neuron_id: `neuron_${Math.floor(Math.random() * 20).toString().padStart(3, '0')}`,
        timestamp: Date.now() - i * 100,
        amplitude: Math.random() * 50 + 10,
        post_synaptic_targets: Array.from({ length: Math.floor(Math.random() * 5) + 1 }, () => 
          `neuron_${Math.floor(Math.random() * 20).toString().padStart(3, '0')}`
        )
      }));
      setSpikeEvents(mockEvents);
    }
  };

  const startSimulation = async () => {
    try {
      setIsSimulating(true);
      await api.post('/neuromorphic/start', networkParams);
      notification.success({
        message: 'Neuromorphic Simulation Started',
        description: 'Spiking neural network simulation is now running with quantum-inspired dynamics.'
      });
    } catch (error) {
      console.error('Failed to start simulation:', error);
      notification.error({
        message: 'Simulation Error',
        description: 'Failed to start neuromorphic simulation.'
      });
    }
  };

  const stopSimulation = async () => {
    try {
      await api.post('/neuromorphic/stop');
      setIsSimulating(false);
      notification.info({
        message: 'Simulation Stopped',
        description: 'Neuromorphic simulation has been stopped.'
      });
    } catch (error) {
      console.error('Failed to stop simulation:', error);
    }
  };

  const injectSpikePattern = async () => {
    const selectedNeurons = neurons.slice(0, 5).map(n => n.id);
    const pattern = [1.0, 0.8, 0.6, 0.4, 0.2];
    
    try {
      await api.post('/neuromorphic/inject-spikes', {
        neuron_ids: selectedNeurons,
        pattern,
        amplitude: 1.0
      });
      notification.success({
        message: 'Spike Pattern Injected',
        description: `Injected spike pattern to ${selectedNeurons.length} neurons.`
      });
    } catch (error) {
      console.error('Failed to inject spike pattern:', error);
    }
  };

  const drawSpikeVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1f1f1f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw spike raster plot
    const recentSpikes = spikeEvents.slice(0, 200);
    const neuronIds = [...new Set(recentSpikes.map(e => e.neuron_id))];
    const maxTime = Math.max(...recentSpikes.map(e => e.timestamp));
    const minTime = maxTime - 5000; // Last 5 seconds

    recentSpikes.forEach((spike, index) => {
      if (spike.timestamp < minTime) return;

      const neuronIndex = neuronIds.indexOf(spike.neuron_id);
      const x = ((spike.timestamp - minTime) / 5000) * canvas.width;
      const y = (neuronIndex / neuronIds.length) * canvas.height;

      // Color based on amplitude
      const intensity = Math.min(spike.amplitude / 50, 1);
      ctx.fillStyle = `rgba(24, 144, 255, ${intensity})`;
      ctx.fillRect(x, y, 2, 3);
    });

    // Draw grid
    ctx.strokeStyle = '#434343';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 10; i++) {
      const x = (i / 10) * canvas.width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
  };

  const firingRateChartData = {
    labels: Array.from({ length: 50 }, (_, i) => i),
    datasets: [
      {
        label: 'Average Firing Rate (Hz)',
        data: Array.from({ length: 50 }, () => 20 + Math.random() * 10),
        borderColor: '#faad14',
        backgroundColor: 'rgba(250, 173, 20, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

  const membranePotentialData = {
    labels: Array.from({ length: 100 }, (_, i) => i),
    datasets: [
      {
        label: 'Membrane Potential (mV)',
        data: metrics?.membrane_potential_history?.[0] || [],
        borderColor: '#1890ff',
        backgroundColor: 'rgba(24, 144, 255, 0.1)',
        fill: false,
        tension: 0.3
      }
    ]
  };

  const neuronColumns = [
    {
      title: 'Neuron ID',
      dataIndex: 'id',
      key: 'id',
      render: (text: string, record: Neuron) => (
        <Space>
          <span style={{ fontWeight: 500 }}>{text}</span>
          <Tag color={record.state === 'active' ? 'green' : record.state === 'refractory' ? 'orange' : 'default'}>
            {record.state.toUpperCase()}
          </Tag>
        </Space>
      )
    },
    {
      title: 'Type',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'QuantumLIF' ? 'purple' : 'blue'}>{type}</Tag>
      )
    },
    {
      title: 'Membrane Potential',
      dataIndex: 'membrane_potential',
      key: 'membrane_potential',
      render: (value: number) => `${value.toFixed(1)} mV`
    },
    {
      title: 'Spike Count',
      dataIndex: 'spike_count',
      key: 'spike_count'
    },
    {
      title: 'Quantum Coherence',
      dataIndex: 'quantum_coherence',
      key: 'quantum_coherence',
      render: (value?: number) => value ? (
        <Progress 
          percent={Math.round(value * 100)} 
          size="small" 
          strokeColor="#722ed1"
        />
      ) : 'N/A'
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: Neuron) => (
        <Space>
          <Button 
            size="small" 
            icon={<SettingOutlined />}
            onClick={() => setSelectedNeuron(record.id)}
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
          <Card title="Neuromorphic Computing Control Center" className="neuromorphic-glow">
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
                    icon={<ThunderboltOutlined />}
                    onClick={injectSpikePattern}
                  >
                    Inject Spike Pattern
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={loadMetrics}
                  >
                    Refresh Metrics
                  </Button>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Statistic
                    title="Active Neurons"
                    value={neurons.filter(n => n.state === 'active').length}
                    prefix={<BrainOutlined style={{ color: '#faad14' }} />}
                  />
                  <Statistic
                    title="Total Spikes"
                    value={metrics?.total_spikes || 0}
                    prefix={<FireOutlined style={{ color: '#f5222d' }} />}
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
                  title="Firing Rate"
                  value={metrics?.average_firing_rate || 0}
                  precision={1}
                  suffix="Hz"
                  valueStyle={{ color: '#faad14' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Network Synchrony"
                  value={metrics?.network_synchrony ? metrics.network_synchrony * 100 : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Synaptic Plasticity"
                  value={metrics?.synaptic_plasticity ? metrics.synaptic_plasticity * 100 : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Quantum Entanglement"
                  value={metrics?.quantum_entanglement ? metrics.quantum_entanglement * 100 : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Spike Visualization */}
        <Col span={24}>
          <Card title="Real-time Spike Activity" className="neuromorphic-spikes">
            <canvas
              ref={canvasRef}
              width={800}
              height={300}
              style={{ 
                width: '100%', 
                height: '300px', 
                border: '1px solid #434343',
                borderRadius: '4px',
                background: '#1f1f1f'
              }}
            />
            <div style={{ marginTop: 16, color: '#8c8c8c' }}>
              Real-time spike raster plot showing neural activity across the network. Each dot represents a spike event.
            </div>
          </Card>
        </Col>

        {/* Main Content Tabs */}
        <Col span={24}>
          <Card>
            <Tabs
              defaultActiveKey="neurons"
              items={[
                {
                  key: 'neurons',
                  label: 'Neural Network',
                  children: (
                    <Table
                      columns={neuronColumns}
                      dataSource={neurons.slice(0, 20)} // Show first 20 for performance
                      rowKey="id"
                      loading={isLoading}
                      pagination={{ pageSize: 10 }}
                    />
                  )
                },
                {
                  key: 'metrics',
                  label: 'Performance Metrics',
                  children: (
                    <Row gutter={[24, 24]}>
                      <Col span={12}>
                        <Card title="Firing Rate Evolution" size="small">
                          <div style={{ height: '300px' }}>
                            <Line data={firingRateChartData} options={chartOptions} />
                          </div>
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="Membrane Potential" size="small">
                          <div style={{ height: '300px' }}>
                            <Line data={membranePotentialData} options={chartOptions} />
                          </div>
                        </Card>
                      </Col>
                    </Row>
                  )
                },
                {
                  key: 'configuration',
                  label: 'Network Configuration',
                  children: (
                    <Row gutter={[24, 24]}>
                      <Col span={12}>
                        <Card title="Network Parameters" size="small">
                          <Space direction="vertical" style={{ width: '100%' }}>
                            <div>
                              <label>Number of Neurons:</label>
                              <InputNumber
                                min={100}
                                max={100000}
                                value={networkParams.num_neurons}
                                onChange={(value) => setNetworkParams(prev => ({ ...prev, num_neurons: value || 10000 }))}
                                style={{ width: '100%', marginTop: 8 }}
                              />
                            </div>
                            <div>
                              <label>Connectivity:</label>
                              <Slider
                                min={0}
                                max={1}
                                step={0.01}
                                value={networkParams.connectivity}
                                onChange={(value) => setNetworkParams(prev => ({ ...prev, connectivity: value }))}
                                tooltip={{ formatter: (value) => `${(value! * 100).toFixed(0)}%` }}
                              />
                            </div>
                            <div>
                              <label>Learning Rate:</label>
                              <Slider
                                min={0}
                                max={0.1}
                                step={0.001}
                                value={networkParams.learning_rate}
                                onChange={(value) => setNetworkParams(prev => ({ ...prev, learning_rate: value }))}
                                tooltip={{ formatter: (value) => `${value?.toFixed(3)}` }}
                              />
                            </div>
                            <div>
                              <label>Quantum Coupling:</label>
                              <Slider
                                min={0}
                                max={1}
                                step={0.01}
                                value={networkParams.quantum_coupling}
                                onChange={(value) => setNetworkParams(prev => ({ ...prev, quantum_coupling: value }))}
                                tooltip={{ formatter: (value) => `${value?.toFixed(2)}` }}
                              />
                            </div>
                            <div>
                              <label>Synaptic Plasticity:</label>
                              <Switch
                                checked={networkParams.plasticity_enabled}
                                onChange={(checked) => setNetworkParams(prev => ({ ...prev, plasticity_enabled: checked }))}
                              />
                            </div>
                          </Space>
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="Recent Spike Events" size="small">
                          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                            <Timeline
                              size="small"
                              items={spikeEvents.slice(0, 10).map((event, index) => ({
                                color: 'blue',
                                children: (
                                  <div key={index}>
                                    <strong>{event.neuron_id}</strong> fired
                                    <br />
                                    <small>Amplitude: {event.amplitude.toFixed(1)}</small>
                                  </div>
                                )
                              }))}
                            />
                          </div>
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
              message="Neuromorphic Simulation Active"
              description="Spiking neural network with quantum-inspired dynamics is currently running. Monitor spike patterns and network synchronization."
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

export default Neuromorphic;