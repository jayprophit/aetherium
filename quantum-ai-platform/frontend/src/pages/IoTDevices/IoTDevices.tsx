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
  Switch,
  Tabs,
  Timeline,
  notification,
  Modal,
  Form,
  Input,
  Select,
  Badge,
  Descriptions,
  Tooltip
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  WifiOutlined,
  DisconnectOutlined,
  PlusOutlined,
  DeleteOutlined,
  SyncOutlined,
  ThunderboltOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { Line, Doughnut } from 'react-chartjs-2';
import { useApi } from '../../contexts/ApiContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

interface IoTDevice {
  id: string;
  name: string;
  type: string;
  status: 'online' | 'offline' | 'warning' | 'error';
  last_seen: string;
  quantum_sync_enabled: boolean;
  metrics: {
    cpu_usage?: number;
    memory_usage?: number;
    temperature?: number;
    battery_level?: number;
    signal_strength?: number;
  };
  location?: {
    latitude: number;
    longitude: number;
    altitude?: number;
  };
  firmware_version: string;
  ip_address: string;
}

interface DeviceMetrics {
  total_devices: number;
  online_devices: number;
  data_rate: number;
  quantum_sync_rate: number;
  network_health: number;
  recent_events: any[];
  performance_history: number[];
}

const IoTDevices: React.FC = () => {
  const [devices, setDevices] = useState<IoTDevice[]>([]);
  const [metrics, setMetrics] = useState<DeviceMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedDevice, setSelectedDevice] = useState<IoTDevice | null>(null);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isDetailModalVisible, setIsDetailModalVisible] = useState(false);
  const [form] = Form.useForm();

  const { api } = useApi();
  const { subscribe, sendMessage } = useWebSocket();

  useEffect(() => {
    loadDevices();
    loadMetrics();

    // Subscribe to real-time updates
    const unsubscribeDevice = subscribe('iot_device_update', (data) => {
      if (data.device_id) {
        setDevices(prev => prev.map(device => 
          device.id === data.device_id ? { ...device, ...data } : device
        ));
      }
    });

    const unsubscribeMetrics = subscribe('iot_metrics_update', (data) => {
      setMetrics(data);
    });

    return () => {
      unsubscribeDevice();
      unsubscribeMetrics();
    };
  }, [subscribe]);

  const loadDevices = async () => {
    try {
      const response = await api.get('/iot/devices');
      setDevices(response.data.devices || []);
    } catch (error) {
      console.error('Failed to load IoT devices:', error);
      // Mock data for development
      setDevices([
        {
          id: 'device_001',
          name: 'Quantum Sensor Array 1',
          type: 'quantum_sensor',
          status: 'online',
          last_seen: new Date().toISOString(),
          quantum_sync_enabled: true,
          metrics: {
            cpu_usage: 45,
            memory_usage: 62,
            temperature: 28.5,
            battery_level: 87,
            signal_strength: -45
          },
          location: {
            latitude: 37.7749,
            longitude: -122.4194,
            altitude: 50
          },
          firmware_version: '2.1.3',
          ip_address: '192.168.1.101'
        },
        {
          id: 'device_002',
          name: 'Neural Processing Unit',
          type: 'neural_processor',
          status: 'warning',
          last_seen: new Date(Date.now() - 120000).toISOString(),
          quantum_sync_enabled: false,
          metrics: {
            cpu_usage: 78,
            memory_usage: 85,
            temperature: 45.2,
            signal_strength: -67
          },
          firmware_version: '1.8.1',
          ip_address: '192.168.1.102'
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadMetrics = async () => {
    try {
      const response = await api.get('/iot/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to load IoT metrics:', error);
      // Mock data for development
      setMetrics({
        total_devices: 12,
        online_devices: 9,
        data_rate: 1.2, // MB/s
        quantum_sync_rate: 0.87,
        network_health: 0.92,
        recent_events: [],
        performance_history: Array.from({ length: 50 }, () => 80 + Math.random() * 20)
      });
    }
  };

  const registerDevice = async (deviceData: any) => {
    try {
      const response = await api.post('/iot/register', deviceData);
      notification.success({
        message: 'Device Registered',
        description: `IoT device ${deviceData.name} has been successfully registered.`
      });
      loadDevices();
      setIsAddModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('Failed to register device:', error);
      notification.error({
        message: 'Registration Failed',
        description: 'Failed to register IoT device.'
      });
    }
  };

  const toggleQuantumSync = async (deviceId: string, enabled: boolean) => {
    try {
      await api.post(`/iot/quantum-sync/${deviceId}`, { enabled });
      notification.success({
        message: 'Quantum Sync Updated',
        description: `Quantum synchronization ${enabled ? 'enabled' : 'disabled'} for device.`
      });
      loadDevices();
    } catch (error) {
      console.error('Failed to toggle quantum sync:', error);
    }
  };

  const restartDevice = async (deviceId: string) => {
    try {
      await api.post(`/iot/restart/${deviceId}`);
      notification.success({
        message: 'Device Restart Initiated',
        description: 'Device restart command has been sent.'
      });
    } catch (error) {
      console.error('Failed to restart device:', error);
    }
  };

  const removeDevice = async (deviceId: string) => {
    try {
      await api.delete(`/iot/devices/${deviceId}`);
      notification.success({
        message: 'Device Removed',
        description: 'IoT device has been removed from the network.'
      });
      loadDevices();
    } catch (error) {
      console.error('Failed to remove device:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'error':
        return <DisconnectOutlined style={{ color: '#f5222d' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const deviceColumns = [
    {
      title: 'Device',
      key: 'device',
      render: (_, record: IoTDevice) => (
        <Space>
          {getStatusIcon(record.status)}
          <div>
            <div style={{ fontWeight: 500 }}>{record.name}</div>
            <div style={{ fontSize: '12px', color: '#8c8c8c' }}>{record.type}</div>
          </div>
        </Space>
      )
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={
          status === 'online' ? 'green' : 
          status === 'warning' ? 'orange' : 
          status === 'error' ? 'red' : 'default'
        }>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'Quantum Sync',
      dataIndex: 'quantum_sync_enabled',
      key: 'quantum_sync',
      render: (enabled: boolean, record: IoTDevice) => (
        <Switch
          checked={enabled}
          onChange={(checked) => toggleQuantumSync(record.id, checked)}
          size="small"
        />
      )
    },
    {
      title: 'Performance',
      key: 'performance',
      render: (_, record: IoTDevice) => (
        <Space direction="vertical" size="small" style={{ width: '100%' }}>
          {record.metrics.cpu_usage && (
            <div>
              <span style={{ fontSize: '12px' }}>CPU: </span>
              <Progress 
                percent={record.metrics.cpu_usage} 
                size="small"
                strokeColor={record.metrics.cpu_usage > 80 ? '#f5222d' : '#1890ff'}
                showInfo={false}
                style={{ width: 60 }}
              />
              <span style={{ fontSize: '12px', marginLeft: 8 }}>{record.metrics.cpu_usage}%</span>
            </div>
          )}
          {record.metrics.temperature && (
            <div>
              <span style={{ fontSize: '12px' }}>Temp: {record.metrics.temperature}°C</span>
            </div>
          )}
        </Space>
      )
    },
    {
      title: 'Last Seen',
      dataIndex: 'last_seen',
      key: 'last_seen',
      render: (timestamp: string) => {
        const date = new Date(timestamp);
        return (
          <Tooltip title={date.toLocaleString()}>
            <span style={{ fontSize: '12px' }}>
              {Math.round((Date.now() - date.getTime()) / 60000)}m ago
            </span>
          </Tooltip>
        );
      }
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record: IoTDevice) => (
        <Space>
          <Button 
            size="small" 
            icon={<SettingOutlined />}
            onClick={() => {
              setSelectedDevice(record);
              setIsDetailModalVisible(true);
            }}
          >
            Details
          </Button>
          <Button 
            size="small" 
            icon={<ReloadOutlined />}
            onClick={() => restartDevice(record.id)}
          >
            Restart
          </Button>
          <Button 
            size="small" 
            danger
            icon={<DeleteOutlined />}
            onClick={() => removeDevice(record.id)}
          >
            Remove
          </Button>
        </Space>
      )
    }
  ];

  const networkHealthData = {
    labels: ['Online', 'Warning', 'Offline'],
    datasets: [
      {
        data: [
          devices.filter(d => d.status === 'online').length,
          devices.filter(d => d.status === 'warning').length,
          devices.filter(d => d.status === 'offline' || d.status === 'error').length
        ],
        backgroundColor: ['#52c41a', '#faad14', '#f5222d'],
        borderWidth: 0
      }
    ]
  };

  const performanceChartData = {
    labels: Array.from({ length: 50 }, (_, i) => i),
    datasets: [
      {
        label: 'Network Performance (%)',
        data: metrics?.performance_history || [],
        borderColor: '#1890ff',
        backgroundColor: 'rgba(24, 144, 255, 0.1)',
        fill: true,
        tension: 0.4
      }
    ]
  };

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
          <Card title="IoT Device Management" className="device-status-online">
            <Row gutter={[16, 16]} align="middle">
              <Col flex="auto">
                <Space size="large">
                  <Button
                    type="primary"
                    size="large"
                    icon={<PlusOutlined />}
                    onClick={() => setIsAddModalVisible(true)}
                  >
                    Register Device
                  </Button>
                  <Button
                    icon={<SyncOutlined />}
                    onClick={loadDevices}
                  >
                    Refresh Devices
                  </Button>
                  <Button
                    icon={<WifiOutlined />}
                    onClick={loadMetrics}
                  >
                    Network Scan
                  </Button>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Statistic
                    title="Online Devices"
                    value={metrics?.online_devices || 0}
                    suffix={`/ ${metrics?.total_devices || 0}`}
                    prefix={<WifiOutlined style={{ color: '#52c41a' }} />}
                  />
                  <Statistic
                    title="Data Rate"
                    value={metrics?.data_rate || 0}
                    precision={2}
                    suffix="MB/s"
                    prefix={<ThunderboltOutlined style={{ color: '#1890ff' }} />}
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
                  title="Total Devices"
                  value={devices.length}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Quantum Sync Rate"
                  value={metrics?.quantum_sync_rate ? metrics.quantum_sync_rate * 100 : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Network Health"
                  value={metrics?.network_health ? metrics.network_health * 100 : 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="Avg Response Time"
                  value={42}
                  suffix="ms"
                  valueStyle={{ color: '#faad14' }}
                />
              </Card>
            </Col>
          </Row>
        </Col>

        {/* Main Content */}
        <Col span={18}>
          <Card>
            <Tabs
              defaultActiveKey="devices"
              items={[
                {
                  key: 'devices',
                  label: 'Device List',
                  children: (
                    <Table
                      columns={deviceColumns}
                      dataSource={devices}
                      rowKey="id"
                      loading={isLoading}
                      pagination={{ pageSize: 10 }}
                    />
                  )
                },
                {
                  key: 'performance',
                  label: 'Network Performance',
                  children: (
                    <div style={{ height: '400px' }}>
                      <Line data={performanceChartData} options={chartOptions} />
                    </div>
                  )
                }
              ]}
            />
          </Card>
        </Col>

        {/* Network Health Sidebar */}
        <Col span={6}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Card title="Network Health" size="small">
              <div style={{ height: '200px' }}>
                <Doughnut 
                  data={networkHealthData} 
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                        labels: { color: '#fff' }
                      }
                    }
                  }}
                />
              </div>
            </Card>

            <Card title="Recent Events" size="small">
              <Timeline
                size="small"
                items={[
                  {
                    color: 'green',
                    children: 'Device_001 synchronized'
                  },
                  {
                    color: 'blue',
                    children: 'New device registered'
                  },
                  {
                    color: 'orange',
                    children: 'Device_002 warning: high CPU'
                  },
                  {
                    color: 'red',
                    children: 'Connection timeout'
                  }
                ]}
              />
            </Card>
          </Space>
        </Col>
      </Row>

      {/* Add Device Modal */}
      <Modal
        title="Register New IoT Device"
        open={isAddModalVisible}
        onCancel={() => setIsAddModalVisible(false)}
        onOk={() => form.submit()}
        okText="Register"
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={registerDevice}
        >
          <Form.Item
            name="name"
            label="Device Name"
            rules={[{ required: true, message: 'Please enter device name' }]}
          >
            <Input placeholder="Enter device name" />
          </Form.Item>
          <Form.Item
            name="type"
            label="Device Type"
            rules={[{ required: true, message: 'Please select device type' }]}
          >
            <Select placeholder="Select device type">
              <Select.Option value="quantum_sensor">Quantum Sensor</Select.Option>
              <Select.Option value="neural_processor">Neural Processor</Select.Option>
              <Select.Option value="environmental_monitor">Environmental Monitor</Select.Option>
              <Select.Option value="actuator">Actuator</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="ip_address"
            label="IP Address"
            rules={[{ required: true, message: 'Please enter IP address' }]}
          >
            <Input placeholder="192.168.1.100" />
          </Form.Item>
          <Form.Item
            name="quantum_sync_enabled"
            label="Enable Quantum Synchronization"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* Device Detail Modal */}
      <Modal
        title={selectedDevice?.name}
        open={isDetailModalVisible}
        onCancel={() => setIsDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedDevice && (
          <Descriptions bordered column={2}>
            <Descriptions.Item label="Device ID">{selectedDevice.id}</Descriptions.Item>
            <Descriptions.Item label="Type">{selectedDevice.type}</Descriptions.Item>
            <Descriptions.Item label="Status">
              <Badge 
                status={selectedDevice.status === 'online' ? 'success' : 'error'} 
                text={selectedDevice.status.toUpperCase()} 
              />
            </Descriptions.Item>
            <Descriptions.Item label="IP Address">{selectedDevice.ip_address}</Descriptions.Item>
            <Descriptions.Item label="Firmware">{selectedDevice.firmware_version}</Descriptions.Item>
            <Descriptions.Item label="Last Seen">{new Date(selectedDevice.last_seen).toLocaleString()}</Descriptions.Item>
            {selectedDevice.metrics.cpu_usage && (
              <Descriptions.Item label="CPU Usage">
                <Progress percent={selectedDevice.metrics.cpu_usage} size="small" />
              </Descriptions.Item>
            )}
            {selectedDevice.metrics.memory_usage && (
              <Descriptions.Item label="Memory Usage">
                <Progress percent={selectedDevice.metrics.memory_usage} size="small" />
              </Descriptions.Item>
            )}
            {selectedDevice.metrics.temperature && (
              <Descriptions.Item label="Temperature">{selectedDevice.metrics.temperature}°C</Descriptions.Item>
            )}
            {selectedDevice.metrics.battery_level && (
              <Descriptions.Item label="Battery Level">
                <Progress percent={selectedDevice.metrics.battery_level} size="small" />
              </Descriptions.Item>
            )}
          </Descriptions>
        )}
      </Modal>
    </div>
  );
};

export default IoTDevices;