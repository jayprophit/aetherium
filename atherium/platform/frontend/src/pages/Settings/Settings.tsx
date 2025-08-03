import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Form,
  Input,
  Button,
  Switch,
  Select,
  Slider,
  Divider,
  Space,
  notification,
  Tabs,
  Alert,
  InputNumber,
  Upload,
  Progress,
  Typography
} from 'antd';
import {
  SaveOutlined,
  ReloadOutlined,
  UploadOutlined,
  DownloadOutlined,
  SecurityScanOutlined,
  SettingOutlined,
  DatabaseOutlined,
  CloudOutlined,
  LockOutlined
} from '@ant-design/icons';
import { useApi } from '../../contexts/ApiContext';
import { useAuth } from '../../contexts/AuthContext';

const { Title, Text } = Typography;

interface SystemSettings {
  general: {
    system_name: string;
    debug_mode: boolean;
    log_level: string;
    auto_backup: boolean;
    backup_interval: number;
  };
  quantum: {
    default_qubits: number;
    simulation_precision: number;
    error_correction: boolean;
    quantum_noise_level: number;
  };
  ai_ml: {
    model_cache_size: number;
    training_batch_size: number;
    learning_rate: number;
    gpu_acceleration: boolean;
  };
  security: {
    session_timeout: number;
    password_policy: string;
    two_factor_required: boolean;
    api_rate_limit: number;
  };
  database: {
    connection_pool_size: number;
    query_timeout: number;
    auto_vacuum: boolean;
    backup_retention_days: number;
  };
}

const Settings: React.FC = () => {
  const [form] = Form.useForm();
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('general');
  
  const { api } = useApi();
  const { user, hasPermission } = useAuth();

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setIsLoading(true);
      const response = await api.get('/settings');
      setSettings(response.data.settings);
      form.setFieldsValue(response.data.settings);
    } catch (error) {
      console.error('Failed to load settings:', error);
      // Mock settings for development
      const mockSettings: SystemSettings = {
        general: {
          system_name: 'Quantum AI Platform',
          debug_mode: false,
          log_level: 'INFO',
          auto_backup: true,
          backup_interval: 24
        },
        quantum: {
          default_qubits: 32,
          simulation_precision: 0.001,
          error_correction: true,
          quantum_noise_level: 0.1
        },
        ai_ml: {
          model_cache_size: 2048,
          training_batch_size: 32,
          learning_rate: 0.001,
          gpu_acceleration: true
        },
        security: {
          session_timeout: 3600,
          password_policy: 'strong',
          two_factor_required: false,
          api_rate_limit: 1000
        },
        database: {
          connection_pool_size: 20,
          query_timeout: 30,
          auto_vacuum: true,
          backup_retention_days: 30
        }
      };
      setSettings(mockSettings);
      form.setFieldsValue(mockSettings);
    } finally {
      setIsLoading(false);
    }
  };

  const saveSettings = async (values: SystemSettings) => {
    if (!hasPermission('admin.settings.write')) {
      notification.error({
        message: 'Permission Denied',
        description: 'You do not have permission to modify system settings.'
      });
      return;
    }

    try {
      setIsSaving(true);
      await api.post('/settings', { settings: values });
      setSettings(values);
      notification.success({
        message: 'Settings Saved',
        description: 'System settings have been successfully updated.'
      });
    } catch (error) {
      console.error('Failed to save settings:', error);
      notification.error({
        message: 'Save Failed',
        description: 'Failed to save system settings.'
      });
    } finally {
      setIsSaving(false);
    }
  };

  const resetToDefaults = async () => {
    try {
      await api.post('/settings/reset');
      await loadSettings();
      notification.success({
        message: 'Settings Reset',
        description: 'System settings have been reset to defaults.'
      });
    } catch (error) {
      console.error('Failed to reset settings:', error);
    }
  };

  const exportSettings = async () => {
    try {
      const response = await api.get('/settings/export');
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `quantum-ai-settings-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export settings:', error);
    }
  };

  return (
    <div style={{ padding: '24px', background: '#000', minHeight: '100vh' }}>
      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card 
            title={
              <Space>
                <SettingOutlined />
                <span>System Settings</span>
              </Space>
            }
            extra={
              <Space>
                <Button icon={<DownloadOutlined />} onClick={exportSettings}>
                  Export
                </Button>
                <Button icon={<ReloadOutlined />} onClick={loadSettings}>
                  Refresh
                </Button>
                <Button 
                  type="primary" 
                  icon={<SaveOutlined />} 
                  loading={isSaving}
                  onClick={() => form.submit()}
                >
                  Save Changes
                </Button>
              </Space>
            }
          >
            {!hasPermission('admin.settings.read') ? (
              <Alert
                message="Access Denied"
                description="You do not have permission to view system settings."
                type="error"
                showIcon
              />
            ) : (
              <Form
                form={form}
                layout="vertical"
                onFinish={saveSettings}
                initialValues={settings}
              >
                <Tabs
                  activeKey={activeTab}
                  onChange={setActiveTab}
                  items={[
                    {
                      key: 'general',
                      label: 'General',
                      children: (
                        <Row gutter={[24, 24]}>
                          <Col span={12}>
                            <Card title="System Configuration" size="small">
                              <Form.Item
                                name={['general', 'system_name']}
                                label="System Name"
                                rules={[{ required: true }]}
                              >
                                <Input />
                              </Form.Item>
                              <Form.Item
                                name={['general', 'debug_mode']}
                                label="Debug Mode"
                                valuePropName="checked"
                              >
                                <Switch />
                              </Form.Item>
                              <Form.Item
                                name={['general', 'log_level']}
                                label="Log Level"
                              >
                                <Select>
                                  <Select.Option value="DEBUG">DEBUG</Select.Option>
                                  <Select.Option value="INFO">INFO</Select.Option>
                                  <Select.Option value="WARNING">WARNING</Select.Option>
                                  <Select.Option value="ERROR">ERROR</Select.Option>
                                </Select>
                              </Form.Item>
                            </Card>
                          </Col>
                          <Col span={12}>
                            <Card title="Backup Configuration" size="small">
                              <Form.Item
                                name={['general', 'auto_backup']}
                                label="Automatic Backup"
                                valuePropName="checked"
                              >
                                <Switch />
                              </Form.Item>
                              <Form.Item
                                name={['general', 'backup_interval']}
                                label="Backup Interval (hours)"
                              >
                                <InputNumber min={1} max={168} />
                              </Form.Item>
                            </Card>
                          </Col>
                        </Row>
                      )
                    },
                    {
                      key: 'quantum',
                      label: 'Quantum Computing',
                      children: (
                        <Row gutter={[24, 24]}>
                          <Col span={12}>
                            <Card title="Quantum Simulation" size="small">
                              <Form.Item
                                name={['quantum', 'default_qubits']}
                                label="Default Qubits"
                              >
                                <Slider min={1} max={64} marks={{1: '1', 16: '16', 32: '32', 64: '64'}} />
                              </Form.Item>
                              <Form.Item
                                name={['quantum', 'simulation_precision']}
                                label="Simulation Precision"
                              >
                                <InputNumber min={0.0001} max={0.1} step={0.0001} />
                              </Form.Item>
                              <Form.Item
                                name={['quantum', 'error_correction']}
                                label="Quantum Error Correction"
                                valuePropName="checked"
                              >
                                <Switch />
                              </Form.Item>
                            </Card>
                          </Col>
                          <Col span={12}>
                            <Card title="Noise Model" size="small">
                              <Form.Item
                                name={['quantum', 'quantum_noise_level']}
                                label="Noise Level"
                              >
                                <Slider min={0} max={1} step={0.01} />
                              </Form.Item>
                              <Alert
                                message="Quantum Settings"
                                description="These settings affect the behavior of quantum circuit simulations."
                                type="info"
                                size="small"
                              />
                            </Card>
                          </Col>
                        </Row>
                      )
                    },
                    {
                      key: 'ai_ml',
                      label: 'AI/ML',
                      children: (
                        <Row gutter={[24, 24]}>
                          <Col span={12}>
                            <Card title="Model Configuration" size="small">
                              <Form.Item
                                name={['ai_ml', 'model_cache_size']}
                                label="Model Cache Size (MB)"
                              >
                                <InputNumber min={512} max={8192} />
                              </Form.Item>
                              <Form.Item
                                name={['ai_ml', 'training_batch_size']}
                                label="Training Batch Size"
                              >
                                <InputNumber min={1} max={256} />
                              </Form.Item>
                              <Form.Item
                                name={['ai_ml', 'learning_rate']}
                                label="Default Learning Rate"
                              >
                                <InputNumber min={0.0001} max={0.1} step={0.0001} />
                              </Form.Item>
                            </Card>
                          </Col>
                          <Col span={12}>
                            <Card title="Hardware Acceleration" size="small">
                              <Form.Item
                                name={['ai_ml', 'gpu_acceleration']}
                                label="GPU Acceleration"
                                valuePropName="checked"
                              >
                                <Switch />
                              </Form.Item>
                              <Alert
                                message="AI/ML Settings"
                                description="Configure AI model training and inference parameters."
                                type="info"
                                size="small"
                              />
                            </Card>
                          </Col>
                        </Row>
                      )
                    },
                    {
                      key: 'security',
                      label: 'Security',
                      children: (
                        <Row gutter={[24, 24]}>
                          <Col span={12}>
                            <Card title="Authentication" size="small">
                              <Form.Item
                                name={['security', 'session_timeout']}
                                label="Session Timeout (seconds)"
                              >
                                <InputNumber min={300} max={86400} />
                              </Form.Item>
                              <Form.Item
                                name={['security', 'password_policy']}
                                label="Password Policy"
                              >
                                <Select>
                                  <Select.Option value="basic">Basic</Select.Option>
                                  <Select.Option value="strong">Strong</Select.Option>
                                  <Select.Option value="enterprise">Enterprise</Select.Option>
                                </Select>
                              </Form.Item>
                              <Form.Item
                                name={['security', 'two_factor_required']}
                                label="Require Two-Factor Authentication"
                                valuePropName="checked"
                              >
                                <Switch />
                              </Form.Item>
                            </Card>
                          </Col>
                          <Col span={12}>
                            <Card title="API Security" size="small">
                              <Form.Item
                                name={['security', 'api_rate_limit']}
                                label="API Rate Limit (requests/hour)"
                              >
                                <InputNumber min={100} max={10000} />
                              </Form.Item>
                              <Alert
                                message="Security Settings"
                                description="Configure authentication and access control policies."
                                type="warning"
                                size="small"
                              />
                            </Card>
                          </Col>
                        </Row>
                      )
                    },
                    {
                      key: 'database',
                      label: 'Database',
                      children: (
                        <Row gutter={[24, 24]}>
                          <Col span={12}>
                            <Card title="Connection Settings" size="small">
                              <Form.Item
                                name={['database', 'connection_pool_size']}
                                label="Connection Pool Size"
                              >
                                <InputNumber min={5} max={100} />
                              </Form.Item>
                              <Form.Item
                                name={['database', 'query_timeout']}
                                label="Query Timeout (seconds)"
                              >
                                <InputNumber min={5} max={300} />
                              </Form.Item>
                              <Form.Item
                                name={['database', 'auto_vacuum']}
                                label="Auto Vacuum"
                                valuePropName="checked"
                              >
                                <Switch />
                              </Form.Item>
                            </Card>
                          </Col>
                          <Col span={12}>
                            <Card title="Backup Settings" size="small">
                              <Form.Item
                                name={['database', 'backup_retention_days']}
                                label="Backup Retention (days)"
                              >
                                <InputNumber min={7} max={365} />
                              </Form.Item>
                              <Alert
                                message="Database Settings"
                                description="Configure database performance and backup parameters."
                                type="info"
                                size="small"
                              />
                            </Card>
                          </Col>
                        </Row>
                      )
                    }
                  ]}
                />
              </Form>
            )}
          </Card>
        </Col>

        {/* System Status */}
        <Col span={24}>
          <Card title="System Status" size="small">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Card>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>CPU Usage</Text>
                    <Progress percent={45} strokeColor="#1890ff" />
                  </Space>
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>Memory Usage</Text>
                    <Progress percent={62} strokeColor="#52c41a" />
                  </Space>
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>Disk Usage</Text>
                    <Progress percent={78} strokeColor="#faad14" />
                  </Space>
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>Network I/O</Text>
                    <Progress percent={23} strokeColor="#722ed1" />
                  </Space>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Danger Zone */}
        {hasPermission('admin.system.manage') && (
          <Col span={24}>
            <Card title="Danger Zone" size="small">
              <Alert
                message="Destructive Actions"
                description="These actions are irreversible and may cause system downtime."
                type="error"
                showIcon
                style={{ marginBottom: 16 }}
              />
              <Space>
                <Button 
                  danger 
                  onClick={resetToDefaults}
                  disabled={!hasPermission('admin.settings.write')}
                >
                  Reset to Defaults
                </Button>
                <Button 
                  danger 
                  disabled={!hasPermission('admin.system.manage')}
                >
                  Clear All Caches
                </Button>
                <Button 
                  danger 
                  disabled={!hasPermission('admin.system.manage')}
                >
                  Restart System
                </Button>
              </Space>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export default Settings;