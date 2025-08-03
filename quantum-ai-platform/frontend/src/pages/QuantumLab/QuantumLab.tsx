import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Button, Select, Input, Table, Statistic, Progress, message } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, ReloadOutlined, ExperimentOutlined } from '@ant-design/icons';
import { Line } from 'react-chartjs-2';

const { Option } = Select;
const { TextArea } = Input;

interface QuantumCircuit {
  id: string;
  name: string;
  qubits: number;
  gates: number;
  depth: number;
  fidelity: number;
  status: 'ready' | 'running' | 'completed' | 'error';
}

interface QuantumState {
  state_vector: number[];
  amplitudes: { [key: string]: number };
  probabilities: { [key: string]: number };
  entanglement: number;
  coherence: number;
}

const QuantumLab: React.FC = () => {
  const [circuits, setCircuits] = useState<QuantumCircuit[]>([]);
  const [selectedCircuit, setSelectedCircuit] = useState<string>('');
  const [quantumState, setQuantumState] = useState<QuantumState | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [circuitCode, setCircuitCode] = useState('');

  useEffect(() => {
    fetchCircuits();
    fetchQuantumState();
  }, []);

  const fetchCircuits = async () => {
    // Mock data - in real implementation, this would call the backend API
    const mockCircuits: QuantumCircuit[] = [
      {
        id: 'qft_8',
        name: 'Quantum Fourier Transform (8 qubits)',
        qubits: 8,
        gates: 32,
        depth: 12,
        fidelity: 0.987,
        status: 'ready'
      },
      {
        id: 'vqe_h2',
        name: 'VQE for H2 molecule',
        qubits: 4,
        gates: 24,
        depth: 8,
        fidelity: 0.952,
        status: 'completed'
      },
      {
        id: 'qaoa_maxcut',
        name: 'QAOA Max-Cut',
        qubits: 6,
        gates: 18,
        depth: 6,
        fidelity: 0.943,
        status: 'ready'
      }
    ];
    setCircuits(mockCircuits);
  };

  const fetchQuantumState = async () => {
    // Mock quantum state data
    const mockState: QuantumState = {
      state_vector: [0.707, 0, 0, 0.707, 0, 0, 0, 0],
      amplitudes: {
        '000': 0.707,
        '001': 0,
        '010': 0,
        '011': 0.707,
        '100': 0,
        '101': 0,
        '110': 0,
        '111': 0
      },
      probabilities: {
        '000': 0.5,
        '001': 0,
        '010': 0,
        '011': 0.5,
        '100': 0,
        '101': 0,
        '110': 0,
        '111': 0
      },
      entanglement: 0.85,
      coherence: 0.92
    };
    setQuantumState(mockState);
  };

  const runCircuit = async () => {
    if (!selectedCircuit) {
      message.warning('Please select a circuit to run');
      return;
    }

    setIsRunning(true);
    
    try {
      // Simulate circuit execution
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Update circuit status
      setCircuits(prev => prev.map(circuit => 
        circuit.id === selectedCircuit 
          ? { ...circuit, status: 'completed' as const }
          : circuit
      ));
      
      // Refresh quantum state
      await fetchQuantumState();
      
      message.success('Circuit executed successfully!');
    } catch (error) {
      message.error('Circuit execution failed');
    } finally {
      setIsRunning(false);
    }
  };

  const createCustomCircuit = async () => {
    if (!circuitCode.trim()) {
      message.warning('Please enter circuit code');
      return;
    }

    try {
      // In real implementation, this would send the code to the backend
      const newCircuit: QuantumCircuit = {
        id: `custom_${Date.now()}`,
        name: 'Custom Circuit',
        qubits: 4,
        gates: 8,
        depth: 4,
        fidelity: 0.95,
        status: 'ready'
      };

      setCircuits(prev => [...prev, newCircuit]);
      setCircuitCode('');
      message.success('Custom circuit created successfully!');
    } catch (error) {
      message.error('Failed to create custom circuit');
    }
  };

  const circuitColumns = [
    {
      title: 'Circuit Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Qubits',
      dataIndex: 'qubits',
      key: 'qubits',
    },
    {
      title: 'Gates',
      dataIndex: 'gates',
      key: 'gates',
    },
    {
      title: 'Depth',
      dataIndex: 'depth',
      key: 'depth',
    },
    {
      title: 'Fidelity',
      dataIndex: 'fidelity',
      key: 'fidelity',
      render: (fidelity: number) => `${(fidelity * 100).toFixed(1)}%`
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <span style={{ 
          color: status === 'completed' ? '#52c41a' : 
                 status === 'running' ? '#1890ff' : 
                 status === 'error' ? '#f5222d' : '#fff' 
        }}>
          {status.toUpperCase()}
        </span>
      )
    },
  ];

  const fidelityData = {
    labels: ['Circuit 1', 'Circuit 2', 'Circuit 3', 'Circuit 4', 'Circuit 5'],
    datasets: [
      {
        label: 'Quantum Fidelity',
        data: [0.987, 0.952, 0.943, 0.968, 0.955],
        borderColor: '#1890ff',
        backgroundColor: 'rgba(24, 144, 255, 0.1)',
        tension: 0.4,
      },
    ],
  };

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <h2 style={{ color: '#fff', marginBottom: 0 }}>Quantum Computing Laboratory</h2>
          <p style={{ color: '#8c8c8c' }}>Design, simulate, and execute quantum circuits</p>
        </Col>
      </Row>

      {/* Quantum State Visualization */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={8}>
          <Card 
            title={<span style={{ color: '#fff' }}>Quantum State</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <Statistic
              title={<span style={{ color: '#fff' }}>Entanglement</span>}
              value={quantumState?.entanglement || 0}
              precision={3}
              valueStyle={{ color: '#1890ff' }}
            />
            <Progress 
              percent={(quantumState?.entanglement || 0) * 100} 
              showInfo={false}
              strokeColor="#1890ff"
            />
            
            <Statistic
              title={<span style={{ color: '#fff', marginTop: 16 }}>Coherence</span>}
              value={quantumState?.coherence || 0}
              precision={3}
              valueStyle={{ color: '#52c41a' }}
              style={{ marginTop: 16 }}
            />
            <Progress 
              percent={(quantumState?.coherence || 0) * 100} 
              showInfo={false}
              strokeColor="#52c41a"
            />
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card 
            title={<span style={{ color: '#fff' }}>Circuit Fidelity Trends</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <Line 
              data={fidelityData} 
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
                    min: 0.9,
                    max: 1.0
                  }
                }
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Circuit Controls */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={12}>
          <Card 
            title={<span style={{ color: '#fff' }}>Circuit Execution</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <div style={{ marginBottom: 16 }}>
              <label style={{ color: '#fff', display: 'block', marginBottom: 8 }}>Select Circuit:</label>
              <Select
                style={{ width: '100%' }}
                placeholder="Choose a quantum circuit"
                value={selectedCircuit}
                onChange={setSelectedCircuit}
              >
                {circuits.map(circuit => (
                  <Option key={circuit.id} value={circuit.id}>
                    {circuit.name} ({circuit.qubits} qubits)
                  </Option>
                ))}
              </Select>
            </div>

            <div style={{ display: 'flex', gap: 12 }}>
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                loading={isRunning}
                onClick={runCircuit}
                disabled={!selectedCircuit}
              >
                {isRunning ? 'Running...' : 'Execute Circuit'}
              </Button>
              
              <Button 
                icon={<ReloadOutlined />}
                onClick={fetchQuantumState}
              >
                Refresh State
              </Button>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card 
            title={<span style={{ color: '#fff' }}>Custom Circuit</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <div style={{ marginBottom: 16 }}>
              <label style={{ color: '#fff', display: 'block', marginBottom: 8 }}>Circuit Code (QASM):</label>
              <TextArea
                rows={4}
                placeholder="Enter your quantum circuit code here..."
                value={circuitCode}
                onChange={(e) => setCircuitCode(e.target.value)}
                style={{ backgroundColor: '#1f1f1f', borderColor: '#434343', color: '#fff' }}
              />
            </div>

            <Button 
              type="primary" 
              icon={<ExperimentOutlined />}
              onClick={createCustomCircuit}
              disabled={!circuitCode.trim()}
            >
              Create Circuit
            </Button>
          </Card>
        </Col>
      </Row>

      {/* Circuit Library */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card 
            title={<span style={{ color: '#fff' }}>Circuit Library</span>}
            style={{ background: '#262626', border: '1px solid #434343' }}
          >
            <Table
              columns={circuitColumns}
              dataSource={circuits}
              rowKey="id"
              pagination={false}
              style={{ 
                backgroundColor: '#1f1f1f'
              }}
              onRow={(record) => ({
                onClick: () => setSelectedCircuit(record.id),
                style: { 
                  cursor: 'pointer',
                  backgroundColor: selectedCircuit === record.id ? '#1890ff20' : 'transparent'
                }
              })}
            />
          </Card>
        </Col>
      </Row>

      {/* Quantum State Details */}
      {quantumState && (
        <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
          <Col xs={24} lg={12}>
            <Card 
              title={<span style={{ color: '#fff' }}>State Amplitudes</span>}
              style={{ background: '#262626', border: '1px solid #434343' }}
            >
              <div style={{ maxHeight: 200, overflow: 'auto' }}>
                {Object.entries(quantumState.amplitudes).map(([state, amplitude]) => (
                  <div key={state} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    marginBottom: 8,
                    color: '#fff'
                  }}>
                    <span>|{state}⟩</span>
                    <span>{amplitude.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </Card>
          </Col>

          <Col xs={24} lg={12}>
            <Card 
              title={<span style={{ color: '#fff' }}>Measurement Probabilities</span>}
              style={{ background: '#262626', border: '1px solid #434343' }}
            >
              <div style={{ maxHeight: 200, overflow: 'auto' }}>
                {Object.entries(quantumState.probabilities).map(([state, probability]) => (
                  <div key={state} style={{ marginBottom: 8 }}>
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      marginBottom: 4,
                      color: '#fff'
                    }}>
                      <span>|{state}⟩</span>
                      <span>{(probability * 100).toFixed(1)}%</span>
                    </div>
                    <Progress 
                      percent={probability * 100} 
                      showInfo={false}
                      strokeColor="#52c41a"
                      size="small"
                    />
                  </div>
                ))}
              </div>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default QuantumLab;