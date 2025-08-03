import React from 'react';
import { Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  ExperimentOutlined,
  ClockCircleOutlined,
  BranchesOutlined,
  RobotOutlined,
  WifiOutlined,
  BarChartOutlined,
  SettingOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/quantum',
      icon: <ExperimentOutlined />,
      label: 'Quantum Lab',
    },
    {
      key: '/time-crystals',
      icon: <ClockCircleOutlined />,
      label: 'Time Crystals',
    },
    {
      key: '/neuromorphic',
      icon: <BranchesOutlined />,
      label: 'Neuromorphic',
    },
    {
      key: '/ai-optimization',
      icon: <RobotOutlined />,
      label: 'AI Optimization',
    },
    {
      key: '/iot',
      icon: <WifiOutlined />,
      label: 'IoT Devices',
    },
    {
      key: '/metrics',
      icon: <BarChartOutlined />,
      label: 'System Metrics',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: 'Settings',
    },
  ];

  const handleMenuClick = (item: any) => {
    navigate(item.key);
  };

  return (
    <Menu
      theme="dark"
      selectedKeys={[location.pathname]}
      mode="inline"
      items={menuItems}
      onClick={handleMenuClick}
      style={{ borderRight: 0 }}
    />
  );
};

export default Sidebar;