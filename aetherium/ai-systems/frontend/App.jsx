import React, { useEffect, useState } from 'react';
import ChatApp from './ChatApp';
import PluginMarketplace from './PluginMarketplace';
import PluginChainBuilder from './PluginChainBuilder';
import PluginGraphBuilder from './PluginGraphBuilder';

function App() {
  const [plugins, setPlugins] = useState([]);
  useEffect(() => {
    fetch('/plugin_marketplace')
      .then(res => res.json())
      .then(data => setPlugins(data.plugins));
  }, []);
  return (
    <div>
      <h1>Knowledge Base AI Platform</h1>
      <ChatApp />
      <hr/>
      <PluginMarketplace />
      <PluginChainBuilder plugins={plugins} />
      <PluginGraphBuilder plugins={plugins} />
    </div>
  );
}

export default App;
