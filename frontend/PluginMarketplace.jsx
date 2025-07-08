import React, { useState, useEffect } from 'react';

function PluginMarketplace() {
  const [plugins, setPlugins] = useState([]);
  const [selected, setSelected] = useState(null);
  const [args, setArgs] = useState('{}');
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch('/plugin_marketplace')
      .then(res => res.json())
      .then(data => setPlugins(data.plugins));
  }, []);

  const runPlugin = () => {
    fetch('/run_plugin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: selected, args: JSON.parse(args) })
    })
      .then(res => res.json())
      .then(data => setResult(data.result));
  };

  return (
    <div style={{padding: 20}}>
      <h2>Plugin Marketplace</h2>
      <ul>
        {plugins.map(p => (
          <li key={p.name}>
            <b>{p.name}</b> {p.quantum_properties.length > 0 && <span style={{color:'purple'}}>({p.quantum_properties.join(', ')})</span>}<br/>
            <i>{p.description}</i>
            <button onClick={() => setSelected(p.name)}>Select</button>
          </li>
        ))}
      </ul>
      {selected && (
        <div style={{marginTop: 20}}>
          <h3>Run Plugin: {selected}</h3>
          <textarea rows={3} cols={40} value={args} onChange={e => setArgs(e.target.value)} placeholder='{"a": "Alice", "b": "Bob"}' />
          <br/>
          <button onClick={runPlugin}>Run</button>
          {result && (
            <pre style={{background:'#eee', padding:10}}>{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}

export default PluginMarketplace;
