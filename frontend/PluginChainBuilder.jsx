import React, { useState } from 'react';

function PluginChainBuilder({ plugins }) {
  const [chain, setChain] = useState([]);
  const [current, setCurrent] = useState('');
  const [args, setArgs] = useState('{}');
  const [inputKey, setInputKey] = useState('');
  const [outputKey, setOutputKey] = useState('');
  const [condition, setCondition] = useState('');
  const [loop, setLoop] = useState(1);
  const [useContext, setUseContext] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const addStep = () => {
    if (!current) return;
    const step = { tool: current, args: JSON.parse(args) };
    if (inputKey) step.input_key = inputKey;
    if (outputKey) step.output_key = outputKey;
    if (condition) {
      try { step.condition = JSON.parse(condition); } catch {}
    }
    if (loop > 1) step.loop = Number(loop);
    if (useContext) step.use_context = useContext;
    setChain([...chain, step]);
    setCurrent(''); setArgs('{}'); setInputKey(''); setOutputKey(''); setCondition(''); setLoop(1); setUseContext('');
  };

  const runChain = () => {
    setLoading(true);
    setError(null);
    setResult(null);
    fetch('/run_plugin_chain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chain })
    })
      .then(res => res.json())
      .then(data => { setResult(data.results); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  };

  return (
    <div style={{marginTop:40, padding:20, border:'1px solid #ccc', borderRadius:8}}>
      <h2>Plugin Chain Builder</h2>
      <div>
        <select value={current} onChange={e => setCurrent(e.target.value)}>
          <option value="">Select plugin...</option>
          {plugins.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
        </select>
        <input style={{marginLeft:10, width:200}} value={args} onChange={e => setArgs(e.target.value)} placeholder='{"a": "Alice"}' />
        <input style={{marginLeft:10, width:80}} value={inputKey} onChange={e => setInputKey(e.target.value)} placeholder='input_key' />
        <input style={{marginLeft:10, width:80}} value={outputKey} onChange={e => setOutputKey(e.target.value)} placeholder='output_key' />
        <input style={{marginLeft:10, width:80}} value={useContext} onChange={e => setUseContext(e.target.value)} placeholder='use_context' />
        <input style={{marginLeft:10, width:60}} type="number" min={1} value={loop} onChange={e => setLoop(e.target.value)} placeholder='loop' />
        <input style={{marginLeft:10, width:120}} value={condition} onChange={e => setCondition(e.target.value)} placeholder='{"key": "val"}' />
        <button style={{marginLeft:10}} onClick={addStep}>Add Step</button>
      </div>
      <ol>
        {chain.map((step, i) => (
          <li key={i} style={{marginBottom:4}}>
            <b>{step.tool}</b> args: <code>{JSON.stringify(step.args)}</code>
            {step.input_key && <span> | input_key: <code>{step.input_key}</code></span>}
            {step.output_key && <span> | output_key: <code>{step.output_key}</code></span>}
            {step.use_context && <span> | use_context: <code>{step.use_context}</code></span>}
            {step.loop && step.loop > 1 && <span> | loop: <code>{step.loop}</code></span>}
            {step.condition && <span> | condition: <code>{JSON.stringify(step.condition)}</code></span>}
          </li>
        ))}
      </ol>
      <button onClick={runChain} disabled={loading || chain.length === 0}>Run Chain</button>
      {loading && <span style={{marginLeft:10}}>Running...</span>}
      {error && <div style={{color:'red', marginTop:10}}>{error}</div>}
      {result && (
        <div style={{background:'#f8f8f8', padding:10, marginTop:10, borderRadius:4}}>
          <b>Chain Results:</b>
          <ol>
            {result.map((r, i) => (
              <li key={i} style={{marginBottom:6}}>
                <b>{r.tool}</b>:
                <pre style={{display:'inline', background:'#e0f7fa', borderRadius:4, padding:4}}>{JSON.stringify(r.result, null, 2)}</pre>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export default PluginChainBuilder;
