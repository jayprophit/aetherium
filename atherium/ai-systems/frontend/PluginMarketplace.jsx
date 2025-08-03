import React, { useState, useEffect } from 'react';

const ARG_HINTS = {
  quantum_superposition: '{"options": ["cat", "dog"]}',
  quantum_entanglement: '{"a": "Alice", "b": "Bob"}',
  quantum_time_crystal: '{"period": 2, "cycles": 3}',
  quantum_random: '{"n": 5}',
  arxiv_quantum_latest: '{}',
  arxiv_summarize: '{"keyword": "entanglement"}',
  ai_code_generator: '{"task": "fibonacci sequence"}',
  wikipedia_summary: '{"topic": "Quantum mechanics"}',
  wolframalpha_compute: '{"appid": "YOUR_APP_ID", "query": "integrate x^2"}',
  weather_info: '{"city": "London", "apikey": "YOUR_API_KEY"}',
  github_repo_info: '{"repo": "octocat/Hello-World"}',
  pdf_summarizer: '{"pdf_b64": "..."}',
  youtube_transcript_summarizer: '{"video_id": "dQw4w9WgXcQ"}',
  news_headlines: '{"apikey": "YOUR_API_KEY"}',
  math_solver: '{"equation": "x**2 - 4"}',
  ai_image_generator: '{"prompt": "A cat riding a bicycle"}',
  quantum_circuit_simulator: '{"qubits": 2}'
};

const CATEGORY_COLORS = {
  ai: '#0074D9',
  quantum: '#B10DC9',
  external: '#2ECC40',
  utility: '#FF851B',
  math: '#FF4136',
  image: '#FFDC00',
  simulation: '#39CCCC'
};

function PluginMarketplace() {
  const [plugins, setPlugins] = useState([]);
  const [selected, setSelected] = useState(null);
  const [args, setArgs] = useState('{}');
  const [result, setResult] = useState(null);
  const [filter, setFilter] = useState('');
  const [expanded, setExpanded] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/plugin_marketplace')
      .then(res => res.json())
      .then(data => setPlugins(data.plugins));
  }, []);

  const runPlugin = () => {
    setLoading(true);
    setError(null);
    setResult(null);
    fetch('/run_plugin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool: selected, args: JSON.parse(args) })
    })
      .then(res => res.json())
      .then(data => {
        setResult(data.result);
        setLoading(false);
        if (data.result && data.result.error) setError(data.result.error);
      })
      .catch(e => { setError(String(e)); setLoading(false); });
  };

  const filtered = plugins.filter(p =>
    p.name.toLowerCase().includes(filter.toLowerCase()) ||
    (p.description && p.description.toLowerCase().includes(filter.toLowerCase()))
  );

  const handleFile = e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      const b64 = btoa(ev.target.result);
      setArgs(JSON.stringify({ pdf_b64: b64 }));
    };
    reader.readAsBinaryString(file);
  };

  const copyResult = () => {
    if (result) navigator.clipboard.writeText(typeof result === 'object' ? JSON.stringify(result, null, 2) : String(result));
  };

  return (
    <div style={{padding: 20}}>
      <h2>Plugin Marketplace</h2>
      <input
        type="text"
        placeholder="Search plugins..."
        value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{marginBottom: 10, width: 300}}
      />
      <ul>
        {filtered.map(p => (
          <li key={p.name} style={{marginBottom: 10, borderBottom:'1px solid #eee', paddingBottom:10}}>
            <b>{p.name}</b>
            {p.quantum_properties.map(tag => (
              <span key={tag} style={{background: CATEGORY_COLORS[tag] || '#ddd', color:'#222', borderRadius:4, padding:'2px 6px', marginLeft:6, fontSize:12}}>{tag}</span>
            ))}
            <button style={{marginLeft:10}} onClick={() => setExpanded(e => ({...e, [p.name]: !e[p.name]}))}>
              {expanded[p.name] ? 'Hide' : 'Details'}
            </button>
            {expanded[p.name] && (
              <div style={{marginTop:6, marginBottom:6}}>
                <i>{p.description}</i>
                <br/>
                <small>Args example: <code>{ARG_HINTS[p.name] || '{}'}</code></small>
                {p.name === 'pdf_summarizer' && (
                  <div>
                    <input type="file" accept="application/pdf" onChange={handleFile} />
                  </div>
                )}
                <button onClick={() => { setSelected(p.name); setArgs(ARG_HINTS[p.name] || '{}'); setResult(null); }}>Select</button>
              </div>
            )}
          </li>
        ))}
      </ul>
      {selected && (
        <div style={{marginTop: 20}}>
          <h3>Run Plugin: {selected}</h3>
          <textarea rows={3} cols={50} value={args} onChange={e => setArgs(e.target.value)} placeholder='{"a": "Alice", "b": "Bob"}' />
          <br/>
          <button onClick={runPlugin} disabled={loading}>Run</button>
          {loading && <span style={{marginLeft:10}}>Loading...</span>}
          {error && <div style={{color:'red', marginTop:10}}>{error}</div>}
          {result && !error && (
            <div style={{background:'#f8f8f8', padding:10, marginTop:10, borderRadius:4}}>
              <b>Result:</b>
              <button style={{marginLeft:10}} onClick={copyResult}>Copy</button>
              <pre style={{whiteSpace:'pre-wrap'}}>{typeof result === 'object' ? JSON.stringify(result, null, 2) : String(result)}</pre>
              {result.image_url && <img src={result.image_url} alt="Generated" style={{maxWidth:300, marginTop:10}} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PluginMarketplace;
