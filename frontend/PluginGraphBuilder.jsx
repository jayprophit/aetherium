import React, { useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  MiniMap, Controls, Background, addEdge, useNodesState, useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';
import io from 'socket.io-client';

function PluginGraphBuilder({ plugins }) {
  const socket = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([
    { id: '1', type: 'input', data: { label: 'Start' }, position: { x: 0, y: 50 } }
  ]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedPlugin, setSelectedPlugin] = useState('');
  const [args, setArgs] = useState('{}');
  const [selectedNode, setSelectedNode] = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [executing, setExecuting] = useState(false);
  const [nodeResults, setNodeResults] = useState({});

  useEffect(() => {
    socket.current = io('http://localhost:4000');
    socket.current.on('graph', ({ nodes: n, edges: e }) => {
      setNodes(n.length ? n : [{ id: '1', type: 'input', data: { label: 'Start' }, position: { x: 0, y: 50 } }]);
      setEdges(e);
    });
    return () => { socket.current && socket.current.disconnect(); };
  }, []);

  // Sync graph changes
  useEffect(() => {
    if (socket.current) {
      socket.current.emit('update_graph', { nodes, edges });
    }
    // eslint-disable-next-line
  }, [nodes, edges]);

  const addPluginNode = () => {
    if (!selectedPlugin) return;
    const id = (nodes.length + 1).toString();
    setNodes(nds => {
      const newNodes = [...nds, {
        id,
        data: { label: selectedPlugin, args },
        position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 }
      }];
      if (socket.current) socket.current.emit('update_graph', { nodes: newNodes, edges });
      return newNodes;
    });
    setSelectedPlugin('');
    setArgs('{}');
  };

  const onConnect = useCallback((params) => setEdges(eds => addEdge(params, eds)), [setEdges]);

  // Node config popup
  const handleNodeClick = (evt, node) => {
    setSelectedNode(node);
    setArgs(node.data.args || '{}');
  };
  const saveNodeConfig = () => {
    setNodes(nds => {
      const newNodes = nds.map(n => n.id === selectedNode.id ? { ...n, data: { ...n.data, args } } : n);
      if (socket.current) socket.current.emit('update_graph', { nodes: newNodes, edges });
      return newNodes;
    });
    setSelectedNode(null);
  };

  // Graph-to-chain execution
  const runGraph = async () => {
    setExecuting(true);
    setLiveResult(null);
    setNodeResults({});
    // Topological sort (simple): assume edges are parent->child
    const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
    const chain = [];
    const visited = new Set();
    function visit(id) {
      if (visited.has(id)) return;
      visited.add(id);
      edges.filter(e => e.source === id).forEach(e => visit(e.target));
      if (id !== '1') { // skip Start node
        const n = nodeMap[id];
        chain.push({ tool: n.data.label, args: JSON.parse(n.data.args || '{}'), nodeId: id });
      }
    }
    visit('1');
    chain.reverse();
    const res = await fetch('/api/run_plugin_chain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chain })
    }).then(r => r.json());
    setLiveResult(res.results);
    // Map results to node ids
    const resultMap = {};
    if (res.results && Array.isArray(res.results)) {
      res.results.forEach((r, i) => {
        const nodeId = chain[i]?.nodeId;
        if (nodeId) resultMap[nodeId] = r.result;
      });
    }
    setNodeResults(resultMap);
    setNodes(nds => nds.map(n => n.id in resultMap ? { ...n, data: { ...n.data, result: resultMap[n.id] } } : { ...n, data: { ...n.data, result: undefined } }));
    setExecuting(false);
  };

  // Custom node renderer for result badge/tooltip
  const nodeTypes = {
    default: ({ id, data, selected }) => (
      <div style={{
        border: selected ? '2px solid #1976d2' : '1px solid #bbb',
        borderRadius: 8,
        background: data.result !== undefined ? '#e3fcec' : '#fff',
        minWidth: 90,
        minHeight: 40,
        padding: 8,
        position: 'relative',
        boxShadow: data.result !== undefined ? '0 0 8px #b2f5ea' : undefined
      }}>
        <div style={{fontWeight:'bold'}}>{data.label}</div>
        {data.args && <div style={{fontSize:12, color:'#888'}}>args: {data.args}</div>}
        {data.result !== undefined && (
          <div style={{
            position:'absolute', right:6, top:6, background:'#1976d2', color:'#fff', borderRadius:4, padding:'2px 6px', fontSize:11
          }} title={typeof data.result === 'object' ? JSON.stringify(data.result) : String(data.result)}>
            ✓
          </div>
        )}
        {data.result !== undefined && (
          <div style={{marginTop:6, fontSize:12, color:'#1976d2', wordBreak:'break-all'}}>
            {typeof data.result === 'object' ? JSON.stringify(data.result) : String(data.result)}
          </div>
        )}
      </div>
    )
  };

  return (
    <div style={{height: 600, border: '1px solid #ccc', borderRadius: 8, marginTop: 40, position:'relative'}}>
      <div style={{padding: 10}}>
        <select value={selectedPlugin} onChange={e => setSelectedPlugin(e.target.value)}>
          <option value="">Add plugin...</option>
          {plugins.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
        </select>
        <input style={{marginLeft:10, width:200}} value={args} onChange={e => setArgs(e.target.value)} placeholder='{"a": "Alice"}' />
        <button style={{marginLeft:10}} onClick={addPluginNode}>Add Node</button>
        <button style={{marginLeft:10}} onClick={runGraph} disabled={executing}>Run Graph</button>
        {executing && <span style={{marginLeft:10}}>Executing...</span>}
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
      {selectedNode && (
        <div style={{position:'absolute', top:100, left:100, background:'#fff', border:'1px solid #888', padding:20, zIndex:10, borderRadius:8, minWidth:320}}>
          <h4>Edit Node: {selectedNode.data.label}</h4>
          <textarea rows={3} cols={40} value={args} onChange={e => setArgs(e.target.value)} style={{width:'100%', fontFamily:'monospace'}}/>
          <br/>
          <button onClick={saveNodeConfig}>Save</button>
          <button style={{marginLeft:10}} onClick={()=>setSelectedNode(null)}>Cancel</button>
        </div>
      )}
      {liveResult && (
        <div style={{background:'#f8f8f8', padding:10, marginTop:10, borderRadius:4}}>
          <b>Graph Results:</b>
          <ol>
            {liveResult.map((r, i) => (
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

export default PluginGraphBuilder;
