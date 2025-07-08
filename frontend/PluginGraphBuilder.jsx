import React, { useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  MiniMap, Controls, Background, addEdge, useNodesState, useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';
import io from 'socket.io-client';
import { FaAtom, FaRobot, FaPlug, FaPuzzlePiece, FaLock, FaUnlock, FaChevronDown, FaChevronRight, FaExclamationTriangle, FaCommentDots, FaHistory } from 'react-icons/fa';

const PLUGIN_COLORS = {
  quantum: '#e1bee7',
  ai: '#b3e5fc',
  api: '#ffe082',
  default: '#f5f5f5'
};

const WORKFLOW_TEMPLATES = [
  {
    name: 'Quantum Chain',
    nodes: [
      { id: '1', type: 'input', data: { label: 'Start' }, position: { x: 0, y: 50 } },
      { id: '2', data: { label: 'quantum_superposition', args: '{"options":["0","1"]}' }, position: { x: 200, y: 100 } },
      { id: '3', data: { label: 'quantum_entanglement', args: '{"a":"Alice","b":"Bob"}' }, position: { x: 400, y: 100 } }
    ],
    edges: [
      { id: 'e1-2', source: '1', target: '2' },
      { id: 'e2-3', source: '2', target: '3' }
    ]
  },
  {
    name: 'AI Summarizer',
    nodes: [
      { id: '1', type: 'input', data: { label: 'Start' }, position: { x: 0, y: 50 } },
      { id: '2', data: { label: 'summarizer', args: '{"text":"Paste text here"}' }, position: { x: 200, y: 100 } }
    ],
    edges: [
      { id: 'e1-2', source: '1', target: '2' }
    ]
  }
];

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
  const [showDelete, setShowDelete] = useState(false);
  const [importExport, setImportExport] = useState('');
  const [showImport, setShowImport] = useState(false);
  const [hoveredPlugin, setHoveredPlugin] = useState(null);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [search, setSearch] = useState('');
  const [clipboard, setClipboard] = useState(null);
  const [groups, setGroups] = useState([]); // [{id, nodeIds:[] }]
  const [nodeNotes, setNodeNotes] = useState({}); // {nodeId: note}
  const [lockedNodes, setLockedNodes] = useState([]); // [nodeId]
  const [collapsedGroups, setCollapsedGroups] = useState([]); // [groupId]
  const [validationWarning, setValidationWarning] = useState('');
  const [nodeVersions, setNodeVersions] = useState({}); // {nodeId: [history]}
  const [showVersion, setShowVersion] = useState(null); // nodeId
  const [nodeComments, setNodeComments] = useState({}); // {nodeId: [{user, text, time}]}
  const [showComments, setShowComments] = useState(null); // nodeId
  const [userName] = useState(() => 'User' + Math.floor(Math.random()*1000));
  const [advValidation, setAdvValidation] = useState('');

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

  // Keyboard shortcuts: delete node, run graph
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Delete' && selectedNode && selectedNode.id !== '1') {
        setShowDelete(true);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        runGraph();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
    // eslint-disable-next-line
  }, [selectedNode, nodes, edges]);

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

  const onConnect = useCallback((params) => setEdges(eds => addEdge({ ...params, animated: true, style: { stroke: '#1976d2' } }, eds)), [setEdges]);

  // Multi-select: shift+click
  const handleNodeClick = (evt, node) => {
    if (evt.shiftKey) {
      setSelectedNodes(sel => sel.includes(node.id) ? sel.filter(id => id !== node.id) : [...sel, node.id]);
    } else {
      setSelectedNode(node);
      setSelectedNodes([node.id]);
      setArgs(node.data.args || '{}');
    }
  };

  // Undo/redo
  useEffect(() => {
    setUndoStack([]);
    setRedoStack([]);
  }, []);
  useEffect(() => {
    setUndoStack(stack => [...stack, { nodes, edges }]);
    // eslint-disable-next-line
  }, [nodes, edges]);
  const undo = () => {
    setUndoStack(stack => {
      if (stack.length < 2) return stack;
      const prev = stack[stack.length - 2];
      setRedoStack(rstack => [stack[stack.length - 1], ...rstack]);
      setNodes(prev.nodes);
      setEdges(prev.edges);
      return stack.slice(0, -1);
    });
  };
  const redo = () => {
    setRedoStack(rstack => {
      if (!rstack.length) return rstack;
      const next = rstack[0];
      setNodes(next.nodes);
      setEdges(next.edges);
      setUndoStack(stack => [...stack, next]);
      return rstack.slice(1);
    });
  };
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') undo();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') redo();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undoStack, redoStack]);

  // Node color coding
  function getNodeColor(label) {
    if (!label) return PLUGIN_COLORS.default;
    if (label.startsWith('quantum')) return PLUGIN_COLORS.quantum;
    if (label.startsWith('ai') || label === 'summarizer') return PLUGIN_COLORS.ai;
    if (label.endsWith('api') || label.endsWith('API')) return PLUGIN_COLORS.api;
    return PLUGIN_COLORS.default;
  }

  function getNodeIcon(label) {
    if (!label) return <FaPuzzlePiece style={{color:'#aaa'}}/>;
    if (label.startsWith('quantum')) return <FaAtom style={{color:'#8e24aa'}}/>;
    if (label.startsWith('ai') || label === 'summarizer') return <FaRobot style={{color:'#0288d1'}}/>;
    if (label.endsWith('api') || label.endsWith('API')) return <FaPlug style={{color:'#fbc02d'}}/>;
    return <FaPuzzlePiece style={{color:'#aaa'}}/>;
  }

  // Insert workflow template
  const insertTemplate = (tpl) => {
    setNodes(tpl.nodes);
    setEdges(tpl.edges);
    setLiveResult(null);
    setNodeResults({});
  };

  // Node config popup
  const saveNodeConfig = () => {
    setNodes(nds => {
      const newNodes = nds.map(n => n.id === selectedNode.id ? { ...n, data: { ...n.data, args } } : n);
      if (socket.current) socket.current.emit('update_graph', { nodes: newNodes, edges });
      // Save version
      setNodeVersions(vers => ({
        ...vers,
        [selectedNode.id]: [ ...(vers[selectedNode.id]||[]), { args, time: Date.now() } ]
      }));
      return newNodes;
    });
    setSelectedNode(null);
  };
  const deleteNode = () => {
    setNodes(nds => {
      const newNodes = nds.filter(n => n.id !== selectedNode.id);
      const newEdges = edges.filter(e => e.source !== selectedNode.id && e.target !== selectedNode.id);
      if (socket.current) socket.current.emit('update_graph', { nodes: newNodes, edges: newEdges });
      setEdges(newEdges);
      return newNodes;
    });
    setSelectedNode(null);
    setShowDelete(false);
  };

  // Drag-to-reorder: handled by ReactFlow (position updates)
  const onNodeDragStop = (evt, node) => {
    setNodes(nds => nds.map(n => n.id === node.id ? { ...n, position: node.position } : n));
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

  // Export/import
  const exportGraph = () => {
    setImportExport(JSON.stringify({ nodes, edges }, null, 2));
    setShowImport(true);
  };
  const importGraph = () => {
    try {
      const { nodes: n, edges: e } = JSON.parse(importExport);
      setNodes(n);
      setEdges(e);
      setShowImport(false);
    } catch {
      alert('Invalid JSON');
    }
  };
  const clearGraph = () => {
    setNodes([{ id: '1', type: 'input', data: { label: 'Start' }, position: { x: 0, y: 50 } }]);
    setEdges([]);
    setLiveResult(null);
    setNodeResults({});
  };

  // Custom node renderer for result badge/tooltip
  const nodeTypes = {
    default: ({ id, data, selected }) => (
      <div style={{
        border: selected || selectedNodes.includes(id) ? '2.5px solid #1976d2' : '1.5px solid #bbb',
        borderRadius: 10,
        background: data.result !== undefined ? '#e3fcec' : getNodeColor(data.label),
        minWidth: 90,
        minHeight: 40,
        padding: 8,
        position: 'relative',
        boxShadow: data.result !== undefined ? '0 0 12px #b2f5ea' : undefined,
        transition: 'box-shadow 0.2s, border 0.2s',
        cursor: lockedNodes.includes(id) ? 'not-allowed' : 'pointer',
        outline: selected || selectedNodes.includes(id) ? '2px solid #90caf9' : undefined,
        opacity: lockedNodes.includes(id) ? 0.7 : 1
      }}>
        <div style={{display:'flex',alignItems:'center',gap:6}}>
          {getNodeIcon(data.label)}
          <span style={{fontWeight:'bold'}}>{data.label}</span>
          <span style={{marginLeft:4, cursor:'pointer'}} onClick={e=>{e.stopPropagation();toggleLockNode(id);}} title={lockedNodes.includes(id)?'Unlock':'Lock'}>
            {lockedNodes.includes(id)?<FaLock/>:<FaUnlock/>}
          </span>
          <span style={{marginLeft:2, cursor:'pointer'}} onClick={e=>{e.stopPropagation();openNote(id);}} title={nodeNotes[id]?'Edit note':'Add note'}>
            <FaPuzzlePiece style={{color: nodeNotes[id]?'#1976d2':'#bbb'}}/>
          </span>
          <span style={{marginLeft:2, cursor:'pointer'}} onClick={e=>{e.stopPropagation();openComments(id);}} title="Comments"><FaCommentDots/></span>
          <span style={{marginLeft:2, cursor:'pointer'}} onClick={e=>{e.stopPropagation();openVersion(id);}} title="Version history"><FaHistory/></span>
        </div>
        {data.args && <div style={{fontSize:12, color:'#888'}}>args: {data.args}</div>}
        {nodeNotes[id] && <div style={{fontSize:11, color:'#1976d2', marginTop:2, fontStyle:'italic'}}>{nodeNotes[id]}</div>}
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

  // Plugin info tooltip
  const pluginInfo = hoveredPlugin && plugins.find(p => p.name === hoveredPlugin);

  // Node search/filter
  const filteredNodes = search
    ? nodes.filter(n => n.data.label.toLowerCase().includes(search.toLowerCase()))
    : nodes;

  // Copy/paste
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c' && selectedNodes.length) {
        setClipboard(nodes.filter(n => selectedNodes.includes(n.id)));
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v' && clipboard) {
        // Paste nodes with new ids/positions
        const maxId = Math.max(...nodes.map(n => parseInt(n.id, 10) || 1));
        const pasted = clipboard.map((n, i) => ({
          ...n,
          id: (maxId + i + 1).toString(),
          position: { x: n.position.x + 40, y: n.position.y + 40 }
        }));
        setNodes(nds => [...nds, ...pasted]);
      }
      // Group/Ungroup
      if (e.key.toLowerCase() === 'g' && selectedNodes.length > 1) {
        const groupId = 'group-' + Date.now();
        setGroups(gs => [...gs, { id: groupId, nodeIds: selectedNodes }]);
      }
      if (e.key.toLowerCase() === 'u' && selectedNodes.length) {
        setGroups(gs => gs.filter(g => !g.nodeIds.some(id => selectedNodes.includes(id))));
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedNodes, clipboard, nodes]);

  // Node locking
  const toggleLockNode = (id) => {
    setLockedNodes(locked => locked.includes(id) ? locked.filter(l=>l!==id) : [...locked, id]);
  };
  // Collapsible groups
  const toggleCollapseGroup = (gid) => {
    setCollapsedGroups(cg => cg.includes(gid) ? cg.filter(g=>g!==gid) : [...cg, gid]);
  };
  // Node notes
  const openNote = (id) => {
    const note = prompt('Edit note for node:', nodeNotes[id] || '');
    if (note !== null) setNodeNotes(nn => ({...nn, [id]: note}));
  };
  // Workflow validation: warn if any node (except Start) has no incoming edge
  useEffect(() => {
    const nodeIds = nodes.map(n=>n.id).filter(id=>id!=='1');
    const targets = edges.map(e=>e.target);
    const unconnected = nodeIds.filter(id=>!targets.includes(id));
    if (unconnected.length) {
      setValidationWarning(`Warning: Node(s) ${unconnected.join(', ')} not connected!`);
    } else {
      setValidationWarning('');
    }
  }, [nodes, edges]);

  // Advanced validation: cycle detection, required args
  useEffect(() => {
    // Cycle detection
    const graph = {};
    nodes.forEach(n => { graph[n.id] = []; });
    edges.forEach(e => { graph[e.source].push(e.target); });
    let hasCycle = false;
    const visited = {}, recStack = {};
    function dfs(v) {
      visited[v] = true; recStack[v] = true;
      for (const n of graph[v]) {
        if (!visited[n] && dfs(n)) return true;
        else if (recStack[n]) return true;
      }
      recStack[v] = false; return false;
    }
    for (const n of nodes.map(n=>n.id)) {
      if (!visited[n] && dfs(n)) { hasCycle = true; break; }
    }
    // Required args check (simple: must be non-empty JSON)
    const missingArgs = nodes.filter(n => n.id !== '1' && (!n.data.args || n.data.args.trim() === '{}' || n.data.args.trim() === ''));
    let msg = '';
    if (hasCycle) msg += 'Error: Graph has a cycle! ';
    if (missingArgs.length) msg += `Node(s) missing args: ${missingArgs.map(n=>n.data.label).join(', ')}`;
    setAdvValidation(msg);
  }, [nodes, edges]);

  return (
    <div style={{height: 600, border: '1px solid #ccc', borderRadius: 8, marginTop: 40, position:'relative'}}>
      {/* Toolbar */}
      <div style={{padding: 10, display:'flex', alignItems:'center', gap:12}}>
        <input style={{width:120}} value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search nodes..." />
        <select value={selectedPlugin} onChange={e => setSelectedPlugin(e.target.value)}
          onMouseOver={e => setHoveredPlugin(e.target.value)}
          onMouseOut={() => setHoveredPlugin(null)}>
          <option value="">Add plugin...</option>
          {plugins.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
        </select>
        <input style={{marginLeft:10, width:200}} value={args} onChange={e => setArgs(e.target.value)} placeholder='{"a": "Alice"}' />
        <button style={{marginLeft:10}} onClick={addPluginNode}>Add Node</button>
        <button style={{marginLeft:10}} onClick={runGraph} disabled={executing}>Run Graph (Ctrl+Enter)</button>
        <button style={{marginLeft:10}} onClick={clearGraph}>Clear</button>
        <button style={{marginLeft:10}} onClick={exportGraph}>Export</button>
        <button style={{marginLeft:10}} onClick={()=>setShowImport(true)}>Import</button>
        <button style={{marginLeft:10}} onClick={undo}>Undo</button>
        <button style={{marginLeft:10}} onClick={redo}>Redo</button>
        <select style={{marginLeft:10}} onChange={e => {
          const tpl = WORKFLOW_TEMPLATES.find(t => t.name === e.target.value);
          if (tpl) insertTemplate(tpl);
        }} defaultValue="">
          <option value="">Templates...</option>
          {WORKFLOW_TEMPLATES.map(t => <option key={t.name} value={t.name}>{t.name}</option>)}
        </select>
        {executing && <span style={{marginLeft:10}}>Executing...</span>}
      </div>
      {pluginInfo && (
        <div style={{position:'absolute', left:180, top:50, background:'#fff', border:'1px solid #1976d2', borderRadius:8, padding:10, zIndex:20, minWidth:220, boxShadow:'0 2px 12px #90caf9'}}>
          <b>{pluginInfo.name}</b><br/>
          <span style={{fontSize:13}}>{pluginInfo.description || 'No description.'}</span>
          {pluginInfo.quantum_properties && pluginInfo.quantum_properties.length > 0 && (
            <div style={{marginTop:6, fontSize:12, color:'#1976d2'}}>
              Quantum: {pluginInfo.quantum_properties.join(', ')}
            </div>
          )}
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={handleNodeClick}
        onNodeDragStop={onNodeDragStop}
        nodeTypes={nodeTypes}
        fitView
      >
        <MiniMap />
        <Controls />
        <Background />
        {/* Group highlight */}
        {groups.map(g => !collapsedGroups.includes(g.id) && (
          <div key={g.id} style={{
            position:'absolute',
            left: Math.min(...nodes.filter(n=>g.nodeIds.includes(n.id)).map(n=>n.position.x))-20,
            top: Math.min(...nodes.filter(n=>g.nodeIds.includes(n.id)).map(n=>n.position.y))-20,
            width: Math.max(...nodes.filter(n=>g.nodeIds.includes(n.id)).map(n=>n.position.x+100))-Math.min(...nodes.filter(n=>g.nodeIds.includes(n.id)).map(n=>n.position.x))+40,
            height: Math.max(...nodes.filter(n=>g.nodeIds.includes(n.id)).map(n=>n.position.y+60))-Math.min(...nodes.filter(n=>g.nodeIds.includes(n.id)).map(n=>n.position.y))+40,
            border:'2.5px dashed #1976d2',
            borderRadius:18,
            background:'rgba(25,118,210,0.07)',
            zIndex:2,
            display:'flex', alignItems:'flex-start', justifyContent:'flex-end',
          }}>
            <span style={{margin:4, cursor:'pointer', color:'#1976d2', background:'#fff', borderRadius:6, padding:'2px 6px', fontSize:13, border:'1px solid #1976d2'}}
              onClick={()=>toggleCollapseGroup(g.id)}
              title={collapsedGroups.includes(g.id)?'Expand group':'Collapse group'}>
              {collapsedGroups.includes(g.id)?<FaChevronRight/>:<FaChevronDown/>}
            </span>
          </div>
        ))}
        {/* Validation warning */}
        {validationWarning && (
          <div style={{position:'absolute', top:10, right:20, background:'#fffbe6', color:'#b26a00', border:'1.5px solid #ffe082', borderRadius:8, padding:'6px 16px', zIndex:30, display:'flex', alignItems:'center', gap:8}}>
            <FaExclamationTriangle/>
            {validationWarning}
          </div>
        )}
        {/* Advanced validation warning */}
        {advValidation && (
          <div style={{position:'absolute', top:50, right:20, background:'#fff3f3', color:'#c62828', border:'1.5px solid #ffcdd2', borderRadius:8, padding:'6px 16px', zIndex:31, display:'flex', alignItems:'center', gap:8}}>
            <FaExclamationTriangle/>
            {advValidation}
          </div>
        )}
      </ReactFlow>
      {selectedNode && (
        <div style={{position:'absolute', top:100, left:100, background:'#fff', border:'1px solid #888', padding:20, zIndex:10, borderRadius:8, minWidth:320}}>
          <h4>Edit Node: {selectedNode.data.label}</h4>
          <textarea rows={3} cols={40} value={args} onChange={e => setArgs(e.target.value)} style={{width:'100%', fontFamily:'monospace'}}/>
          <br/>
          <button onClick={saveNodeConfig}>Save</button>
          {selectedNode.id !== '1' && <button style={{marginLeft:10, color:'#c00'}} onClick={()=>setShowDelete(true)}>Delete</button>}
          <button style={{marginLeft:10}} onClick={()=>setSelectedNode(null)}>Cancel</button>
        </div>
      )}
      {showDelete && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:100}}>
          <div style={{position:'absolute', top:'40%', left:'40%', background:'#fff', border:'2px solid #c00', borderRadius:10, padding:30, minWidth:260}}>
            <b>Delete node "{selectedNode?.data?.label}"?</b>
            <div style={{marginTop:16}}>
              <button style={{color:'#c00'}} onClick={deleteNode}>Delete</button>
              <button style={{marginLeft:10}} onClick={()=>setShowDelete(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
      {showImport && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:100}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:420}}>
            <b>Import/Export Graph</b>
            <textarea rows={8} cols={60} value={importExport} onChange={e=>setImportExport(e.target.value)} style={{width:'100%', fontFamily:'monospace', marginTop:10}}/>
            <div style={{marginTop:16}}>
              <button onClick={importGraph}>Import</button>
              <button style={{marginLeft:10}} onClick={()=>setShowImport(false)}>Close</button>
            </div>
          </div>
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
      {/* Node version history */}
      {showVersion && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:200}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:320}}>
            <b>Version history for node {showVersion}</b>
            <ul style={{marginTop:10}}>
              {(nodeVersions[showVersion]||[]).map((v,i)=>(
                <li key={i} style={{marginBottom:6}}>
                  <span style={{fontFamily:'monospace', fontSize:13}}>{v.args}</span>
                  <span style={{marginLeft:10, fontSize:11, color:'#888'}}>{new Date(v.time).toLocaleString()}</span>
                  <button style={{marginLeft:10}} onClick={()=>restoreVersion(showVersion, v)}>Restore</button>
                </li>
              ))}
            </ul>
            <button onClick={()=>setShowVersion(null)}>Close</button>
          </div>
        </div>
      )}
      {/* Node comments */}
      {showComments && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:200}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:320}}>
            <b>Comments for node {showComments}</b>
            <ul style={{marginTop:10, maxHeight:180, overflowY:'auto'}}>
              {(nodeComments[showComments]||[]).map((c,i)=>(
                <li key={i} style={{marginBottom:6}}>
                  <span style={{fontWeight:'bold', color:'#1976d2'}}>{c.user}</span>:
                  <span style={{marginLeft:6}}>{c.text}</span>
                  <span style={{marginLeft:10, fontSize:11, color:'#888'}}>{new Date(c.time).toLocaleString()}</span>
                </li>
              ))}
            </ul>
            <form onSubmit={e=>{e.preventDefault(); const text=e.target.comment.value.trim(); if(text) addComment(showComments, text); e.target.comment.value='';}}>
              <input name="comment" style={{width:'80%'}} placeholder="Add comment..." autoFocus />
              <button type="submit">Send</button>
            </form>
            <button style={{marginTop:10}} onClick={()=>setShowComments(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default PluginGraphBuilder;
