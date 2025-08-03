import React, { useCallback, useState, useRef, useEffect } from 'react';
import ReactFlow, {
  MiniMap, Controls, Background, addEdge, useNodesState, useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';
import io from 'socket.io-client';
import { FaAtom, FaRobot, FaPlug, FaPuzzlePiece, FaLock, FaUnlock, FaChevronDown, FaChevronRight, FaExclamationTriangle, FaCommentDots, FaHistory, FaUserShield, FaUsers, FaStore } from 'react-icons/fa';

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
  const [nodePermissions, setNodePermissions] = useState({}); // {nodeId: {owner, editors:[]}}
  const [showPerms, setShowPerms] = useState(null); // nodeId
  const [nodeLogs, setNodeLogs] = useState({}); // {nodeId: [log]}
  const [showLogs, setShowLogs] = useState(null); // nodeId
  const [showMarketplace, setShowMarketplace] = useState(false);
  const [triggeringNode, setTriggeringNode] = useState(null); // nodeId for per-node trigger
  const [aiSuggesting, setAiSuggesting] = useState(false);
  const [aiSuggestion, setAiSuggestion] = useState(null);
  const [pluginInstallStatus, setPluginInstallStatus] = useState({}); // {pluginName: 'installed'|'installing'|'upgrading'}
  // --- Automation/AI state ---
  const [autoCompleting, setAutoCompleting] = useState(false);
  const [autoCompleteResult, setAutoCompleteResult] = useState(null);
  const [smartConnecting, setSmartConnecting] = useState(false);
  const [smartConnectResult, setSmartConnectResult] = useState(null);
  const [optimizing, setOptimizing] = useState(false);
  const [optimizeResult, setOptimizeResult] = useState(null);

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
    // Save logs
    setNodeLogs(logs => {
      const newLogs = {...logs};
      Object.entries(resultMap).forEach(([id, res]) => {
        if (!newLogs[id]) newLogs[id] = [];
        newLogs[id].push({ result: res, time: Date.now() });
      });
      return newLogs;
    });
    setExecuting(false);
  };

  // Per-node execution trigger
  const runNode = async (nodeId) => {
    setTriggeringNode(nodeId);
    // Find all upstream nodes (simple BFS)
    const upstream = new Set();
    function collectUpstream(id) {
      edges.filter(e => e.target === id).forEach(e => {
        if (!upstream.has(e.source)) {
          upstream.add(e.source);
          collectUpstream(e.source);
        }
      });
    }
    collectUpstream(nodeId);
    const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
    const chain = [...upstream].map(id => nodeMap[id]).concat([nodeMap[nodeId]])
      .filter(Boolean)
      .filter(n => n.id !== '1')
      .map(n => ({ tool: n.data.label, args: JSON.parse(n.data.args || '{}'), nodeId: n.id }));
    const res = await fetch('/api/run_plugin_chain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chain })
    }).then(r => r.json());
    // Map results to node ids
    const resultMap = {};
    if (res.results && Array.isArray(res.results)) {
      res.results.forEach((r, i) => {
        const nodeId = chain[i]?.nodeId;
        if (nodeId) resultMap[nodeId] = r.result;
      });
    }
    setNodeResults(r => ({...r, ...resultMap}));
    setNodes(nds => nds.map(n => n.id in resultMap ? { ...n, data: { ...n.data, result: resultMap[n.id] } } : n));
    setNodeLogs(logs => {
      const newLogs = {...logs};
      Object.entries(resultMap).forEach(([id, res]) => {
        if (!newLogs[id]) newLogs[id] = [];
        newLogs[id].push({ result: res, time: Date.now() });
      });
      return newLogs;
    });
    setTriggeringNode(null);
  };

  // --- AI-powered graph suggestion (real backend, retry) ---
  const suggestNextNode = async (retry = false) => {
    setAiSuggesting(true);
    setAiSuggestion(null);
    try {
      const res = await fetch('/api/suggest_next_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setAiSuggestion(data.suggestion);
    } catch (e) {
      setAiSuggestion({ error: 'Failed to get suggestion from backend.' });
    }
    setAiSuggesting(false);
  };
  const addAiSuggestion = () => {
    if (!aiSuggestion || !aiSuggestion.label) return;
    const id = (nodes.length + 1).toString();
    setNodes(nds => [...nds, {
      id,
      data: { label: aiSuggestion.label, args: JSON.stringify(aiSuggestion.args||{}) },
      position: { x: 120 + Math.random() * 200, y: 120 + Math.random() * 200 },
      trigger: aiSuggestion.trigger || null
    }]);
    setAiSuggestion(null);
  };
  // --- Auto-complete graph (AI): suggest and add multiple nodes/edges ---
  const autoCompleteGraph = async () => {
    setAutoCompleting(true);
    setAutoCompleteResult(null);
    try {
      const res = await fetch('/api/auto_complete_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setAutoCompleteResult(data);
    } catch {
      setAutoCompleteResult({ error: 'Failed to auto-complete graph.' });
    }
    setAutoCompleting(false);
  };
  const acceptAutoComplete = () => {
    if (!autoCompleteResult || !autoCompleteResult.nodes) return;
    setNodes(autoCompleteResult.nodes);
    setEdges(autoCompleteResult.edges);
    setAutoCompleteResult(null);
  };
  const rejectAutoComplete = () => setAutoCompleteResult(null);
  // --- Smart connect (AI): auto-connect unconnected nodes ---
  const smartConnect = async () => {
    setSmartConnecting(true);
    setSmartConnectResult(null);
    try {
      const res = await fetch('/api/smart_connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setSmartConnectResult(data);
    } catch {
      setSmartConnectResult({ error: 'Failed to smart connect.' });
    }
    setSmartConnecting(false);
  };
  const acceptSmartConnect = () => {
    if (!smartConnectResult || !smartConnectResult.edges) return;
    setEdges(smartConnectResult.edges);
    setSmartConnectResult(null);
  };
  const rejectSmartConnect = () => setSmartConnectResult(null);
  // --- One-click optimize (AI): optimize workflow ---
  const optimizeGraph = async () => {
    setOptimizing(true);
    setOptimizeResult(null);
    try {
      const res = await fetch('/api/optimize_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setOptimizeResult(data);
    } catch {
      setOptimizeResult({ error: 'Failed to optimize graph.' });
    }
    setOptimizing(false);
  };
  const acceptOptimize = () => {
    if (!optimizeResult || !optimizeResult.nodes) return;
    setNodes(optimizeResult.nodes);
    setEdges(optimizeResult.edges);
    setOptimizeResult(null);
  };
  const rejectOptimize = () => setOptimizeResult(null);

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
    // Save logs
    setNodeLogs(logs => {
      const newLogs = {...logs};
      Object.entries(resultMap).forEach(([id, res]) => {
        if (!newLogs[id]) newLogs[id] = [];
        newLogs[id].push({ result: res, time: Date.now() });
      });
      return newLogs;
    });
    setExecuting(false);
  };

  // Per-node execution trigger
  const runNode = async (nodeId) => {
    setTriggeringNode(nodeId);
    // Find all upstream nodes (simple BFS)
    const upstream = new Set();
    function collectUpstream(id) {
      edges.filter(e => e.target === id).forEach(e => {
        if (!upstream.has(e.source)) {
          upstream.add(e.source);
          collectUpstream(e.source);
        }
      });
    }
    collectUpstream(nodeId);
    const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
    const chain = [...upstream].map(id => nodeMap[id]).concat([nodeMap[nodeId]])
      .filter(Boolean)
      .filter(n => n.id !== '1')
      .map(n => ({ tool: n.data.label, args: JSON.parse(n.data.args || '{}'), nodeId: n.id }));
    const res = await fetch('/api/run_plugin_chain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chain })
    }).then(r => r.json());
    // Map results to node ids
    const resultMap = {};
    if (res.results && Array.isArray(res.results)) {
      res.results.forEach((r, i) => {
        const nodeId = chain[i]?.nodeId;
        if (nodeId) resultMap[nodeId] = r.result;
      });
    }
    setNodeResults(r => ({...r, ...resultMap}));
    setNodes(nds => nds.map(n => n.id in resultMap ? { ...n, data: { ...n.data, result: resultMap[n.id] } } : n));
    setNodeLogs(logs => {
      const newLogs = {...logs};
      Object.entries(resultMap).forEach(([id, res]) => {
        if (!newLogs[id]) newLogs[id] = [];
        newLogs[id].push({ result: res, time: Date.now() });
      });
      return newLogs;
    });
    setTriggeringNode(null);
  };

  // --- AI-powered graph suggestion ---
  const suggestNextNode = async () => {
    setAiSuggesting(true);
    setAiSuggestion(null);
    try {
      const res = await fetch('/api/suggest_next_node', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setAiSuggestion(data.suggestion);
    } catch (e) {
      setAiSuggestion({ error: 'Failed to get suggestion from backend.' });
    }
    setAiSuggesting(false);
  };
  const addAiSuggestion = () => {
    if (!aiSuggestion || !aiSuggestion.label) return;
    const id = (nodes.length + 1).toString();
    setNodes(nds => [...nds, {
      id,
      data: { label: aiSuggestion.label, args: JSON.stringify(aiSuggestion.args||{}) },
      position: { x: 120 + Math.random() * 200, y: 120 + Math.random() * 200 },
      trigger: aiSuggestion.trigger || null
    }]);
    setAiSuggestion(null);
  };

  // Auto-complete graph (AI): suggest and add multiple nodes/edges
  const autoCompleteGraph = async () => {
    setAutoCompleting(true);
    setAutoCompleteResult(null);
    try {
      const res = await fetch('/api/auto_complete_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setAutoCompleteResult(data);
    } catch {
      setAutoCompleteResult({ error: 'Failed to auto-complete graph.' });
    }
    setAutoCompleting(false);
  };
  const acceptAutoComplete = () => {
    if (!autoCompleteResult || !autoCompleteResult.nodes) return;
    setNodes(autoCompleteResult.nodes);
    setEdges(autoCompleteResult.edges);
    setAutoCompleteResult(null);
  };
  const rejectAutoComplete = () => setAutoCompleteResult(null);

  // Smart connect (AI): auto-connect unconnected nodes
  const smartConnect = async () => {
    setSmartConnecting(true);
    setSmartConnectResult(null);
    try {
      const res = await fetch('/api/smart_connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setSmartConnectResult(data);
    } catch {
      setSmartConnectResult({ error: 'Failed to smart connect.' });
    }
    setSmartConnecting(false);
  };
  const acceptSmartConnect = () => {
    if (!smartConnectResult || !smartConnectResult.edges) return;
    setEdges(smartConnectResult.edges);
    setSmartConnectResult(null);
  };
  const rejectSmartConnect = () => setSmartConnectResult(null);

  // One-click optimize (AI): optimize workflow
  const optimizeGraph = async () => {
    setOptimizing(true);
    setOptimizeResult(null);
    try {
      const res = await fetch('/api/optimize_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges })
      });
      if (!res.ok) throw new Error('Backend error');
      const data = await res.json();
      setOptimizeResult(data);
    } catch {
      setOptimizeResult({ error: 'Failed to optimize graph.' });
    }
    setOptimizing(false);
  };
  const acceptOptimize = () => {
    if (!optimizeResult || !optimizeResult.nodes) return;
    setNodes(optimizeResult.nodes);
    setEdges(optimizeResult.edges);
    setOptimizeResult(null);
  };
  const rejectOptimize = () => setOptimizeResult(null);

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
        <button style={{marginLeft:10}} onClick={openMarketplace}><FaStore/> Marketplace</button>
        {/* AI-powered suggestion */}
        <button style={{marginLeft:10}} onClick={suggestNextNode} disabled={aiSuggesting}>{aiSuggesting?'Suggesting...':'Suggest Next Node (AI)'}</button>
        <button style={{marginLeft:10}} onClick={autoCompleteGraph} disabled={autoCompleting}>{autoCompleting?'Auto-completing...':'Auto-complete Graph'}</button>
        <button style={{marginLeft:10}} onClick={smartConnect} disabled={smartConnecting}>{smartConnecting?'Connecting...':'Smart Connect'}</button>
        <button style={{marginLeft:10}} onClick={optimizeGraph} disabled={optimizing}>{optimizing?'Optimizing...':'One-click Optimize'}</button>
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
      {/* Node permissions */}
      {showPerms && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:200}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:320}}>
            <b>Permissions for node {showPerms}</b>
            <form onSubmit={e=>{e.preventDefault(); savePerms(showPerms, e.target.owner.value, e.target.editors.value.split(',').map(s=>s.trim()));}}>
              <div style={{marginTop:10}}>
                Owner: <input name="owner" defaultValue={nodePermissions[showPerms]?.owner||userName} />
              </div>
              <div style={{marginTop:10}}>
                Editors (comma separated): <input name="editors" defaultValue={(nodePermissions[showPerms]?.editors||[]).join(', ')} />
              </div>
              <button style={{marginTop:10}} type="submit">Save</button>
              <button style={{marginLeft:10}} onClick={()=>setShowPerms(null)}>Cancel</button>
            </form>
          </div>
        </div>
      )}
      {/* Node execution logs */}
      {showLogs && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:200}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:320}}>
            <b>Execution logs for node {showLogs}</b>
            <ul style={{marginTop:10, maxHeight:180, overflowY:'auto'}}>
              {(nodeLogs[showLogs]||[]).map((l,i)=>(
                <li key={i} style={{marginBottom:6}}>
                  <span style={{fontFamily:'monospace', fontSize:13}}>{JSON.stringify(l.result)}</span>
                  <span style={{marginLeft:10, fontSize:11, color:'#888'}}>{new Date(l.time).toLocaleString()}</span>
                </li>
              ))}
            </ul>
            <button style={{marginTop:10}} onClick={()=>setShowLogs(null)}>Close</button>
          </div>
        </div>
      )}
      {/* Plugin marketplace modal */}
      {showMarketplace && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:300}}>
          <div style={{position:'absolute', top:'20%', left:'25%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:480, maxHeight:500, overflowY:'auto'}}>
            <b>Plugin Marketplace</b>
            <ul style={{marginTop:10}}>
              {plugins.map(p => (
                <li key={p.name} style={{marginBottom:10, display:'flex', alignItems:'center', gap:8}}>
                  <span style={{fontWeight:'bold'}}>{p.name}</span>
                  <span style={{fontSize:13, color:'#888'}}>{p.description}</span>
                  <button style={{marginLeft:10}} onClick={()=>addPluginFromMarketplace(p)}>Add</button>
                  <button style={{marginLeft:6}} onClick={()=>installPlugin(p)} disabled={pluginInstallStatus[p.name]==='installing' || pluginInstallStatus[p.name]==='installed'}>
                    {pluginInstallStatus[p.name]==='installing'?'Installing...':pluginInstallStatus[p.name]==='installed'?'Installed':'Install'}
                  </button>
                  <button style={{marginLeft:6}} onClick={()=>upgradePlugin(p)} disabled={pluginInstallStatus[p.name]!=='installed'}>
                    {pluginInstallStatus[p.name]==='upgrading'?'Upgrading...':'Upgrade'}
                  </button>
                </li>
              ))}
            </ul>
            <button style={{marginTop:10}} onClick={()=>setShowMarketplace(false)}>Close</button>
          </div>
        </div>
      )}
      {/* AI suggestion modal */}
      {aiSuggestion && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:400}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:320}}>
            <b>AI Suggestion</b>
            {aiSuggestion.error ? (
              <div style={{color:'#c00', marginTop:10}}>{aiSuggestion.error} <button onClick={()=>suggestNextNode(true)}>Retry</button></div>
            ) : (
              <div style={{marginTop:10}}>
                <div><b>Label:</b> {aiSuggestion.label}</div>
                <div><b>Args:</b> {JSON.stringify(aiSuggestion.args)}</div>
                {aiSuggestion.trigger && <div><b>Trigger:</b> {aiSuggestion.trigger}</div>}
                <button style={{marginTop:10}} onClick={addAiSuggestion}>Add to Graph</button>
              </div>
            )}
            <button style={{marginTop:10, marginLeft:10}} onClick={()=>setAiSuggestion(null)}>Close</button>
          </div>
        </div>
      )}
      {/* Auto-complete modal */}
      {autoCompleteResult && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:401}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:340}}>
            <b>AI Auto-complete</b>
            {autoCompleteResult.error ? (
              <div style={{color:'#c00', marginTop:10}}>{autoCompleteResult.error}</div>
            ) : (
              <div style={{marginTop:10}}>
                <div><b>Nodes:</b> {autoCompleteResult.nodes?.length}</div>
                <div><b>Edges:</b> {autoCompleteResult.edges?.length}</div>
                <button style={{marginTop:10}} onClick={acceptAutoComplete}>Accept</button>
                <button style={{marginLeft:10}} onClick={rejectAutoComplete}>Reject</button>
              </div>
            )}
            <button style={{marginTop:10, marginLeft:10}} onClick={()=>setAutoCompleteResult(null)}>Close</button>
          </div>
        </div>
      )}
      {/* Smart connect modal */}
      {smartConnectResult && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:402}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:340}}>
            <b>AI Smart Connect</b>
            {smartConnectResult.error ? (
              <div style={{color:'#c00', marginTop:10}}>{smartConnectResult.error}</div>
            ) : (
              <div style={{marginTop:10}}>
                <div><b>Edges:</b> {smartConnectResult.edges?.length}</div>
                <button style={{marginTop:10}} onClick={acceptSmartConnect}>Accept</button>
                <button style={{marginLeft:10}} onClick={rejectSmartConnect}>Reject</button>
              </div>
            )}
            <button style={{marginTop:10, marginLeft:10}} onClick={()=>setSmartConnectResult(null)}>Close</button>
          </div>
        </div>
      )}
      {/* Optimize modal */}
      {optimizeResult && (
        <div style={{position:'fixed', top:0, left:0, width:'100vw', height:'100vh', background:'rgba(0,0,0,0.2)', zIndex:403}}>
          <div style={{position:'absolute', top:'30%', left:'30%', background:'#fff', border:'2px solid #1976d2', borderRadius:10, padding:30, minWidth:340}}>
            <b>AI Workflow Optimization</b>
            {optimizeResult.error ? (
              <div style={{color:'#c00', marginTop:10}}>{optimizeResult.error}</div>
            ) : (
              <div style={{marginTop:10}}>
                <div><b>Nodes:</b> {optimizeResult.nodes?.length}</div>
                <div><b>Edges:</b> {optimizeResult.edges?.length}</div>
                <button style={{marginTop:10}} onClick={acceptOptimize}>Accept</button>
                <button style={{marginLeft:10}} onClick={rejectOptimize}>Reject</button>
              </div>
            )}
            <button style={{marginTop:10, marginLeft:10}} onClick={()=>setOptimizeResult(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default PluginGraphBuilder;
