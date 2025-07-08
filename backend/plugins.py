"""
Plugin/tool system for Knowledge Base AI Assistant
Add new tools by subclassing ToolPlugin and registering in the registry.
"""
class ToolPlugin:
    name = "base"
    def run(self, args):
        return f"Base tool does nothing. Args: {args}"

# Example plugin: Echo
class EchoPlugin(ToolPlugin):
    name = "echo"
    def run(self, args):
        return f"Echo: {args.get('text', '')}"

# Plugin registry
tool_registry = {
    EchoPlugin.name: EchoPlugin(),
}

def run_tool(tool, args):
    plugin = tool_registry.get(tool)
    if plugin:
        return plugin.run(args)
    return f"Tool {tool} not found."

import requests

# Plugin registry for marketplace
PLUGIN_REGISTRY = {}

def plugin(name=None, quantum_properties=None):
    def decorator(fn):
        PLUGIN_REGISTRY[name or fn.__name__] = {
            'function': fn,
            'quantum_properties': quantum_properties or []
        }
        return fn
    return decorator

@plugin(name="quantum_superposition", quantum_properties=["superposition"])
def quantum_superposition_plugin(args):
    """Simulate quantum superposition: return multiple possible answers"""
    options = args.get('options', ['0', '1'])
    return {
        'result': f"In superposition: {options}",
        'explanation': "This plugin simulates quantum superposition by returning all possible states."
    }

@plugin(name="quantum_entanglement", quantum_properties=["entanglement"])
def quantum_entanglement_plugin(args):
    """Simulate quantum entanglement: link two variables"""
    a = args.get('a', 'Alice')
    b = args.get('b', 'Bob')
    return {
        'result': f"{a} and {b} are now entangled. Changing one affects the other!",
        'explanation': "This plugin simulates quantum entanglement between two entities."
    }

@plugin(name="arxiv_quantum_latest")
def arxiv_quantum_latest_plugin(args):
    """Fetch latest quantum physics papers from arXiv"""
    url = "http://export.arxiv.org/api/query?search_query=cat:quant-ph&sortBy=lastUpdatedDate&max_results=3"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch from arXiv'}
    # Simple parse: extract titles
    import re
    titles = re.findall(r'<title>(.*?)</title>', resp.text)[1:]  # skip feed title
    return {'papers': titles}

def run_tool(tool, args):
    plugin = PLUGIN_REGISTRY.get(tool)
    if not plugin:
        return f"Plugin '{tool}' not found."
    return plugin['function'](args)

def list_plugins():
    return [
        {
            'name': name,
            'quantum_properties': meta['quantum_properties'],
            'description': meta['function'].__doc__ or ''
        }
        for name, meta in PLUGIN_REGISTRY.items()
    ]
