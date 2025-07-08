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
