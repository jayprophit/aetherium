"""
Plugin/tool system for Knowledge Base AI Assistant
Add new tools by subclassing ToolPlugin and registering in the registry.
"""
import requests
import random
import time
import base64
import sympy
try:
    from qiskit import QuantumCircuit, Aer, execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

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

@plugin(name="quantum_time_crystal", quantum_properties=["time_crystal", "periodicity"])
def quantum_time_crystal_plugin(args):
    """Simulate a time crystal: periodic plugin execution and memory"""
    period = args.get('period', 2)
    cycles = args.get('cycles', 3)
    events = []
    for i in range(cycles):
        events.append(f"Cycle {i+1}: Time crystal state at t={i*period}s")
    return {
        'result': events,
        'explanation': f"Simulated time crystal with period {period}s for {cycles} cycles."
    }

@plugin(name="quantum_random", quantum_properties=["quantum_randomness"])
def quantum_random_plugin(args):
    """Quantum random number generator (simulated)"""
    n = args.get('n', 1)
    return {
        'result': [random.choice([0, 1]) for _ in range(n)],
        'explanation': "Simulated quantum random bitstring."
    }

@plugin(name="arxiv_quantum_latest")
def arxiv_quantum_latest_plugin(args):
    """Fetch latest quantum physics papers from arXiv"""
    url = "http://export.arxiv.org/api/query?search_query=cat:quant-ph&sortBy=lastUpdatedDate&max_results=3"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch from arXiv'}
    import re
    titles = re.findall(r'<title>(.*?)</title>', resp.text)[1:]
    return {'papers': titles}

@plugin(name="arxiv_summarize")
def arxiv_summarize_plugin(args):
    """Fetch and summarize a quantum physics paper from arXiv by title keyword"""
    keyword = args.get('keyword', 'quantum')
    url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&sortBy=lastUpdatedDate&max_results=1"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch from arXiv'}
    import re
    titles = re.findall(r'<title>(.*?)</title>', resp.text)[1:]
    summaries = re.findall(r'<summary>(.*?)</summary>', resp.text, re.DOTALL)
    return {'title': titles[0] if titles else '', 'summary': summaries[0].strip() if summaries else ''}

@plugin(name="ai_code_generator", quantum_properties=["ai", "code_generation"])
def ai_code_generator_plugin(args):
    """Generate Python code for a given task using LLM (stub)"""
    task = args.get('task', 'print Hello World')
    # In production, call LLM API here
    return {
        'result': f"def solution():\n    # {task}\n    print('Hello World')",
        'explanation': "Stub: Replace with LLM code generation API."
    }

@plugin(name="wikipedia_summary")
def wikipedia_summary_plugin(args):
    """Fetch a summary for a topic from Wikipedia"""
    topic = args.get('topic', 'Quantum mechanics')
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(" ", "_")}'
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch from Wikipedia'}
    data = resp.json()
    return {'title': data.get('title'), 'summary': data.get('extract')}

@plugin(name="wolframalpha_compute")
def wolframalpha_compute_plugin(args):
    """Compute a result using WolframAlpha (requires API key)"""
    appid = args.get('appid', 'YOUR_APP_ID')
    query = args.get('query', '2+2')
    url = f'https://api.wolframalpha.com/v1/result?i={query}&appid={appid}'
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch from WolframAlpha'}
    return {'result': resp.text}

@plugin(name="weather_info")
def weather_info_plugin(args):
    """Get current weather for a city (OpenWeatherMap, requires API key)"""
    city = args.get('city', 'London')
    apikey = args.get('apikey', 'YOUR_API_KEY')
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&units=metric'
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch weather'}
    data = resp.json()
    return {'city': city, 'temp_C': data['main']['temp'], 'weather': data['weather'][0]['description']}

@plugin(name="github_repo_info")
def github_repo_info_plugin(args):
    """Get info about a GitHub repository"""
    repo = args.get('repo', 'octocat/Hello-World')
    url = f'https://api.github.com/repos/{repo}'
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch from GitHub'}
    data = resp.json()
    return {'full_name': data['full_name'], 'description': data['description'], 'stars': data['stargazers_count']}

@plugin(name="pdf_summarizer", quantum_properties=["ai", "utility"],)
def pdf_summarizer_plugin(args):
    """Summarize a PDF file (base64-encoded, first page only, stub)"""
    pdf_b64 = args.get('pdf_b64')
    if not pdf_b64:
        return {'error': 'No PDF provided'}
    # In production, decode and use a PDF parser + LLM
    return {'summary': 'Stub: PDF summarization not implemented in demo.'}

@plugin(name="youtube_transcript_summarizer", quantum_properties=["ai", "external"])
def youtube_transcript_summarizer_plugin(args):
    """Summarize a YouTube video transcript (stub)"""
    video_id = args.get('video_id', 'dQw4w9WgXcQ')
    # In production, fetch transcript and summarize
    return {'summary': f'Stub: Summarized transcript for video {video_id}.'}

@plugin(name="news_headlines", quantum_properties=["external", "utility"])
def news_headlines_plugin(args):
    """Fetch latest news headlines (NewsAPI, requires API key)"""
    apikey = args.get('apikey', 'YOUR_API_KEY')
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={apikey}'
    resp = requests.get(url)
    if resp.status_code != 200:
        return {'error': 'Failed to fetch news'}
    data = resp.json()
    return {'headlines': [a['title'] for a in data.get('articles', [])[:5]]}

@plugin(name="math_solver", quantum_properties=["ai", "math"])
def math_solver_plugin(args):
    """Solve a math equation symbolically using SymPy"""
    eq = args.get('equation', 'x**2 - 4')
    x = sympy.symbols('x')
    try:
        sol = sympy.solve(eq, x)
        return {'solution': str(sol)}
    except Exception as e:
        return {'error': str(e)}

@plugin(name="ai_image_generator", quantum_properties=["ai", "image"])
def ai_image_generator_plugin(args):
    """Generate an image from a prompt (stub for DALL·E/SD)"""
    prompt = args.get('prompt', 'A cat riding a bicycle')
    return {'image_url': 'https://placekitten.com/400/300', 'note': f'Stub: Would generate image for prompt: {prompt}'}

@plugin(name="quantum_circuit_simulator", quantum_properties=["quantum", "simulation"])
def quantum_circuit_simulator_plugin(args):
    """Simulate a basic quantum circuit and return statevector (Qiskit required)"""
    if not QISKIT_AVAILABLE:
        return {'error': 'Qiskit not installed'}
    n = args.get('qubits', 2)
    qc = QuantumCircuit(n)
    qc.h(0)
    qc.cx(0, 1)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    statevector = result.get_statevector().tolist()
    return {'statevector': statevector, 'circuit': qc.qasm()}

@plugin(name="quantum_teleportation", quantum_properties=["quantum", "teleportation"])
def quantum_teleportation_plugin(args):
    """Simulate quantum teleportation of a qubit state (theoretical)"""
    state = args.get('state', '|0>')
    return {
        'result': f"Qubit state {state} teleported from Alice to Bob!",
        'explanation': "This plugin simulates quantum teleportation (no-cloning, entanglement, measurement)."
    }

@plugin(name="ai_text_to_speech", quantum_properties=["ai", "audio"])
def ai_text_to_speech_plugin(args):
    """Convert text to speech (stub, ready for TTS API)"""
    text = args.get('text', 'Hello, world!')
    return {
        'audio_url': 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
        'note': f'Stub: Would generate speech for: {text}'
    }

@plugin(name="ai_document_classifier", quantum_properties=["ai", "classification"])
def ai_document_classifier_plugin(args):
    """Classify a document into categories (stub, ready for LLM)"""
    doc = args.get('text', '')
    return {
        'category': 'science' if 'quantum' in doc.lower() else 'general',
        'note': 'Stub: Replace with LLM-based classifier.'
    }

@plugin(name="quantum_oracle", quantum_properties=["quantum", "oracle"])
def quantum_oracle_plugin(args):
    """Simulate a quantum oracle (returns a hidden function's output)"""
    secret = args.get('secret', 42)
    x = args.get('x', 0)
    return {'oracle_output': (x ^ secret)}

@plugin(name="ai_summarizer", quantum_properties=["ai", "summarization"])
def ai_summarizer_plugin(args):
    """Summarize a given text using LLM (stub)"""
    text = args.get('text', '')
    if not text:
        return {'summary': ''}
    # In production, call LLM API
    return {'summary': text[:100] + ('...' if len(text) > 100 else '')}

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

def run_plugin_chain(chain):
    """
    Run a chain of plugins. Each item: {tool: str, args: dict, input_key: str (optional)}
    If input_key is set, previous result[input_key] is merged into args.
    """
    results = []
    prev_result = None
    for step in chain:
        args = dict(step.get('args', {}))
        if prev_result and 'input_key' in step and step['input_key'] in prev_result:
            # Merge previous result into args
            args['input'] = prev_result[step['input_key']]
        result = run_tool(step['tool'], args)
        results.append({'tool': step['tool'], 'result': result})
        prev_result = result
    return results

def run_plugin_chain(chain):
    """
    Run a chain of plugins with advanced workflow logic.
    Each item: {tool: str, args: dict, input_key: str (optional), output_key: str (optional), condition: dict (optional), loop: int (optional)}
    - If input_key is set, previous result[input_key] is merged into args.
    - If output_key is set, result is stored under that key for later steps.
    - If condition is set, only run if previous result matches condition.
    - If loop is set, run the step multiple times (with index in args['i']).
    """
    results = []
    context = {}
    prev_result = None
    for step in chain:
        args = dict(step.get('args', {}))
        # Input mapping
        if prev_result and 'input_key' in step and step['input_key'] in prev_result:
            args['input'] = prev_result[step['input_key']]
        # Context mapping
        if 'use_context' in step and step['use_context'] in context:
            args['context'] = context[step['use_context']]
        # Conditional execution
        if 'condition' in step and prev_result:
            cond = step['condition']
            if not all(prev_result.get(k) == v for k, v in cond.items()):
                continue
        # Loop support
        loop = step.get('loop', 1)
        step_results = []
        for i in range(loop):
            if loop > 1:
                args['i'] = i
            result = run_tool(step['tool'], args)
            step_results.append(result)
            prev_result = result
        # Output mapping
        if 'output_key' in step:
            context[step['output_key']] = step_results[-1] if loop == 1 else step_results
        results.append({'tool': step['tool'], 'result': step_results if loop > 1 else step_results[0]})
    return results
