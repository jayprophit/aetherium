import os
import re
import requests
from urllib.parse import urlparse
from pathlib import Path

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def check_links_in_file(file_path, root_dir):
    broken_links = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all markdown links [text](url)
    for match in re.finditer(r'\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)', content):
        url = match.group('url').split(' ')[0]  # Handle [text](url "title")
        
        # Skip external URLs
        if is_valid_url(url):
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                if response.status_code >= 400:
                    broken_links.append((match.group(0), f"HTTP {response.status_code}"))
            except Exception as e:
                broken_links.append((match.group(0), str(e)))
        # Check local file references
        elif not url.startswith(('http://', 'https://', '#')):
            # Handle relative paths
            if url.startswith('/'):
                target_path = Path(root_dir) / url[1:]
            else:
                target_path = Path(file_path).parent / url
            
            # Normalize path
            target_path = target_path.resolve()
            
            # Check if file exists
            if not target_path.exists():
                broken_links.append((match.group(0), f"File not found: {target_path}"))
    
    return broken_links

def check_all_links(root_dir):
    broken_links = {}
    
    for root, _, files in os.walk(root_dir):
        # Skip certain directories
        if any(skip_dir in root for skip_dir in ['.git', 'node_modules', '__pycache__', '.venv']):
            continue
            
        for file in files:
            if file.endswith(('.md', '.mdx', '.markdown')):
                file_path = os.path.join(root, file)
                broken = check_links_in_file(file_path, root_dir)
                if broken:
                    broken_links[file_path] = broken
    
    return broken_links

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    broken_links = check_all_links(root_dir)
    
    if broken_links:
        print("\nBroken links found:")
        print("-" * 80)
        for file_path, links in broken_links.items():
            rel_path = os.path.relpath(file_path, root_dir)
            print(f"\nFile: {rel_path}")
            for link, error in links:
                print(f"  - {link}")
                print(f"    {error}")
        print("\n" + "-" * 80)
        print(f"Found {sum(len(links) for links in broken_links.values())} broken links in {len(broken_links)} files.")
        exit(1)
    else:
        print("No broken links found!")
        exit(0)
