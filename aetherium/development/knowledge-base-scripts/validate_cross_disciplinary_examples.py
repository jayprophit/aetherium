# Validation Script: Cross-Disciplinary Example Checker

"""
This script validates that all .md files in the cross_disciplinary_examples directory:
- Contain a 'Machine-Readable Data' section with valid JSON
- Include at least one cross-link
- Have a summary and key concepts
- (NEW) Checks for additional categories: podcasts, webcomics, short stories, games, research papers

Usage:
    python validate_cross_disciplinary_examples.py
"""
import os
import re
import json

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '../resources/documentation/docs/cross_disciplinary_examples')

REQUIRED_SECTIONS = [
    'Summary',
    'Key Concepts',
    'Machine-Readable Data',
    'Cross-Links'
]

NEW_CATEGORIES = [
    'Podcast', 'Webcomic', 'Short Story', 'Game', 'Research Paper'
]

def validate_file(filepath: str) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    errors: list[str] = []
    for section in REQUIRED_SECTIONS:
        if section not in content:
            errors.append(f"Missing section: {section}")
    # Validate JSON
    match = re.search(r'```json\s*(\{[\s\S]+?\})\s*```', content)
    if not match:
        errors.append("Missing or invalid JSON block in 'Machine-Readable Data'")
    else:
        try:
            json.loads(match.group(1))
        except Exception as e:
            errors.append(f"Invalid JSON: {e}")
    # Check for at least one cross-link
    if 'Cross-Links' in content and not re.search(r'See ', content):
        errors.append("No cross-links found in 'Cross-Links' section")
    # Check for new categories in the title
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        for cat in NEW_CATEGORIES:
            if cat in first_line:
                break
        else:
            if any(cat.lower() in filepath.lower() for cat in [c.replace(' ', '_') for c in NEW_CATEGORIES]):
                errors.append("File appears to be a new category but does not mention it in the title")
    return errors

def main():
    failed = False
    for fname in os.listdir(EXAMPLES_DIR):
        if fname.endswith('.md'):
            path = os.path.join(EXAMPLES_DIR, fname)
            errors = validate_file(path)
            if errors:
                failed = True
                print(f"[FAIL] {fname}:")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"[PASS] {fname}")
    if failed:
        exit(1)
    else:
        print("All files passed validation.")

if __name__ == "__main__":
    main()
