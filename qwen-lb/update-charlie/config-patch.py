"""Patch Hermes config.yaml with critical fixes.

Adds model.max_tokens (prevents write_file truncation at 4096 tokens),
enables compression, and sets context engine to priority_compressor.
"""
import sys
import re

def patch_config(path):
    with open(path, 'r') as f:
        content = f.read()

    changes = []

    # 1. Add max_tokens to model section if missing
    if 'model:' in content and 'max_tokens' not in content.split('providers:')[0]:
        content = content.replace(
            'context_length: 131072',
            'context_length: 131072\n  max_tokens: 32768',
        )
        changes.append('Added model.max_tokens: 32768')

    # 2. Add context engine config if missing
    if 'context:' not in content or 'engine: priority_compressor' not in content:
        if 'context:' in content:
            # Replace existing context section header
            old = content[content.index('context:'):content.index('context:')+50]
            if 'engine:' not in old:
                content = content.replace(
                    'context:\n',
                    'context:\n  engine: priority_compressor\n  threshold_tokens: 95000\n',
                    1
                )
                changes.append('Added context.engine: priority_compressor')
        else:
            content = content.replace(
                'memory:\n',
                'context:\n  engine: priority_compressor\n  threshold_tokens: 95000\n\nmemory:\n',
                1
            )
            changes.append('Added context section with priority_compressor')

    # 3. Enable compression if disabled
    if 'compression:' in content:
        # Find compression section and ensure enabled: true
        comp_section = content[content.index('compression:'):]
        comp_header = comp_section[:comp_section.index('\n') + 1]
        rest = comp_section[comp_section.index('\n') + 1:]

        if 'enabled: false' in rest[:200]:
            rest = rest.replace('enabled: false', 'enabled: true', 1)
            content = content[:content.index('compression:')] + comp_header + rest
            changes.append('Enabled compression')

    with open(path, 'w') as f:
        f.write(content)

    if changes:
        print(f'Patched {path}:')
        for c in changes:
            print(f'  - {c}')
    else:
        print(f'{path}: no changes needed')

if __name__ == '__main__':
    patch_config(sys.argv[1] if len(sys.argv) > 1 else sys.exit('Usage: config-patch.py <config.yaml>'))
