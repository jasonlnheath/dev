"""Patch Hermes config.yaml with critical fixes.

Adds model.max_tokens (prevents write_file truncation at 4096 tokens),
enables compression, sets context engine to priority_compressor,
and configures optimal compression thresholds.
"""
import sys

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

    # 2. Add/update context engine config
    if 'context:' not in content or 'engine: priority_compressor' not in content:
        if 'context:' in content:
            old = content[content.index('context:'):content.index('context:')+50]
            if 'engine:' not in old:
                content = content.replace(
                    'context:\n',
                    'context:\n  engine: priority_compressor\n  threshold_tokens: 75000\n  tail_budget_pct: 0.2\n',
                    1
                )
                changes.append('Added context.engine: priority_compressor')
        else:
            content = content.replace(
                'memory:\n',
                'context:\n  engine: priority_compressor\n  threshold_tokens: 75000\n  tail_budget_pct: 0.2\n\nmemory:\n',
                1
            )
            changes.append('Added context section with priority_compressor')
    else:
        # Update existing threshold to 75K if it's higher
        if 'threshold_tokens: 95000' in content:
            content = content.replace('threshold_tokens: 95000', 'threshold_tokens: 75000')
            changes.append('Lowered threshold_tokens: 95000 -> 75000')
        if 'tail_budget_pct: 0.1' in content:
            content = content.replace('tail_budget_pct: 0.1', 'tail_budget_pct: 0.2')
            changes.append('Increased tail_budget_pct: 0.1 -> 0.2')

    # 3. Enable compression if disabled
    if 'compression:' in content:
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
