import subprocess
import sys
import os
import re


def _bracket_depth(s):
    """Count bracket depth respecting strings but not ; comments."""
    depth = 0
    in_str = False
    j = 0
    while j < len(s):
        ch = s[j]
        if in_str:
            if ch == '\\' and j + 1 < len(s):
                j += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        j += 1
    return depth


def _is_expected_line(line):
    """Check if line is an expected-output directive (;=> or ;/)."""
    return line.startswith(';=>') or line.startswith(';/')


def _expected_value(line):
    """Extract expected value from ;=> or ;/ line. Returns (pattern, is_regex)."""
    if line.startswith(';=>'):
        return line[3:].strip(), False
    if line.startswith(';/'):
        return line[2:].strip(), True
    return None, False


def _collect_expected(lines, i):
    """Collect all ;/ and ;=> lines for one test. Returns (expected_parts, new_i).
    Each part is (value, is_regex). Multiple parts mean multi-line expected output."""
    parts = []
    while i < len(lines):
        eline = lines[i].rstrip('\r\n')
        if _is_expected_line(eline):
            val, is_rx = _expected_value(eline)
            parts.append((val, is_rx))
            i += 1
            # If this was ;=> (return value), we're done with this test's expected
            if eline.startswith('=>') or eline.startswith(';=>'):
                break
        elif not eline.strip() or (eline.startswith(';') and not _is_expected_line(eline)):
            i += 1
            continue
        else:
            break
    return parts, i


def parse_tests(test_file):
    """Parse MAL test file. Each non-blank, non-comment line is an input.
    Consecutive lines with unbalanced brackets form multi-line inputs.
    ;=> value = exact expected output.  ;/ pattern = regex expected output."""
    with open(test_file, 'r') as f:
        lines = f.readlines()

    inputs = []
    expected_outputs = []  # list of (pattern, is_regex) or None

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\r\n')

        # Skip blank lines and comment-only lines (not ;=> or ;/)
        if not line.strip() or (line.startswith(';') and not _is_expected_line(line)):
            i += 1
            continue

        # Collect input lines
        input_lines = []
        while i < len(lines):
            line = lines[i].rstrip('\r\n')

            # Blank line ends input
            if not line.strip():
                break
            # Expected output line ends input
            if _is_expected_line(line):
                break
            # Comment line (not ;=> / ;/) ends input if we already have content
            if line.startswith(';'):
                if input_lines:
                    break
                i += 1
                continue

            input_lines.append(line)
            i += 1

            # Check if expression is complete (depth=0)
            # or if next line is expected output (depth>0 but still complete)
            depth = _bracket_depth('\n'.join(input_lines))
            if depth <= 0:
                break

            # depth > 0: peek at next line to decide if we continue
            if i < len(lines):
                next_line = lines[i].rstrip('\r\n')
                # If next line is expected output, blank, or comment -> input is complete
                if (not next_line.strip() or _is_expected_line(next_line)
                        or next_line.startswith(';')):
                    break
                # Otherwise continue collecting (multi-line expression)
            else:
                break  # EOF

        if not input_lines:
            i += 1
            continue

        full_input = '\n'.join(input_lines)

        # Collect all expected output parts (multiple ;/ + one ;=>)
        parts, i = _collect_expected(lines, i)

        inputs.append(full_input)
        expected_outputs.append(parts if parts else None)

    return inputs, expected_outputs


if __name__ == '__main__':
    test_file = sys.argv[1] if len(sys.argv) > 1 else "tests/step0_repl.mal"
    cmd = sys.argv[2:] if len(sys.argv) > 2 else ["python", "step0_repl.py"]

    inputs, expected_outputs = parse_tests(test_file)

    # Strategy: pipe ALL inputs at once, then match outputs.
    # The program produces one output "block" per input.
    # Each block can have 0 or more lines.
    # We match expected outputs against blocks.

    # Handle readline: (readline ...) consumes the next input from stdin as data,
    # so it doesn't produce a separate program input. We need to:
    # 1. Keep the readline input (it produces output)
    # 2. The next input is readline data (sent to stdin but not a test input)
    # 3. Merge the data input's expected output onto the readline input
    # readline consumes the next stdin line as data, so the data line
    # doesn't produce separate output. We keep it in stdin_inputs for
    # piping but track display_inputs (1:1 with expected) for messages.
    stdin_inputs = []
    display_inputs = []
    filtered_expected = []
    skip_next = False
    for i in range(len(inputs)):
        if skip_next:
            skip_next = False
            continue
        stdin_inputs.append(inputs[i])
        if inputs[i].strip().startswith('(readline'):
            if i + 1 < len(inputs):
                skip_next = True
                # Readline reads raw stdin, not MAL syntax.
                # Strip surrounding quotes from MAL string literals.
                raw = inputs[i + 1]
                if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
                    raw = raw[1:-1]
                stdin_inputs.append(raw)
            display_inputs.append(inputs[i])
            if i + 1 < len(expected_outputs) and expected_outputs[i + 1] is not None:
                filtered_expected.append(expected_outputs[i + 1])
            else:
                filtered_expected.append(expected_outputs[i] if i < len(expected_outputs) else None)
        else:
            display_inputs.append(inputs[i])
            filtered_expected.append(expected_outputs[i] if i < len(expected_outputs) else None)

    inputs = stdin_inputs
    expected_outputs = filtered_expected

    input_text = '\n'.join(inputs) + '\n'
    proc = subprocess.run(cmd, input=input_text, capture_output=True, text=True,
                          cwd=os.path.dirname(os.path.abspath(__file__)) or '.')

    actual_output = proc.stdout
    actual_lines = actual_output.split('\n') if actual_output else []
    if actual_lines and actual_lines[-1] == '':
        actual_lines = actual_lines[:-1]

    passed = 0
    failed = 0
    actual_idx = 0

    for exp_idx, exp in enumerate(expected_outputs):
        if exp is None:
            # No expected output — consume one actual line (error/function output)
            if actual_idx < len(actual_lines):
                actual_idx += 1
            continue

        # Try to match starting at actual_idx
        num_parts = len(exp)

        # For multi-line regex patterns, try matching across multiple lines
        if len(exp) == 1 and exp[0][1] and '\\n' in exp[0][0]:
            pattern = exp[0][0]
            newline_count = pattern.count('\\n')
            found = False
            for try_n in range(newline_count + 1, newline_count + 15):
                end = min(actual_idx + try_n, len(actual_lines))
                chunk = '\n'.join(actual_lines[actual_idx:end])
                if re.search(pattern, chunk, re.DOTALL):
                    passed += 1
                    actual_idx += try_n
                    found = True
                    break
            if not found:
                failed += 1
                end = min(actual_idx + newline_count + 2, len(actual_lines))
                chunk = '\n'.join(actual_lines[actual_idx:end])
                print(f"[FAIL] {display_inputs[exp_idx][:60]}")
                print(f"  Expected: /{pattern}/")
                print(f"  Got: {chunk}")
                actual_idx += 1
            continue

        # Try exact match at current position
        if actual_idx + num_parts <= len(actual_lines):
            all_match = True
            for part_idx, (pattern, is_regex) in enumerate(exp):
                actual_line = actual_lines[actual_idx + part_idx]
                if is_regex:
                    if not re.search(pattern, actual_line):
                        all_match = False
                        break
                else:
                    if actual_line != pattern:
                        all_match = False
                        break
            if all_match:
                passed += 1
                actual_idx += num_parts
                continue

        # Exact match failed — try skipping 1 error line ahead
        if actual_idx + 1 + num_parts <= len(actual_lines):
            all_match = True
            for part_idx, (pattern, is_regex) in enumerate(exp):
                actual_line = actual_lines[actual_idx + 1 + part_idx]
                if is_regex:
                    if not re.search(pattern, actual_line):
                        all_match = False
                        break
                else:
                    if actual_line != pattern:
                        all_match = False
                        break
            if all_match:
                passed += 1
                actual_idx += 1 + num_parts
                continue

        failed += 1
        exp_str = ' / '.join(('/' + v + '/' if rx else v) for v, rx in exp)
        print(f"[FAIL] {display_inputs[exp_idx][:60]}")
        print(f"  Expected: {exp_str}")
        show_end = min(actual_idx + 3, len(actual_lines))
        got = '\n'.join(actual_lines[actual_idx:show_end])
        print(f"  Got: {got}")
        if actual_idx < len(actual_lines):
            actual_idx += 1

    total = passed + failed
    print(f"\nResults: {passed} passed, {failed} failed, {total} total")

    # Off-by-one detection
    if failed > 0 and len(actual_lines) > len(display_inputs):
        extra = len(actual_lines) - len(display_inputs)
        if extra <= 3:
            print(f"\nOFF-BY-ONE DETECTED: program printed {len(actual_lines)} output lines for {len(display_inputs)} tests ({extra} extra). "
                  f"The program is printing an extra line somewhere (blank line, error to stdout, or print on EOF). "
                  f"Fix: ensure the REPL only prints ONE line per input and NOTHING on EOF.")
