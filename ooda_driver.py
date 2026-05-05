#!/usr/bin/env python
"""
OODA Driver v6 - Claude-hinted, Qwen-written.

No GLM-5.1 calls. Claude provides diagnosis via hint file, Qwen writes code.
Flow:
  1. Qwen generates initial code (greenfield)
  2. Run test → save failure context to .ooda_hint_request.md
  3. Wait for Claude to write .ooda_hint.md
  4. Read hint, call Qwen with it
  5. Repeat until pass

Usage:
  python ooda_driver.py --project DIR --test "CMD" --task "DESC"
"""

import subprocess, sys, os, re, json, time, shutil, difflib, urllib.request, urllib.error
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

# --- Defaults ---

QWEN_URL = "http://localhost:8033/v1"
QWEN_MODEL = "local-model"
TEMP_THINK = 0.6   # Qwen3 official recommendation for thinking mode
TEMP_WRITE = 0.7   # Qwen3 official recommendation for non-thinking mode
MAX_TOKENS = 8192
MAX_THINK_TOKENS = 32768
THINK_BUDGET = 8192  # per-request thinking cap (leaves room for content)
MAX_DIFF_LINES = 200
SMALL_DIFF_THRESHOLD = 30  # small fixes to protected files are always allowed

HINT_REQUEST_FILE = ".ooda_hint_request.md"
HINT_FILE = ".ooda_hint.md"

# --- Logging ---

_log_file = None
_token_log = []

def log(action, msg, step_label=""):
    global _log_file
    ts = datetime.now().isoformat()
    entry = {"timestamp": ts, "action": action, "message": msg}
    if step_label:
        entry["step"] = step_label
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    label = f"Step {step_label} | " if step_label else ""
    safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
    print(f"[{ts[11:19]}] {label}{action} | {safe_msg}", flush=True)


def log_tokens(model, resp_data):
    global _token_log
    if "usage" in resp_data:
        u = resp_data["usage"]
        inp = u.get("prompt_tokens", u.get("input_tokens", 0))
        out = u.get("completion_tokens", u.get("output_tokens", 0))
        _token_log.append((model, inp, out))
        log("tokens", f"{model}: {inp} in + {out} out = {inp+out} total")
        return inp, out
    return 0, 0


# --- API ---

def call_qwen(prompt, qwen_url, thinking=False, think_budget=None):
    max_tok = MAX_THINK_TOKENS if thinking else MAX_TOKENS
    temp = TEMP_THINK if thinking else TEMP_WRITE
    body = {
        "model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, "max_tokens": max_tok,
        "top_k": 20, "top_p": 0.95, "min_p": 0,
    }
    if thinking:
        body["chat_template_kwargs"] = {"enable_thinking": True}
        budget = think_budget if think_budget is not None else THINK_BUDGET
        body["thinking_budget_tokens"] = budget
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{qwen_url}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    model_label = f"{QWEN_MODEL}+think" if thinking else QWEN_MODEL
    log_tokens(model_label, data)
    msg = data["choices"][0]["message"]
    content = msg.get("content", "") or ""
    # If thinking consumed all tokens and left no content, use the reasoning as fallback
    if thinking and not content.strip():
        reasoning = msg.get("reasoning_content", "") or ""
        if reasoning.strip():
            # Take the last ~2000 chars of reasoning (the conclusion/summary part)
            content = reasoning[-2000:]
            log("think_fallback", f"Empty content, using last 2000 of {len(reasoning)} chars of reasoning")
    Path(".qwen_last_response.txt").write_text(content, encoding="utf-8")
    return content


# --- Failure grouping ---

def group_failures(output, max_chars=1500):
    failures = []
    for l in output.split("\n"):
        l = l.strip()
        if any(kw in l for kw in ["FAIL", "Error", "Expected", "Got", "MISSING", "assert"]):
            failures.append(l)
    if not failures:
        return "(no failures found in output)"
    groups = Counter()
    for f in failures:
        key = re.sub(r'\[FAIL\]\s*\S+', '[FAIL] <test>', f)
        key = re.sub(r'line \d+', 'line N', key)
        groups[key] += 1
    lines = []
    for pattern, count in groups.most_common(20):
        if count > 1:
            lines.append(f"[{count}x] {pattern}")
        else:
            lines.append(pattern)
    result = "\n".join(lines)[:max_chars]
    total = sum(groups.values())
    return f"{total} failures ({len(groups)} distinct patterns):\n{result}"


def extract_raw_failures(output, max_chars=2000):
    lines = output.split("\n")
    result = []
    i = 0
    while i < len(lines):
        if "[FAIL]" in lines[i]:
            block = [lines[i].strip()]
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith("Expected") or lines[j].strip().startswith("Got") or lines[j].strip() == ""):
                if lines[j].strip():
                    block.append(lines[j].strip())
                j += 1
            result.append("\n".join(block))
            i = j
        else:
            i += 1
    text = "\n\n".join(result)
    return text[:max_chars]


# --- Code extraction with diff guard ---

def _clean_code(text):
    """Strip trailing markdown fences and whitespace."""
    text = text.strip()
    text = re.sub(r'\n```\w*\s*$', '', text)
    return text

def extract_code_blocks(response):
    files = {}
    for m in re.finditer(r"(?:#?\s*FILE:|###?)\s*([\w.-]+\.\w+)\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
        files[m.group(1)] = _clean_code(m.group(2))
    for m in re.finditer(r"([\w.-]+\.\w+)\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
        if m.group(1) not in files:
            files[m.group(1)] = _clean_code(m.group(2))
    for m in re.finditer(r"\*\*([\w.-]+\.\w+)\*\*\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
        if m.group(1) not in files:
            files[m.group(1)] = _clean_code(m.group(2))
    if not files:
        for m in re.finditer(r"#?\s*FILE:\s*([\w.-]+\.\w+)\s*\n(.*?)(?=#?\s*FILE:|$)", response, re.DOTALL):
            files[m.group(1)] = _clean_code(m.group(2))
        if files:
            log("extract_nofence", f"Extracted {len(files)} files without code fences")
    if not files:
        for m in re.finditer(r"# FILE:\s*[^\s]*[\\/]([\w.-]+\.\w+)\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
            if m.group(1) not in files:
                files[m.group(1)] = _clean_code(m.group(2))
    return files


def diff_line_count(old_text, new_text):
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, n=0))
    changed = sum(1 for d in diff if d.startswith(('+', '-')) and not d.startswith(('+++', '---')))
    return changed


def extract_and_write(response, project_dir, primary_file=None, protected_files=None):
    files = extract_code_blocks(response)
    if not files:
        blocks = re.findall(r"```[\w]*\n(.*?)```", response, re.DOTALL)
        if blocks:
            log("extract_fallback", "No named files found in response")
            return []
        log("extract_empty", "No code blocks found in response")
        return []
    written = []
    rejected = []
    for name, new_code in files.items():
        dest = project_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        is_protected = protected_files and name in protected_files
        if dest.exists() and name != primary_file and is_protected:
            old_text = dest.read_text(encoding="utf-8")
            changed = diff_line_count(old_text, new_code + "\n")
            if changed <= SMALL_DIFF_THRESHOLD:
                log("diff_small", f"ALLOWED small change to {name}: {changed} lines", "")
            elif changed > MAX_DIFF_LINES:
                log("diff_guard", f"REJECTED {name}: {changed} lines changed (max {MAX_DIFF_LINES})", "")
                rejected.append(name)
                continue
        dest.write_text(new_code + "\n", encoding="utf-8")
        written.append(name)
    if rejected:
        log("diff_guard_summary", f"Rejected {len(rejected)} files: {rejected}", "")
    return written


# --- Test running ---

def _kill_tree(pid):
    try:
        subprocess.run(f"taskkill /PID {pid} /T /F", shell=True, capture_output=True, timeout=5)
    except Exception:
        pass


def run_test(test_cmd, project_dir, timeout=60):
    proc = None
    try:
        proc = subprocess.Popen(test_cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                cwd=str(project_dir),
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            return 0, 1, 1, f"TIMEOUT after {timeout}s"
        output = stdout + "\n" + stderr
        m = re.search(r"(\d+) passed, (\d+) failed(?:, (\d+) total)?", output)
        if m:
            p, f = int(m.group(1)), int(m.group(2))
            return p, f, p + f, output
        if proc.returncode == 0:
            return 1, 0, 1, output
        else:
            return 0, 1, 1, output
    except Exception as e:
        if proc:
            _kill_tree(proc.pid)
        return 0, 1, 1, str(e)


# --- Read project code ---

def read_project_code(project_dir, file_glob="*.py"):
    parts = []
    for p in sorted(project_dir.glob(file_glob)):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else "(no files)"


def read_targeted_code(project_dir, file_glob="*.py", failure_output="", primary_file=None):
    all_files = sorted(p for p in project_dir.glob(file_glob)
                       if not p.name.startswith(".") and not p.name.startswith("_"))
    if not failure_output:
        return read_project_code(project_dir, file_glob)
    mentioned = set()
    for m in re.finditer(r'([\w.-]+\.\w+)', failure_output):
        mentioned.add(m.group(1))
    parts = []
    included = set()
    for p in all_files:
        if p.name == primary_file or p.name in mentioned:
            parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
            included.add(p.name)
    if len(included) < 2:
        for p in all_files:
            if p.name not in included:
                mtime = p.stat().st_mtime
                if time.time() - mtime < 300:
                    parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
                    included.add(p.name)
    if len(included) < 2:
        return read_project_code(project_dir, file_glob)
    return "\n\n".join(parts) if parts else "(no files)"


# --- Checkpointing ---

def save_checkpoint(cp_dir, project_dir, file_glob="*.py"):
    cp_dir.mkdir(parents=True, exist_ok=True)
    for f in project_dir.glob(file_glob):
        if f.name.startswith(".") or f.name.startswith("_"):
            continue
        shutil.copy2(f, cp_dir / f.name)


def restore_checkpoint(cp_dir, project_dir):
    if not cp_dir.exists():
        return False
    for f in cp_dir.glob("*"):
        if f.is_file():
            shutil.copy2(f, project_dir / f.name)
    return True


# --- Hint file management ---

def write_hint_request(project_dir, iteration, passed, total, output, code, task):
    """Write failure context for Claude to analyze."""
    request_path = project_dir / HINT_REQUEST_FILE
    raw_failures = extract_raw_failures(output)
    failure_summary = group_failures(output)

    content = f"""# Hint Request — Iteration {iteration}

## Score: {passed}/{total} passed

## Task
{task}

## Raw Failures (Expected vs Got)
{raw_failures}

## Grouped Failures
{failure_summary}

## Current Code
{code}
"""
    request_path.write_text(content, encoding="utf-8")
    return request_path


def wait_for_hint(project_dir, poll_interval=2):
    """Wait for Claude to write .ooda_hint.md, then read and delete it.

    Also detects external file edits (e.g., Claude editing directly) and returns
    EXTERNAL_EDIT sentinel so the driver re-tests without calling Qwen.
    """
    hint_path = project_dir / HINT_FILE
    request_path = project_dir / HINT_REQUEST_FILE

    # Snapshot file mtimes to detect external edits
    def get_file_state():
        state = {}
        for f in project_dir.glob("*.py"):
            if not f.name.startswith(".") and not f.name.startswith("_"):
                try:
                    state[f.name] = f.stat().st_mtime
                except OSError:
                    pass
        return state

    initial_state = get_file_state()
    log("hint_wait", f"Waiting for {HINT_FILE}... (write your hint to {hint_path})")
    while not hint_path.exists():
        time.sleep(poll_interval)
        # Detect external file changes (Claude direct-edit escalation)
        current_state = get_file_state()
        if current_state != initial_state:
            changed = [k for k in current_state if current_state.get(k) != initial_state.get(k)]
            log("external_edit", f"Detected external edit to {changed}, re-testing", "")
            if request_path.exists():
                request_path.unlink()
            return "__EXTERNAL_EDIT__"
    # Wait a bit for file to be fully written
    time.sleep(0.5)
    hint = hint_path.read_text(encoding="utf-8")
    hint_path.unlink()
    if request_path.exists():
        request_path.unlink()
    log("hint_received", f"Got hint ({len(hint)} chars)")
    return hint


# --- Build prompts ---

def build_initial_prompt(task, context, project_dir, primary_file=None, base_file_content=None):
    if base_file_content:
        p = f"You are modifying an existing working program. ADD the following features:\n\n{task}"
        if primary_file:
            p += f"\n\nThe output file MUST be named exactly: {primary_file}"
        if context:
            p += f"\n\nReference material:\n{context}"
        p += f"\n\n# HERE IS THE CURRENT WORKING CODE TO MODIFY:\n{base_file_content}"
        p += ("\n\nCRITICAL RULES FOR INCREMENTAL DEVELOPMENT:"
              "\n- Output the COMPLETE modified file with the new features ADDED"
              "\n- Do NOT remove any existing functionality"
              "\n- Do NOT rewrite from scratch — keep all existing code and ADD new features"
              "\n- Preserve ALL existing functions, classes, imports, and logic"
              "\n- Format: # FILE: filename.ext followed by a code block")
        return p
    existing = sorted(p.name for p in project_dir.glob("*.py")
                      if not p.name.startswith(".") and not p.name.startswith("_"))
    existing_str = ", ".join(existing) if existing else "none"
    p = f"Implement the following:\n\n{task}\n\nWorking directory: {project_dir}"
    if primary_file:
        p += f"\n\nThe output file MUST be named exactly: {primary_file}"
    if context:
        p += f"\n\nContext/Reference:\n{context}"
    p += f"\n\nIMPORTANT: These files already exist and MUST NOT be overwritten: [{existing_str}]"
    p += "\nOnly write NEW files that don't exist yet. Format: # FILE: filename.ext followed by a code block."
    return p


def build_hint_prompt(task, code, hint, primary_file=None, near_completion=False):
    if near_completion:
        prefix = ("WARNING: You are VERY close to passing all tests. "
                  "Make a MINIMAL, TARGETED fix. Do NOT rewrite large sections. "
                  "Do NOT change code that is already working. "
                  "Only fix the specific failing test.\n\n")
    else:
        prefix = ""
    p = (
        f"{prefix}"
        f"EXPERT DIAGNOSIS (apply this fix):\n{hint}\n\n"
        f"---\n\n"
        f"Task: {task}\n\n"
        f"Current code:\n{code}\n\n"
        f"REMINDER — apply this fix:\n{hint}\n\n"
        f"Output ONLY the files that need fixing. "
        f"Format: # FILE: filename.ext + code block."
    )
    if primary_file:
        p += f"\n\nThe output file MUST be named exactly: {primary_file}"
    return p


def build_diagnosis_prompt(task, passed, total, raw_failures, grouped_failures, code):
    crash_note = "\nNOTE: The test process crashed (0 tests ran). Focus on import errors, wrong API names, and syntax issues first." if total == 0 else ""
    return (
        "You are a bug diagnosis expert. Read the test failures and current code below.\n"
        "Identify the 1-2 root causes. Be surgical: point to exact file/line, give the fix pattern.\n"
        "Do NOT rewrite the code. Only provide diagnosis. Be brief and direct.\n\n"
        f"Task: {task}\n"
        f"Test results: {passed}/{total} passed\n"
        f"{crash_note}\n"
        f"Raw Failures:\n{raw_failures}\n\n"
        f"Grouped Failures:\n{grouped_failures}\n\n"
        f"Current Code:\n{code}\n\n"
        "Write 1-2 bullet points identifying the root cause and fix pattern."
    )

def build_fast_fix_hint(task, passed, total, output):
    """Build a lightweight hint for first-iteration fast retry (no thinking needed)."""
    if total == 0:
        failures = f"CRASH — no tests ran. Full output:\n{output[:2000]}"
    else:
        failures = extract_raw_failures(output)[:2000]
    return (
        f"Test results: {passed}/{total} passed.\n"
        f"Failures:\n{failures}\n\n"
        f"Fix the code to make the tests pass."
    )


def self_hint(project_dir, qwen_url, iteration, passed, total, output, code, task, step_label=""):
    if total == 0:
        # Test crashed entirely (import error, syntax error, etc.)
        raw_failures = f"CRASH — no tests ran. Full output:\n{output[:3000]}"
        grouped_failures = "(test process crashed before running any tests)"
    else:
        raw_failures = extract_raw_failures(output)
        grouped_failures = group_failures(output)
    prompt = build_diagnosis_prompt(task, passed, total, raw_failures, grouped_failures, code)
    log("self_hint", f"Calling Qwen+thinking for diagnosis (iteration {iteration})...", step_label)
    hint = call_qwen(prompt, qwen_url, thinking=True)
    # Save hint for debugging
    hint_path = project_dir / HINT_FILE
    hint_path.write_text(hint, encoding="utf-8")
    log("self_hint_done", f"Got diagnosis ({len(hint)} chars)", step_label)
    return hint


def self_escalate(project_dir, qwen_url, iteration, passed, total,
                  output, code, task, rollbacks, step_label=""):
    if total == 0:
        raw_failures = f"CRASH — no tests ran. Full output:\n{output[:3000]}"
        grouped_failures = "(test process crashed before running any tests)"
    else:
        raw_failures = extract_raw_failures(output)
        grouped_failures = group_failures(output)
    prompt = (
        "You are a senior developer doing a SECOND REVIEW of a bug that has resisted "
        f"{rollbacks} fix attempts. Previous fixes failed because they misidentified the root cause.\n\n"
        "Think step by step:\n"
        "1. What pattern do ALL the failures share?\n"
        "2. What did previous fixes likely get wrong?\n"
        "3. What is the ACTUAL root cause (not the symptom)?\n\n"
        f"Task: {task}\n"
        f"Results: {passed}/{total} passed (stuck for {rollbacks} attempts)\n"
        f"Failures:\n{raw_failures}\n\n"
        f"Patterns:\n{grouped_failures}\n\n"
        f"Current code:\n{code}\n\n"
        "Give 1-2 bullet points with the EXACT fix. Be more specific than a first-pass diagnosis. "
        "If needed, include a small code snippet showing the fix pattern."
    )
    log("self_escalate", f"Qwen re-diagnosing (stuck {rollbacks}x, iteration {iteration})...", step_label)
    hint = call_qwen(prompt, qwen_url, thinking=True)
    hint_path = project_dir / HINT_FILE
    hint_path.write_text(hint, encoding="utf-8")
    log("self_escalate_done", f"Got enhanced diagnosis ({len(hint)} chars)", step_label)
    return hint


# --- Main OODA loop ---

def ooda_run(args):
    global _log_file
    project_dir = Path(args.project).resolve()
    cp_dir = project_dir / ".checkpoints" / "latest"
    _log_file = project_dir / "ooda_log.jsonl"

    step_label = args.step_label or ""
    primary_file = args.primary_file or None

    print("=" * 60)
    mode = "self-hint (Qwen diagnoses + writes)" if args.self_hint else "Claude-hinted, Qwen-written"
    print(f"OODA Driver v7 ({mode})")
    print(f"Project: {project_dir}")
    print(f"Test: {args.test}")
    print(f"Task: {args.task[:80]}...")
    print(f"Primary file: {primary_file or '(none)'}")
    if args.self_hint:
        esc = f", escalates to Claude after {args.stuck_threshold} rollbacks" if args.stuck_threshold > 0 else ", no escalation"
        print(f"Loop: unlimited (Qwen diagnoses{esc})")
    else:
        print(f"Loop: unlimited (Claude hints, Qwen writes)")
    print("=" * 60, flush=True)

    # Pre-flight: check Qwen
    use_qwen = not args.no_qwen
    if use_qwen:
        try:
            req = urllib.request.Request(f"{args.qwen_url}/models")
            urllib.request.urlopen(req, timeout=5)
            log("preflight", "Qwen: OK")
        except Exception as e:
            log("preflight", f"Qwen not responding: {e}. Aborting.")
            return False

    best_passed, best_failed = -1, float("inf")
    consecutive_rollbacks = 0
    escalated = False

    # Diff guard
    protected_files = set()
    driver_written = set()
    for p in project_dir.glob(args.file_glob):
        if not p.name.startswith(".") and not p.name.startswith("_"):
            protected_files.add(p.name)
    log("preflight", f"Protected files: {sorted(protected_files) or 'none'}", step_label)

    def get_protected():
        return protected_files - driver_written

    # Clean up any stale hint files
    for f in [HINT_REQUEST_FILE, HINT_FILE]:
        p = project_dir / f
        if p.exists():
            p.unlink()

    # Phase 1: Qwen initial generation
    base_file_content = None
    if args.base_file:
        base_path = project_dir / args.base_file
        if base_path.exists():
            base_file_content = base_path.read_text(encoding="utf-8")
            if primary_file and not (project_dir / primary_file).exists():
                shutil.copy2(str(base_path), str(project_dir / primary_file))
                log("base_copy", f"Copied {args.base_file} -> {primary_file} as starting point", step_label)
            else:
                log("base_skip_copy", f"Primary file {primary_file} already exists or no primary_file set", step_label)
        else:
            log("base_missing", f"Base file {args.base_file} not found", step_label)

    if use_qwen:
        if primary_file and (project_dir / primary_file).exists() and not base_file_content:
            log("skip_qwen", f"{primary_file} already exists, skipping Qwen")
            use_qwen = False
        else:
            log("qwen", "Qwen generating initial code..." + (" (incremental from base)" if base_file_content else ""), step_label)
            try:
                prompt = build_initial_prompt(args.task, args.context, project_dir, primary_file, base_file_content)
                response = call_qwen(prompt, args.qwen_url)
                log("qwen_raw", f"Response preview: {repr(response[:200])}", step_label)
                files = extract_and_write(response, project_dir, primary_file, get_protected())
                driver_written.update(files)
                log("qwen_done", f"Qwen wrote {len(files)} files: {files}", step_label)
            except Exception as e:
                log("qwen_err", f"Qwen error: {e}", step_label)

    # Main loop: test → diagnose → fix → repeat
    # Iteration 1: non-thinking fast retry (cheap, catches syntax/import errors)
    # Iteration 2+: thinking diagnosis → non-thinking fix (deep reasoning for real bugs)
    iteration = 0
    while True:
        iteration += 1

        # OBSERVE: Run tests
        passed, failed, total, output = run_test(args.test, project_dir, timeout=args.timeout)
        log("test", f"{passed}/{total} passed", step_label)

        # Check pass
        if failed == 0 and total > 0:
            save_checkpoint(cp_dir, project_dir, args.file_glob)
            log("pass", f"All passed on iteration {iteration}", step_label)
            # Clean up
            for f in [HINT_REQUEST_FILE, HINT_FILE]:
                p = project_dir / f
                if p.exists():
                    p.unlink()
            return True

        # Checkpoint or rollback (never rollback to 0-passing code)
        if passed > best_passed or (passed == best_passed and failed < best_failed):
            best_passed, best_failed = passed, failed
            if passed > 0:
                save_checkpoint(cp_dir, project_dir, args.file_glob)
            consecutive_rollbacks = 0
            if escalated:
                log("escalate_reset", "Claude hint helped, returning to self-hint", step_label)
                escalated = False
        else:
            # No improvement — count as stuck regardless of best_passed
            consecutive_rollbacks += 1
            if best_passed > 0 and restore_checkpoint(cp_dir, project_dir):
                log("rollback", f"Rolled back (no improvement, was {best_passed}/{best_passed+best_failed}) [{consecutive_rollbacks} consecutive]", step_label)
            else:
                log("no_improvement", f"No improvement, {consecutive_rollbacks} consecutive non-improvements (best: {best_passed}/{best_passed+best_failed})", step_label)
            if (args.self_hint and not escalated
                    and args.stuck_threshold > 0
                    and consecutive_rollbacks >= args.stuck_threshold):
                escalated = True
                log("escalate", f"Stuck after {consecutive_rollbacks} non-improvements, escalating to Claude", step_label)

        # ORIENT: Build failure context
        code = read_targeted_code(project_dir, args.file_glob, output, primary_file)

        # DECIDE: Get diagnosis
        if args.self_hint and not escalated:
            if iteration == 1:
                # First iteration: skip diagnosis, let non-thinking Qwen try to fix directly
                hint = build_fast_fix_hint(task=args.task, passed=passed, total=total, output=output)
                log("fast_retry", "First iteration: non-thinking fast retry (no diagnosis)", step_label)
            else:
                hint = self_hint(project_dir, args.qwen_url, iteration, passed, total, output, code, args.task, step_label)
        elif args.self_hint and escalated:
            if args.self_escalate:
                hint = self_escalate(project_dir, args.qwen_url, iteration,
                                     passed, total, output, code, args.task,
                                     consecutive_rollbacks, step_label)
            else:
                # Delete stale hint from self_hint() so wait_for_hint() waits for Claude
                (project_dir / HINT_FILE).unlink(missing_ok=True)
                request_path = write_hint_request(project_dir, iteration, passed, total, output, code, args.task)
                log("hint_request", f"Wrote {HINT_REQUEST_FILE} for Claude ({request_path.stat().st_size} bytes)", step_label)
                hint = wait_for_hint(project_dir)
        else:
            request_path = write_hint_request(project_dir, iteration, passed, total, output, code, args.task)
            log("hint_request", f"Wrote {HINT_REQUEST_FILE} ({request_path.stat().st_size} bytes)", step_label)
            hint = wait_for_hint(project_dir)

        # ACT: Call Qwen with hint (unless Claude already edited files directly)
        if hint == "__EXTERNAL_EDIT__":
            log("skip_qwen", "Skipping Qwen fix, files were edited externally", step_label)
        else:
            if args.self_hint and not escalated and iteration == 1:
                source = "fast-retry (non-thinking)"
            elif args.self_hint and not escalated:
                source = "self-hint (thinking→fix)"
            elif args.self_hint and escalated:
                source = "self-escalate" if args.self_escalate else "Claude hint (escalated)"
            else:
                source = "Claude's hint"
            log("qwen_fix", f"Qwen fixing with {source} (iteration {iteration})...", step_label)
            try:
                # Re-read code (may have changed from rollback)
                code = read_targeted_code(project_dir, args.file_glob, output, primary_file)
                near_completion = total > 0 and passed > 0 and (passed / total) > 0.9
                fix_prompt = build_hint_prompt(args.task, code, hint, primary_file, near_completion)
                response = call_qwen(fix_prompt, args.qwen_url)
                files = extract_and_write(response, project_dir, primary_file, get_protected())
                driver_written.update(files)
                log("fix_done", f"Qwen wrote {len(files)} files: {files}", step_label)
            except Exception as e:
                log("qwen_err", f"Qwen error: {e}", step_label)

        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description="OODA Driver v7 (fast-first, thinking-escalate)")
    p.add_argument("--project", required=True, help="Project directory")
    p.add_argument("--test", required=True, help="Test command to run")
    p.add_argument("--task", required=True, help="Task description (what to implement)")
    p.add_argument("--context", default="", help="Additional context (reference code, etc)")
    p.add_argument("--context-file", default="", help="File to read context from")
    p.add_argument("--file-glob", default="*.py", help="File pattern for checkpointing")
    p.add_argument("--max-retries", type=int, default=6, help="(ignored)")
    p.add_argument("--timeout", type=int, default=60, help="Test timeout in seconds")
    p.add_argument("--qwen-url", default=QWEN_URL, help="Qwen API URL")
    p.add_argument("--no-qwen", action="store_true", help="Skip Qwen initial generation")
    p.add_argument("--step-label", default="", help="Label for log messages")
    p.add_argument("--system-extra", default="", help="(ignored)")
    p.add_argument("--primary-file", default="", help="Main file (exempt from diff guard)")
    p.add_argument("--self-hint", action="store_true", help="Qwen diagnoses failures (thinking mode) instead of Claude")
    p.add_argument("--self-escalate", action="store_true",
                   help="When stuck, use Qwen+thinking with enhanced prompt instead of waiting for Claude")
    p.add_argument("--stuck-threshold", type=int, default=1, help="Consecutive rollbacks before escalating to Claude (0=never)")
    p.add_argument("--base-file", default="", help="File to use as starting point (copied to primary-file, uses incremental prompt)")
    args = p.parse_args()

    if args.context_file and not args.context:
        p = Path(args.context_file)
        if p.exists():
            args.context = p.read_text(encoding="utf-8")

    success = ooda_run(args)

    print("\n" + "=" * 60)
    print("TOKEN USAGE (Qwen only — zero cloud)")
    print("=" * 60)
    by_model = {}
    for model, inp, out in _token_log:
        if model not in by_model:
            by_model[model] = {"calls": 0, "input": 0, "output": 0}
        by_model[model]["calls"] += 1
        by_model[model]["input"] += inp
        by_model[model]["output"] += out
    total_input = 0
    total_output = 0
    for model, usage in sorted(by_model.items()):
        total_input += usage["input"]
        total_output += usage["output"]
        print(f"  {model:30s}: {usage['calls']} calls, {usage['input']:>8,} in, {usage['output']:>8,} out")
    print(f"  {'TOTAL':30s}: {len(_token_log)} calls, {total_input:>8,} in, {total_output:>8,} out, {total_input+total_output:>8,} total")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
