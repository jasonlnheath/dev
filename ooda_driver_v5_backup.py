#!/usr/bin/env python
"""
OODA Driver v5 - Generic autonomous development driver.

TIER 1: Qwen (local) generates initial code (greenfield only)
TIER 1b: Qwen gets one retry with raw test hint
TIER 2: GLM-5.1 diagnoses → Qwen fixes (3 rounds, saves cloud output tokens)
TIER 3: GLM-5.1 diagnoses + fixes directly

v5: Context file accumulates learnings (purged after job).
    Raw failure pairs shown to GLM-5.1 (reveals off-by-one patterns).
    3 rounds of cheap GLM-5.1 diagnosis + free Qwen fix before expensive GLM-5.1 writes.

Usage:
  python ooda_driver.py --project DIR --test "CMD" --task "DESC"
  python ooda_driver.py --project DIR --test "CMD" --task "DESC" --context-file REF
  python ooda_driver.py --project DIR --test "CMD" --task "DESC" --no-qwen
"""

import subprocess, sys, os, re, json, time, shutil, difflib, urllib.request, urllib.error
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

# --- Defaults ---

QWEN_URL = "http://localhost:8033/v1"
ZAI_URL = "https://api.z.ai/api/anthropic/v1/messages"
ZAI_KEY = "f02ad9427d384e6a86953bafc196e5a0.onlQzSEVnNYjcaSL"

GLM47_MODEL = "claude-sonnet-4-6-20250514"
GLM51_MODEL = "claude-opus-4-7"
QWEN_MODEL = "local-model"
TEMP = 0.3
MAX_TOKENS = 8192
HINT_TOKENS = 2048
DIAGNOSE_TOKENS = 2048

# Diff guard: reject writes that change more than this many lines in a single file
MAX_DIFF_LINES = 200


# --- Logging ---

_log_file = None
_token_log = []  # [(model, input_tokens, output_tokens), ...]

def log(action, msg, step_label=""):
    global _log_file
    ts = datetime.now().isoformat()
    entry = {"timestamp": ts, "action": action, "message": msg}
    if step_label:
        entry["step"] = step_label
    if _log_file:
        with open(_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    label = f"Step {step_label} | " if step_label else ""
    print(f"[{ts[11:19]}] {label}{action} | {msg}", flush=True)


def log_tokens(model, resp_data):
    """Extract and log token usage from API response."""
    global _token_log
    if "usage" in resp_data:
        u = resp_data["usage"]
        inp = u.get("prompt_tokens", u.get("input_tokens", 0))
        out = u.get("completion_tokens", u.get("output_tokens", 0))
        _token_log.append((model, inp, out))
        log("tokens", f"{model}: {inp} in + {out} out = {inp+out} total")
        return inp, out
    return 0, 0


# --- API Calls ---

def call_qwen(prompt, qwen_url, context=""):
    """Local Qwen via OpenAI API."""
    if context:
        prompt = f"CONTEXT (accumulated learnings from previous iterations):\n{context}\n\n---\n\n{prompt}"
    payload = json.dumps({
        "model": QWEN_MODEL, "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMP, "max_tokens": MAX_TOKENS,
    }).encode("utf-8")
    req = urllib.request.Request(f"{qwen_url}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    log_tokens(QWEN_MODEL, data)
    content = data["choices"][0]["message"]["content"]
    Path(".qwen_last_response.txt").write_text(content, encoding="utf-8")
    return content


def _extract_traceback(output, max_lines=50):
    """Extract the first traceback from test output (the actual Python exception)."""
    lines = output.split("\n")
    tb_lines = []
    in_tb = False
    for line in lines:
        if "Traceback" in line:
            in_tb = True
            tb_lines = [line]
        elif in_tb:
            tb_lines.append(line)
            if len(tb_lines) >= max_lines:
                break
            if line and not line[0].isspace() and not line.startswith("During") and tb_lines and not tb_lines[-2].startswith("During"):
                break
    return "\n".join(tb_lines) if tb_lines else ""


def _extract_raw_failures(output, max_chars=2000):
    """Extract raw FAIL + Expected/Got triplets from test output.

    These reveal patterns like off-by-one output shifts that grouped failures hide.
    """
    lines = output.split("\n")
    result = []
    i = 0
    while i < len(lines):
        if "[FAIL]" in lines[i]:
            # Grab the FAIL line and the next 2 lines (Expected/Got)
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


def call_glm51_diagnose(test_output, code, task_desc, attempt_history="", context=""):
    """GLM-5.1 diagnoses the root cause of failures. Returns diagnosis string."""
    failure_summary = group_failures(test_output)
    raw_failures = _extract_raw_failures(test_output)
    raw_traceback = _extract_traceback(test_output)

    diagnosis_context = f"Task: {task_desc}\n\nFailure summary:\n{failure_summary}\n\n"
    diagnosis_context += f"RAW FAILURES (Expected vs Got — look for patterns like off-by-one):\n{raw_failures}\n\n"
    if raw_traceback:
        diagnosis_context += f"RAW ERROR/TRACEBACK (this is the real error — use it):\n{raw_traceback}\n\n"
    if context:
        diagnosis_context += f"CONTEXT (accumulated learnings):\n{context}\n\n"
    diagnosis_context += f"Code:\n{code}\n\n"
    if attempt_history:
        diagnosis_context += f"ATTEMPT HISTORY (do NOT repeat these approaches):\n{attempt_history}\n\n"
    diagnosis_context += ("What is the root cause and what file/function needs to change? "
                          "Look carefully at the RAW FAILURES — each Got line shows what the program actually output. "
                          "If Got matches a PREVIOUS test's expected output, it's an off-by-one / extra-output-line bug.")

    payload = json.dumps({
        "model": GLM51_MODEL, "max_tokens": DIAGNOSE_TOKENS,
        "system": ("Expert programmer. Analyze test failures and give a CONCISE diagnosis. "
                   "1. What is the ROOT CAUSE (pick the single most likely one)? "
                   "2. Which FILE and FUNCTION needs to change? "
                   "3. What specific change is needed? "
                   "Pay close attention to the RAW FAILURES — look for patterns where Got lines are shifted, "
                   "duplicated, or match a different test's expected output. This reveals off-by-one bugs. "
                   "Also check the RAW TRACEBACK if present. "
                   "Be precise. Do NOT write full solutions."),
        "messages": [{"role": "user", "content": diagnosis_context}],
    }).encode("utf-8")
    req = urllib.request.Request(ZAI_URL, data=payload,
                                 headers={"Content-Type": "application/json", "x-api-key": ZAI_KEY,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    log_tokens(GLM51_MODEL, data)
    return data["content"][0]["text"]


def call_glm51_fix(test_output, code, task_desc, diagnosis="", context=""):
    """GLM-5.1 writes the fix directly when GLM-4.7 fails."""
    failure_summary = group_failures(test_output)
    system = ("You are an expert programmer. Output ONLY the files that need fixing. "
              "Make MINIMAL changes. Do NOT rewrite working files. "
              "Format: # FILE: filename.ext\n```\nCODE\n```")
    prompt = f"Task: {task_desc}\n\nFailure summary:\n{failure_summary}\n\nCurrent code:\n{code}\n\n"
    if context:
        prompt += f"CONTEXT (accumulated learnings):\n{context}\n\n"
    if diagnosis:
        prompt += f"ROOT CAUSE DIAGNOSIS: {diagnosis}\n\n"
    prompt += ("Write ONLY the files that need fixing. "
               "Make minimal, targeted changes. Format: # FILE: filename.ext + code block.")
    payload = json.dumps({
        "model": GLM51_MODEL, "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(ZAI_URL, data=payload,
                                 headers={"Content-Type": "application/json", "x-api-key": ZAI_KEY,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    log_tokens(GLM51_MODEL, data)
    return data["content"][0]["text"]


# --- Failure grouping ---

def extract_failures(output, max_chars=2000):
    """Extract failure lines from test output."""
    lines = []
    for l in output.split("\n"):
        l = l.strip()
        if any(kw in l for kw in ["FAIL", "Error", "Expected", "Got", "MISSING", "assert", "Traceback"]):
            lines.append(l)
    return "\n".join(lines)[:max_chars]


def group_failures(output, max_chars=1500):
    """Group identical failures and return a summary with counts.

    Instead of 47 identical 'dict has no attribute maps' errors,
    returns: '[47x] Error: 'dict' object has no attribute 'maps''
    """
    failures = []
    for l in output.split("\n"):
        l = l.strip()
        if any(kw in l for kw in ["FAIL", "Error", "Expected", "Got", "MISSING", "assert"]):
            failures.append(l)

    if not failures:
        return "(no failures found in output)"

    # Group by normalized key (strip specific values, keep pattern)
    groups = Counter()
    for f in failures:
        # Normalize: remove specific test names, keep error pattern
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


# --- Code extraction with diff guard ---

def extract_code_blocks(response):
    """Extract code blocks from model response. Returns {filename: code}."""
    files = {}
    # Pattern 1: # FILE: name.ext + code block
    for m in re.finditer(r"(?:# FILE:|###?)\s*([\w.-]+\.\w+)\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
        files[m.group(1)] = m.group(2).strip()
    # Pattern 2: filename.ext on line before code block
    for m in re.finditer(r"([\w.-]+\.\w+)\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
        if m.group(1) not in files:
            files[m.group(1)] = m.group(2).strip()
    # Pattern 3: bold filename **name.ext**
    for m in re.finditer(r"\*\*([\w.-]+\.\w+)\*\*\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
        if m.group(1) not in files:
            files[m.group(1)] = m.group(2).strip()

    # Pattern 4: # FILE: name.ext without code fences
    if not files:
        for m in re.finditer(r"# FILE:\s*([\w.-]+\.\w+)\s*\n(.*?)(?=# FILE:|$)", response, re.DOTALL):
            files[m.group(1)] = m.group(2).strip()
        if files:
            log("extract_nofence", f"Extracted {len(files)} files without code fences")

    # Pattern 5: # FILE: with full path
    if not files:
        for m in re.finditer(r"# FILE:\s*[^\s]*[\\/]([\w.-]+\.\w+)\s*\n```[\w]*\n(.*?)```", response, re.DOTALL):
            if m.group(1) not in files:
                files[m.group(1)] = m.group(2).strip()

    return files


def diff_line_count(old_text, new_text):
    """Count how many lines changed between old and new text."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, n=0))
    # Count lines starting with + or - (not +++ or ---)
    changed = sum(1 for d in diff if d.startswith(('+', '-')) and not d.startswith(('+++', '---')))
    return changed


def extract_and_write(response, project_dir, primary_file=None, protected_files=None):
    """Extract code blocks, apply diff guard, write to project_dir.

    primary_file: the main file being worked on (e.g., step impl).
                  Always exempt from diff guard.
    protected_files: set of filenames that should be diff-guarded.
                  Files NOT in this set (or the primary file) can be freely rewritten.
    """
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

        # Diff guard: only protect files in the protected set that aren't the primary file
        is_protected = protected_files and name in protected_files
        if dest.exists() and name != primary_file and is_protected:
            old_text = dest.read_text(encoding="utf-8")
            changed = diff_line_count(old_text, new_code + "\n")
            if changed > MAX_DIFF_LINES:
                log("diff_guard", f"REJECTED {name}: {changed} lines changed (max {MAX_DIFF_LINES}). "
                    "Model likely rewrote the entire file instead of making targeted fix.", "")
                rejected.append(name)
                continue

        dest.write_text(new_code + "\n", encoding="utf-8")
        written.append(name)

    if rejected:
        log("diff_guard_summary", f"Rejected {len(rejected)} files: {rejected}. "
            "Will ask model to be more targeted.", "")

    return written


# --- Test running ---

def _kill_tree(pid):
    """Kill a process and all its descendants (Windows)."""
    try:
        subprocess.run(f"taskkill /PID {pid} /T /F", shell=True,
                       capture_output=True, timeout=5)
    except Exception:
        pass


def run_test(test_cmd, project_dir, timeout=60):
    """Run test command, return (passed, failed, total, output)."""
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

        # Strategy 1: "X passed, Y failed" in output
        m = re.search(r"(\d+) passed, (\d+) failed(?:, (\d+) total)?", output)
        if m:
            p, f = int(m.group(1)), int(m.group(2))
            return p, f, p + f, output

        # Strategy 2: exit code only
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
    """Read all matching source files from project directory."""
    parts = []
    for p in sorted(project_dir.glob(file_glob)):
        if p.name.startswith(".") or p.name.startswith("_"):
            continue
        parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else "(no files)"


def read_targeted_code(project_dir, file_glob="*.py", failure_output="", primary_file=None):
    """Read only files likely relevant to the failures.

    Strategy: always include primary_file. For other files, include
    only those referenced in the failure output or recently modified.
    """
    all_files = sorted(p for p in project_dir.glob(file_glob)
                       if not p.name.startswith(".") and not p.name.startswith("_"))

    if not failure_output:
        return read_project_code(project_dir, file_glob)

    # Parse file names mentioned in failure output
    mentioned = set()
    for m in re.finditer(r'([\w.-]+\.\w+)', failure_output):
        mentioned.add(m.group(1))

    # Always include primary file
    parts = []
    included = set()

    for p in all_files:
        if p.name == primary_file or p.name in mentioned:
            parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
            included.add(p.name)

    # If we included very little, add recently modified files
    if len(included) < 2:
        for p in all_files:
            if p.name not in included:
                mtime = p.stat().st_mtime
                if time.time() - mtime < 300:  # modified in last 5 min
                    parts.append(f"# {p.name}\n{p.read_text(encoding='utf-8')}")
                    included.add(p.name)

    # If still too little, fall back to everything
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


# --- Context file ---

CONTEXT_FILE = ".ooda_context.md"


def context_path(project_dir):
    return project_dir / CONTEXT_FILE


def context_create(project_dir):
    """Create empty context file at start of job."""
    cp = context_path(project_dir)
    cp.write_text("", encoding="utf-8")


def context_read(project_dir):
    """Read current context file. Returns empty string if missing/empty."""
    cp = context_path(project_dir)
    if cp.exists():
        text = cp.read_text(encoding="utf-8").strip()
        return text if text else ""
    return ""


def context_append(project_dir, entry):
    """Append a learning entry to the context file.

    Skips if the new entry is too similar to the last entry (avoids repeating
    the same diagnosis 18 times, which bloated prompts to 30K+ tokens).
    Also caps total context size at ~3K chars.
    """
    cp = context_path(project_dir)
    existing = context_read(project_dir)

    # Dedup: skip if last entry is very similar (>70% overlap with new entry)
    if existing:
        last_entry = existing.split("\n## ")[-1] if "## " in existing else existing
        entry_words = set(entry.lower().split())
        last_words = set(last_entry.lower().split())
        if entry_words and last_words:
            overlap = len(entry_words & last_words) / max(len(entry_words), 1)
            if overlap > 0.7:
                return  # Too similar to last entry, skip

    ts = datetime.now().strftime("%H:%M:%S")
    with open(cp, "a", encoding="utf-8") as f:
        f.write(f"\n## [{ts}] {entry}\n")

    # Cap: keep only last ~3K chars
    text = cp.read_text(encoding="utf-8")
    if len(text) > 3000:
        # Keep the last portion, split at entry boundary
        keep = text[-2800:]
        boundary = keep.find("\n## ")
        if boundary >= 0:
            keep = keep[boundary:]
        cp.write_text(keep, encoding="utf-8")


def context_purge(project_dir):
    """Delete context file. Called at end of every job (success or failure)."""
    cp = context_path(project_dir)
    if cp.exists():
        cp.unlink()
        log("context", "Purged context file")


# --- Build prompts ---

def build_initial_prompt(task, context, project_dir):
    """Build prompt for initial code generation (Qwen greenfield)."""
    existing = sorted(p.name for p in project_dir.glob("*.py")
                      if not p.name.startswith(".") and not p.name.startswith("_"))
    existing_str = ", ".join(existing) if existing else "none"

    p = f"Implement the following:\n\n{task}\n\nWorking directory: {project_dir}"
    if context:
        p += f"\n\nContext/Reference:\n{context}"
    p += f"\n\nIMPORTANT: These files already exist and MUST NOT be overwritten: [{existing_str}]"
    p += "\nOnly write NEW files that don't exist yet. Format: # FILE: filename.ext followed by a code block."
    return p


# --- Main OODA loop ---

def ooda_run(args):
    global _log_file
    project_dir = Path(args.project).resolve()
    cp_dir = project_dir / ".checkpoints" / "latest"
    _log_file = project_dir / "ooda_log.jsonl"

    step_label = args.step_label or ""
    primary_file = args.primary_file or None

    print("=" * 60)
    print("OODA Driver v5")
    print(f"Project: {project_dir}")
    print(f"Test: {args.test}")
    print(f"Task: {args.task[:80]}...")
    print(f"Primary file: {primary_file or '(none)'}")
    print(f"Loop: unlimited (runs until all tests pass)")
    print("=" * 60, flush=True)

    # Pre-flight: check Qwen if using it
    use_qwen = not args.no_qwen
    if use_qwen:
        try:
            req = urllib.request.Request(f"{args.qwen_url}/models")
            urllib.request.urlopen(req, timeout=5)
            log("preflight", "Qwen: OK")
        except Exception as e:
            log("preflight", f"Qwen not responding: {e}. Falling back to GLM-4.7 only.")
            use_qwen = False

    best_passed, best_failed = -1, float("inf")
    diagnosis = ""
    last_output = ""
    attempt_history = []  # [(iteration, action, result), ...]
    qwen_initial_done = False  # Track whether initial Qwen fix pass has run
    qwen_diagnose_rounds = 0  # Count of (GLM-5.1 diagnose → Qwen fix) rounds
    MAX_QWEN_DIAGNOSE_ROUNDS = 6

    # Context file: accumulates learnings across iterations
    context_create(project_dir)
    log("context", "Created context file", step_label)

    # Diff guard: track which files the driver writes. Once written by driver,
    # a file loses protection (it might be broken and need full rewrite).
    # Files the driver has NEVER touched remain protected.
    protected_files = set()
    driver_written = set()
    for p in project_dir.glob(args.file_glob):
        if not p.name.startswith(".") and not p.name.startswith("_"):
            protected_files.add(p.name)
    log("preflight", f"Protected files: {sorted(protected_files) or 'none'}", step_label)

    def get_protected():
        """Return currently protected files (never written by driver)."""
        return protected_files - driver_written

    def try_qwen_fix(label, fix_prompt, output):
        """Run Qwen with a fix prompt. Returns True if files were written."""
        nonlocal qwen_initial_done, qwen_diagnose_rounds
        ctx = context_read(project_dir)
        try:
            response = call_qwen(fix_prompt, args.qwen_url, ctx)
            files = extract_and_write(response, project_dir, primary_file, get_protected())
            driver_written.update(files)
            log(f"{label}_done", f"Qwen wrote {len(files)} files: {files}", step_label)
            attempt_history.append((iteration, label, f"wrote {files}"))
            context_append(project_dir, f"{label}: wrote {files}")
            return len(files) > 0
        except Exception as e:
            log(f"{label}_err", f"Qwen error: {e}", step_label)
            return False

    try:
        # Phase 1: Qwen drafts (or skip)
        if use_qwen:
            if primary_file and (project_dir / primary_file).exists():
                log("skip_qwen", f"{primary_file} already exists, skipping Qwen")
                use_qwen = False
            else:
                log("qwen", "Qwen generating initial code...", step_label)
                try:
                    prompt = build_initial_prompt(args.task, args.context, project_dir)
                    ctx = context_read(project_dir)
                    response = call_qwen(prompt, args.qwen_url, ctx)
                    log("qwen_raw", f"Response preview: {repr(response[:200])}", step_label)
                    files = extract_and_write(response, project_dir, primary_file, get_protected())
                    driver_written.update(files)
                    log("qwen_done", f"Qwen wrote {len(files)} files: {files}", step_label)
                except Exception as e:
                    log("qwen_err", f"Qwen error: {e}", step_label)

        iteration = 0
        while True:
            iteration += 1

            # OBSERVE: Run tests
            passed, failed, total, output = run_test(args.test, project_dir, timeout=args.timeout)
            log("test", f"{passed}/{total} passed", step_label)
            last_output = output

            # Record attempt result
            attempt_history.append((iteration, "test result", f"{passed}/{total} passed"))

            # Check pass
            if failed == 0 and total > 0:
                save_checkpoint(cp_dir, project_dir, args.file_glob)
                log("pass", f"All passed on iteration {iteration}", step_label)
                return True

            # Append learning to context file
            failure_hint = group_failures(output)[:300]
            context_append(project_dir,
                           f"Iteration {iteration}: {passed}/{total} passed. "
                           f"Key failures: {failure_hint}")

            # Phase 1b: Initial Qwen fix (runs ONCE with raw test hint)
            if use_qwen and not qwen_initial_done:
                qwen_initial_done = True
                log("qwen_fix1", "Qwen initial fix (raw hint)...", step_label)
                hint = failure_hint[:500]
                raw_tb = _extract_traceback(output)
                if raw_tb:
                    hint += f"\n\nTraceback:\n{raw_tb[:500]}"
                code = read_targeted_code(project_dir, args.file_glob, output, primary_file)
                fix_prompt = (
                    f"The code you generated has test failures.\n\n"
                    f"Task: {args.task}\n\n"
                    f"Current code:\n{code}\n\n"
                    f"FAILURE HINT:\n{hint}\n\n"
                    f"Fix the code. Output ONLY the files that need fixing. "
                    f"Format: # FILE: filename.ext + code block."
                )
                if try_qwen_fix("qwen_fix1", fix_prompt, output):
                    continue  # Re-test before escalating

            # Checkpoint or rollback
            if passed > best_passed or (passed == best_passed and failed < best_failed):
                best_passed, best_failed = passed, failed
                save_checkpoint(cp_dir, project_dir, args.file_glob)
            else:
                if restore_checkpoint(cp_dir, project_dir):
                    log("rollback", f"Rolled back (no improvement, was {best_passed}/{best_passed+best_failed})", step_label)

            # Phase 2: GLM-5.1 diagnoses, then either Qwen or GLM-5.1 fixes
            log("glm51", f"GLM-5.1 diagnosing (iteration {iteration})...", step_label)
            try:
                code = read_targeted_code(project_dir, args.file_glob, output, primary_file)
                history_str = "\n".join(f"Iteration {it}: {act} -> {res}" for it, act, res in attempt_history[-10:])
                ctx = context_read(project_dir)
                diagnosis = call_glm51_diagnose(output, code, args.task, history_str, ctx)
                log("diagnose", f"Diagnosis: {diagnosis[:200]}...", step_label)
                context_append(project_dir, f"GLM-5.1 diagnosis: {diagnosis[:300]}")

                # Phase 2a: If Qwen available and haven't exhausted rounds,
                # give the diagnosis to Qwen (free) instead of GLM-5.1 (expensive)
                if use_qwen and qwen_diagnose_rounds < MAX_QWEN_DIAGNOSE_ROUNDS:
                    qwen_diagnose_rounds += 1
                    log("qwen_diag_fix", f"Qwen fixing with GLM-5.1 diagnosis (round {qwen_diagnose_rounds}/{MAX_QWEN_DIAGNOSE_ROUNDS})...", step_label)
                    fix_prompt = (
                        f"Task: {args.task}\n\n"
                        f"Current code:\n{code}\n\n"
                        f"EXPERT DIAGNOSIS: {diagnosis}\n\n"
                        f"Apply this diagnosis to fix the code. "
                        f"Output ONLY the files that need fixing. "
                        f"Format: # FILE: filename.ext + code block."
                    )
                    if try_qwen_fix("qwen_diag_fix", fix_prompt, output):
                        continue  # Re-test

                # Phase 2b: GLM-5.1 writes the fix directly
                log("glm51_fix", f"GLM-5.1 writing fix (iteration {iteration})...", step_label)
                response = call_glm51_fix(output, code, args.task, diagnosis, ctx)
                files = extract_and_write(response, project_dir, primary_file, get_protected())
                driver_written.update(files)
                attempt_history.append((iteration, "GLM-5.1 fix", f"wrote {files}"))
                log("fix_done", f"Wrote {len(files)} files: {files}", step_label)
            except Exception as e:
                log("glm51_err", f"GLM-5.1 error: {e}", step_label)

            time.sleep(2)  # Rate limit buffer
    finally:
        # ALWAYS purge context file — this is a benchmark run multiple times
        context_purge(project_dir)


def main():
    p = argparse.ArgumentParser(description="Generic OODA Development Driver v5")
    p.add_argument("--project", required=True, help="Project directory")
    p.add_argument("--test", required=True, help="Test command to run")
    p.add_argument("--task", required=True, help="Task description (what to implement)")
    p.add_argument("--context", default="", help="Additional context (reference code, etc)")
    p.add_argument("--context-file", default="", help="File to read context from (avoids shell escaping)")
    p.add_argument("--file-glob", default="*.py", help="File pattern for checkpointing (default: *.py)")
    p.add_argument("--max-retries", type=int, default=6, help="(ignored — loop runs until pass)")
    p.add_argument("--timeout", type=int, default=60, help="Test command timeout in seconds (default: 60)")
    p.add_argument("--qwen-url", default=QWEN_URL, help="Qwen API URL")
    p.add_argument("--no-qwen", action="store_true", help="Skip Qwen, use GLM-4.7 only (default when code exists)")
    p.add_argument("--step-label", default="", help="Label for log messages (e.g. step number)")
    p.add_argument("--system-extra", default="", help="Extra system prompt for GLM-4.7")
    p.add_argument("--primary-file", default="", help="Main file being worked on (exempt from diff guard)")
    args = p.parse_args()

    # Load context from file if specified
    if args.context_file and not args.context:
        p = Path(args.context_file)
        if p.exists():
            args.context = p.read_text(encoding="utf-8")

    success = ooda_run(args)

    # Print token summary
    print("\n" + "=" * 60)
    print("TOKEN USAGE")
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
