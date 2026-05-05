#!/usr/bin/env python
"""
QwenDev v5 - Qwen drafts, GLM-4.7 fixes, GLM-5.1 hints.

Flow:
  1. Qwen generates initial code (local, fast ~30s)
  2. Test
  3. If fail -> GLM-4.7 generates fix (cloud, ~30s)
  4. If GLM-4.7 fails 2x -> GLM-5.1 gives hint, GLM-4.7 retries with hint
  5. Checkpoint best, rollback on regression

Usage: python qwen_ooda_driver.py [--start N] [--end N]
"""

import subprocess, sys, os, re, json, time, shutil, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent / "mal-stress-test"
LOG_FILE = PROJECT_DIR / "qwen_ooda_log.jsonl"
CHECKPOINT_DIR = PROJECT_DIR / ".checkpoints"

QWEN_URL = "http://localhost:8033/v1"
ZAI_URL = "https://api.z.ai/api/anthropic/v1/messages"
ZAI_KEY = "f02ad9427d384e6a86953bafc196e5a0.onlQzSEVnNYjcaSL"

STEPS = [
    {"num": 0, "name": "step0_repl",       "test": "tests/step0_repl.mal",       "impl": "step0_repl.py"},
    {"num": 1, "name": "step1_read_print",  "test": "tests/step1_read_print.mal", "impl": "step1_read_print.py"},
    {"num": 2, "name": "step2_eval",        "test": "tests/step2_eval.mal",       "impl": "step2_eval.py"},
    {"num": 3, "name": "step3_env",         "test": "tests/step3_env.mal",        "impl": "step3_env.py"},
    {"num": 4, "name": "step4_if_fn_do",    "test": "tests/step4_if_fn_do.mal",   "impl": "step4_if_fn_do.py"},
    {"num": 5, "name": "step5_tco",         "test": "tests/step5_tco.mal",        "impl": "step5_tco.py"},
    {"num": 6, "name": "step6_file",        "test": "tests/step6_file.mal",       "impl": "step6_file.py"},
    {"num": 7, "name": "step7_quote",       "test": "tests/step7_quote.mal",      "impl": "step7_quote.py"},
    {"num": 8, "name": "step8_macros",      "test": "tests/step8_macros.mal",     "impl": "step8_macros.py"},
    {"num": 9, "name": "step9_try",         "test": "tests/step9_try.mal",        "impl": "step9_try.py"},
    {"num": 10, "name": "stepA_mal",        "test": "tests/stepA_mal.mal",        "impl": "stepA_mal.py"},
]

STEP_REQS = {
    0: "Step 0: REPL echo. NO imports. Use input()/print(). while True: line=input(); print(line). EOFError break. No prompt. ONLY step0_repl.py.",
    1: "Step 1: Reader+Printer. Files: mal_types.py (MalList/Vector/HashMap/Symbol/String/Number/Nil/Bool/Keyword), reader.py (tokenize+read_str: parens,strings,escapes,numbers,symbols,comments,@,`,~,~@), printer.py (pr_str), step1_read_print.py. Use mal_types.py NOT types.py. REPL: use input() with NO prompt string, do NOT skip blank lines.",
    2: "Step 2: EVAL with +,-,*,/. Numbers/strings/nil/bool return self. Symbols lookup. Lists eval+apply. step2_eval.py. REPL: input() NO prompt, do NOT skip blank lines.",
    3: "Step 3: Environments. env.py (Env class). def! and let*. step3_env.py. REPL: input() NO prompt, do NOT skip blank lines.",
    4: "Step 4: if/fn*/do. do=eval all return last. if=condition. fn*=closure. step4_if_fn_do.py. REPL: input() NO prompt, do NOT skip blank lines.",
    5: "Step 5: TCO. Tail call optimization in if/do/fn*. step5_tco.py. REPL: input() NO prompt, do NOT skip blank lines.",
    6: "Step 6: load-file, eval, argv. step6_file.py. REPL: input() NO prompt, do NOT skip blank lines.",
    7: "Step 7: quote, quasiquote, unquote, splice-unquote. step7_quote.py. REPL: input() NO prompt, do NOT skip blank lines.",
    8: "Step 8: defmacro!, macroexpand. step8_macros.py. REPL: input() NO prompt, do NOT skip blank lines.",
    9: "Step 9: try*/catch*, throw. step9_try.py. REPL: input() NO prompt, do NOT skip blank lines.",
    10: "Step A: atoms, swap!, reset!, string/time fns. stepA_mal.py. REPL: input() NO prompt, do NOT skip blank lines.",
}


# --- Logging ---

def log(step, action, msg):
    ts = datetime.now().isoformat()
    entry = {"timestamp": ts, "step": step, "action": action, "message": msg}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{ts[11:19]}] Step {step} | {action} | {msg}", flush=True)


# --- API Calls ---

def call_qwen(prompt):
    """Local Qwen via OpenAI API. ~30s for code gen."""
    payload = json.dumps({
        "model": "local-model", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 8192,
    }).encode("utf-8")
    req = urllib.request.Request(f"{QWEN_URL}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def call_glm47(prompt):
    """GLM-4.7 (Sonnet) via Z.AI for code fixes. ~30s."""
    payload = json.dumps({
        "model": "claude-sonnet-4-6-20250514", "max_tokens": 8192,
        "system": "You are a Python expert fixing MAL (Make a Lisp) implementation. "
                  "Output complete fixed Python files. "
                  "Format: # FILE: filename.py\\n```python\\nCODE\\n```\\n"
                  "Write ALL files that need changes. Use mal_types.py not types.py.",
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(ZAI_URL, data=payload,
                                 headers={"Content-Type": "application/json", "x-api-key": ZAI_KEY,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


def call_glm51_for_hint(step, test_output, code):
    """GLM-5.1 (Opus) gives a concise hint. ~20s."""
    failures = get_test_failures(test_output)
    payload = json.dumps({
        "model": "claude-opus-4-7", "max_tokens": 2048,
        "system": "Python/Lisp expert. Give a CONCISE hint about the root cause. "
                  "Name the file and function to fix. Do NOT write the full solution.",
        "messages": [{"role": "user", "content":
            f"Step: {step['name']}\nFailures:\n{failures}\n\nCode:\n{code}\n\nRoot cause?"}],
    }).encode("utf-8")
    req = urllib.request.Request(ZAI_URL, data=payload,
                                 headers={"Content-Type": "application/json", "x-api-key": ZAI_KEY,
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["content"][0]["text"]


# --- Code extraction & file writing ---

def extract_and_write(response, step):
    """Extract code blocks from response, write to files."""
    files = {}
    # Pattern: # FILE: name.py + code block
    for m in re.finditer(r"(?:# FILE:|###?)\s*(\w+\.py)\s*\n```(?:python)?\n(.*?)```", response, re.DOTALL):
        files[m.group(1)] = m.group(2).strip()
    # Fallback: name.py on line before code block
    for m in re.finditer(r"(\w+\.py)\s*\n```(?:python)?\n(.*?)```", response, re.DOTALL):
        if m.group(1) not in files:
            files[m.group(1)] = m.group(2).strip()
    # Last fallback: single code block -> step impl
    if not files:
        blocks = re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
        if blocks:
            files[step["impl"]] = max(blocks, key=len).strip()

    # Fix for ALL steps: strip prompts, mal_readline, and blank-line skips
    for name in files:
        files[name] = files[name].replace("input_(", "input(")
        files[name] = re.sub(r'input\(["\']user>\s*["\']\)', "input()", files[name])
        files[name] = re.sub(r'input\(["\']mal-user>\s*["\']\)', "input()", files[name])
        files[name] = re.sub(r'import mal_readline.*\n', '', files[name])
        files[name] = re.sub(r'mal_readline\.\w+\([^)]*\)', 'input()', files[name])
        # Remove blank-line skips that hide test inputs
        files[name] = re.sub(r'\s*if not line:\s*\n\s*continue\s*\n', '\n', files[name])

    written = []
    for name, code in files.items():
        (PROJECT_DIR / name).write_text(code + "\n", encoding="utf-8")
        written.append(name)
    return written


# --- Testing ---

def run_tests(step):
    cmd = [sys.executable, str(PROJECT_DIR / "run_test.py"), step["test"], sys.executable, step["impl"]]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(PROJECT_DIR))
        out = r.stdout + r.stderr
        m = re.search(r"(\d+) passed, (\d+) failed", out)
        return (int(m.group(1)), int(m.group(2)), out) if m else (0, 1, out)
    except Exception as e:
        return 0, 1, str(e)


def get_test_failures(output):
    return "\n".join(l.strip() for l in output.split("\n")
                     if "[FAIL]" in l or "Expected:" in l or "Got:" in l)[:2000]


# --- Read code ---

def read_code(step):
    parts = []
    for f in [step["impl"]] + ["mal_types.py", "reader.py", "printer.py", "env.py", "core.py", "mal_readline.py"]:
        p = PROJECT_DIR / f
        if p.exists():
            parts.append(f"# {f}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) or "(no files)"


def read_reference(step):
    ref = Path(r"C:\dev\mal\impls\python3")
    files = [step["impl"]]
    if step["num"] >= 1: files += ["mal_types.py", "reader.py", "printer.py"]
    if step["num"] >= 3: files += ["env.py"]
    if step["num"] >= 6: files += ["core.py"]
    parts = []
    for f in files:
        p = ref / f
        if p.exists():
            parts.append(f"# REF {f}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else ""


# --- Checkpointing ---

def save_cp(n):
    d = CHECKPOINT_DIR / f"step{n}"
    d.mkdir(parents=True, exist_ok=True)
    for f in PROJECT_DIR.glob("*.py"):
        if f.name.startswith("_") or f.name in ("ooda_driver.py","run_test.py","parse_tests.py","smoke_test.py"):
            continue
        shutil.copy2(f, d / f.name)

def restore_cp(n):
    d = CHECKPOINT_DIR / f"step{n}"
    if not d.exists(): return False
    for f in d.glob("*.py"):
        shutil.copy2(f, PROJECT_DIR / f.name)
    return True


# --- Build prompts ---

def qwen_prompt(step):
    req = STEP_REQS.get(step["num"], "")
    ref = read_reference(step) if step["num"] > 0 else ""
    p = f"Implement {step['name']} of MAL in Python.\nWorking dir: {PROJECT_DIR}\n\n{req}"
    if ref:
        p += f"\n\nReference:\n{ref}"
    p += "\n\nUse mal_types.py NOT types.py. Write ALL needed files."
    return p


def fix_prompt(step, test_output, hint=""):
    code = read_code(step)
    failures = get_test_failures(test_output)
    p = (f"Fix {step['name']}. Test failures:\n{failures}\n\n"
         f"Current code:\n{code}\n\n"
         f"Write ALL files that need fixing.")
    if hint:
        p += f"\n\nEXPERT HINT: {hint}"
    return p


# --- Main OODA loop ---

def run_step(step, max_retries=6):
    log(step["num"], "start", f"Starting {step['name']}")

    best_passed, best_failed = -1, float("inf")
    glm47_fails = 0  # count GLM-4.7 failures
    hint = ""

    # Phase 1: Qwen drafts
    log(step["num"], "qwen", "Qwen generating initial code...")
    try:
        response = call_qwen(qwen_prompt(step))
        files = extract_and_write(response, step)
        log(step["num"], "qwen_done", f"Qwen wrote {len(files)} files: {files}")
    except Exception as e:
        log(step["num"], "qwen_err", f"Qwen error: {e}")
        files = []

    for iteration in range(1, max_retries + 1):
        # Test
        passed, failed, output = run_tests(step)
        total = passed + failed
        log(step["num"], "test", f"{passed}/{total} passed")

        if failed == 0 and total > 0:
            save_cp(step["num"])
            log(step["num"], "pass", f"All passed on iteration {iteration}")
            return True

        # Checkpoint or rollback
        if passed > best_passed or (passed == best_passed and failed < best_failed):
            best_passed, best_failed = passed, failed
            save_cp(step["num"])
        else:
            restore_cp(step["num"])

        # Phase 2: GLM-4.7 fixes
        if glm47_fails >= 2 and not hint:
            # Phase 3: GLM-5.1 hint
            log(step["num"], "hint_51", "GLM-4.7 stuck, asking GLM-5.1 for hint...")
            try:
                hint = call_glm51_for_hint(step, output, read_code(step))
                log(step["num"], "hint_51_done", f"Hint: {hint[:150]}...")
            except Exception as e:
                log(step["num"], "hint_51_err", f"GLM-5.1 error: {e}")
                hint = ""

        log(step["num"], "glm47", f"GLM-4.7 fixing (attempt {iteration}, glm47_fails={glm47_fails})...")
        try:
            response = call_glm47(fix_prompt(step, output, hint))
            files = extract_and_write(response, step)
            log(step["num"], "glm47_done", f"GLM-4.7 wrote {len(files)} files: {files}")
        except Exception as e:
            log(step["num"], "glm47_err", f"GLM-4.7 error: {e}")

        glm47_fails += 1

    log(step["num"], "exhausted", f"Best: {best_passed}/{best_passed+best_failed}")
    return False


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--max-retries", type=int, default=6)
    args = p.parse_args()

    print("=" * 60)
    print("QwenDev v5: Qwen drafts, GLM-4.7 fixes, GLM-5.1 hints")
    print(f"Steps: {args.start}-{args.end}, Max retries: {args.max_retries}")
    print("=" * 60, flush=True)

    # Pre-flight
    try:
        req = urllib.request.Request(f"{QWEN_URL}/models")
        urllib.request.urlopen(req, timeout=5)
        print("Qwen: OK", flush=True)
    except:
        print("ERROR: Qwen not responding"); sys.exit(1)

    results = []
    for step in STEPS:
        if step["num"] < args.start or step["num"] > args.end:
            continue
        # Only delete the step impl file, keep shared modules from previous steps
        impl = PROJECT_DIR / step["impl"]
        if impl.exists(): impl.unlink()

        ok = run_step(step, args.max_retries)
        results.append({"step": step["num"], "name": step["name"], "passed": ok})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  Step {r['step']:2d} ({r['name']:20s}): {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n{total}/{len(results)} passed")


if __name__ == "__main__":
    main()
