#!/usr/bin/env python
"""
MAL Runner - Runs generic OODA driver for each MAL step.

Usage: python mal_runner.py [--start N] [--end N] [--max-retries N]
"""

import subprocess, sys, os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent / "mal-stress-test"
DRIVER = Path(__file__).parent / "ooda_driver.py"
REF_DIR = Path(r"C:\dev\mal\impls\python3")

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

TASKS = {
    0: ("Step 0: REPL echo. NO imports. Use input()/print(). "
        "while True: line=input(); print(line). EOFError break. "
        "No prompt string. Do NOT skip blank lines. "
        "ONLY step0_repl.py."),
    1: ("Step 1: Reader+Printer for MAL Lisp. "
        "Files: mal_types.py (MalList, MalVector, MalHashMap, MalSymbol, MalString, "
        "MalNumber, MalNil, MalBool, MalKeyword), "
        "reader.py (tokenize with regex for parens/strings/escapes/numbers/symbols/comments/"
        "@/`/~/~@, read_str with Reader class for peek/next/read_form/read_list/read_atom), "
        "printer.py (pr_str function), "
        "step1_read_print.py (READ/EVAL/PRINT/REP + while loop with input()). "
        "Use mal_types.py NOT types.py. "
        "REPL: use input() with NO prompt string, do NOT skip blank lines. "
        "Write ALL needed files."),
    2: ("Step 2: EVAL with arithmetic. "
        "Builds on step1. Add step2_eval.py with EVAL function. "
        "Numbers/strings/nil/bool return self. Symbols lookup in repl_env dict. "
        "Lists: eval all elements, apply first (must be callable). "
        "repl_env has +, -, *, / as Python lambdas on integers. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    3: ("Step 3: Environments. "
        "Builds on step2. Add env.py with Env class (dict + outer pointer). "
        "Add def! (set in current env) and let* (new env with bindings). "
        "step3_env.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    4: ("Step 4: if/fn*/do. "
        "Builds on step3. do=eval all, return last. "
        "if=eval condition, then/else branches. nil/false are falsy. "
        "fn*=create closure (captures env, params, body). "
        "step4_if_fn_do.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    5: ("Step 5: TCO (Tail Call Optimization). "
        "Builds on step4. Modify EVAL loop for TCO in if/do/fn* tail positions. "
        "step5_tco.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    6: ("Step 6: load-file, eval, *ARGV*. "
        "Builds on step5. Add core.py with built-in functions. "
        "load-file reads file, wraps in (do ...), evaluates. "
        "eval: evaluate a string. *ARGV*: command line args. "
        "step6_file.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    7: ("Step 7: quote, quasiquote, unquote, splice-unquote. "
        "Builds on step6. Add quote/quasiquote reader macros. "
        "quote returns unevaluated. quasiquote expands unquote/splice-unquote. "
        "step7_quote.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    8: ("Step 8: defmacro!, macroexpand. "
        "Builds on step7. defmacro! like def! but marks as macro. "
        "macroexpand expands one macro call. EVAL auto-expands macros. "
        "step8_macros.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    9: ("Step 9: try*/catch*, throw. "
        "Builds on step8. try* evaluates body, catch* on exception. "
        "throw raises exceptions. "
        "step9_try.py. "
        "REPL: input() NO prompt, do NOT skip blank lines. "
        "Write ALL needed files."),
    10: ("Step A: atoms, swap!, reset!, string/time fns. "
         "Builds on step9. Atoms: mutable containers. "
         "swap! applies fn to atom value. reset! sets atom value. "
         "String functions: pr-str, str, prn, println, readline, readstring. "
         "Time functions: time-ms. "
         "stepA_mal.py. "
         "REPL: input() NO prompt, do NOT skip blank lines. "
         "Write ALL needed files."),
}


def read_reference(step_num):
    """Read reference implementation files for context."""
    step = STEPS[step_num]
    files_to_read = [step["impl"]]
    if step_num >= 1:
        files_to_read += ["mal_types.py", "reader.py", "printer.py"]
    if step_num >= 3:
        files_to_read += ["env.py"]
    if step_num >= 6:
        files_to_read += ["core.py"]

    parts = []
    for f in files_to_read:
        p = REF_DIR / f
        if p.exists():
            parts.append(f"# REF {f}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else ""


def read_previous_step(step_num):
    """Read the working code from the previous step (already passing tests).

    This gives Qwen a known-good base to build on instead of starting from scratch.
    """
    if step_num == 0:
        return ""
    prev = STEPS[step_num - 1]
    prev_file = PROJECT_DIR / prev["impl"]
    if not prev_file.exists():
        return ""
    return f"# WORKING CODE FROM PREVIOUS STEP ({prev['impl']}):\n{prev_file.read_text(encoding='utf-8')}"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--max-retries", type=int, default=10)
    args = p.parse_args()

    print("=" * 60)
    print(f"MAL Runner: Steps {args.start}-{args.end}")
    print("=" * 60, flush=True)

    results = []
    for step in STEPS:
        if step["num"] < args.start or step["num"] > args.end:
            continue

        # Only delete step impl, keep shared modules from previous steps
        impl = PROJECT_DIR / step["impl"]
        if impl.exists():
            impl.unlink()

        task = TASKS.get(step["num"], "")
        test_cmd = f"python run_test.py {step['test']} python {step['impl']}"

        # Write context to file to avoid shell escaping issues
        context_file = PROJECT_DIR / f".context_step{step['num']}.txt"
        context_parts = []
        prev_code = read_previous_step(step["num"])
        if prev_code:
            context_parts.append(prev_code)
        ref = read_reference(step["num"])
        if ref:
            context_parts.append(ref)
        context = "\n\n".join(context_parts)
        if context:
            context_file.write_text(context, encoding="utf-8")

        cmd = [
            sys.executable, str(DRIVER),
            "--project", str(PROJECT_DIR),
            "--test", test_cmd,
            "--task", task,
            "--file-glob", "*.py",
            "--max-retries", str(args.max_retries),
            "--timeout", "30",
            "--step-label", str(step["num"]),
            "--system-extra", "Use mal_types.py not types.py.",
            "--primary-file", step["impl"],
            "--self-hint",
            "--stuck-threshold", "3",
        ]
        # Use previous step as base for incremental development
        if step["num"] >= 1:
            prev = STEPS[step["num"] - 1]
            prev_impl = PROJECT_DIR / prev["impl"]
            if prev_impl.exists():
                cmd += ["--base-file", prev["impl"]]
        if context:
            cmd += ["--context-file", str(context_file)]

        r = subprocess.run(cmd)
        ok = r.returncode == 0

        # Post-step regression check: re-test previous step to catch shared module breaks
        if ok and step["num"] > 0:
            prev = STEPS[step["num"] - 1]
            prev_impl = PROJECT_DIR / prev["impl"]
            if prev_impl.exists():
                prev_test_cmd = f"python run_test.py {prev['test']} python {prev['impl']}"
                prev_r = subprocess.run(prev_test_cmd.split(), cwd=PROJECT_DIR,
                                        capture_output=True, text=True, timeout=30)
                if prev_r.returncode != 0:
                    print(f"  WARNING: {prev['name']} REGRESSION after {step['name']}! (likely core.py change)")
                    # Extract last line for details
                    last_line = prev_r.stdout.strip().split('\n')[-1] if prev_r.stdout else ""
                    print(f"    {last_line}")

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
