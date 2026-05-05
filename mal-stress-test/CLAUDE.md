# MAL (Make a Lisp) Python Implementation

## Project Structure
- step0_repl.py through stepA_mal.py: incremental implementation files
- reader.py, types.py, printer.py, env.py, core.py: shared modules (created as needed)
- tests/stepN_*.mal: test files for each step

## Testing
Run tests with: `python run_test.py tests/step0_repl.mal python step0_repl.py`

Test format: each line is input, `;=>` lines are expected output. Blank lines and `;` comment lines are ignored.

## Implementation Rules
1. Each step builds on the previous. Read the reference implementation at C:\dev\mal\impls\python3\ for patterns.
2. Non-interactive mode: read from stdin line by line, print to stdout. The prompt "user> " should only appear in interactive mode.
3. After implementing each step, run the test to verify. Fix all failures before moving to the next step.
4. Keep step files self-contained in early steps, refactor to shared modules when natural.

## Step Overview
- step0_repl: Basic REPL echo (read line, print it back)
- step1_read_print: Reader (tokenizer + reader) + Printer
- step2_eval: EVAL with arithmetic and lists
- step3_env: Environments (let, def!)
- step4_if_fn_do: Control flow (if, fn*, do)
- step5_tco: Tail call optimization
- step6_file: File loading (load-file)
- step7_quote: Quoting (quote, quasiquote, splice-unquote)
- step8_macros: Macros (macroexpand, defmacro!)
- step9_try: Exception handling (try*, catch*)
- stepA_mal: Self-hosting (atoms, interop)
