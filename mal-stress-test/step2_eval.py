import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mal_readline
import reader
import mal_types


def READ(source: str):
    return reader.read(source)


def EVAL(ast, env):
    """Step 2: EVAL with arithmetic.
    
    Numbers/strings/nil/bool return self. 
    Symbols lookup in repl_env dict. 
    Lists: eval all elements, apply first (must be callable).
    Vectors and Maps are evaluated recursively.
    """
    if isinstance(ast, mal_types.Symbol):
        # Symbol lookup
        try:
            return env[ast]
        except KeyError:
            raise mal_types.Error(f"Symbol not found: {ast}")
    elif isinstance(ast, mal_types.List):
        if len(ast) == 0:
            return ast
        # Evaluate all elements
        evaluated = [EVAL(e, env) for e in ast]
        func = evaluated[0]
        args = evaluated[1:]
        # Apply the function
        return func.call(list(args))
    elif isinstance(ast, mal_types.Vector):
        # Evaluate all elements in the vector
        return mal_types.Vector([EVAL(e, env) for e in ast])
    elif isinstance(ast, mal_types.Map):
        # Evaluate keys and values. Keys must be hashable (strings, symbols, numbers).
        # Values are evaluated.
        new_map = {}
        for k, v in ast.items():
            # Keys generally don't need evaluation if they are literals, 
            # but to be safe and consistent with some implementations, 
            # we evaluate the value. Keys in Mal/reader are usually literals or symbols.
            # If a key is a symbol, it's already resolved? No, AST keys are usually literals.
            # Let's assume keys are static literals for now, but values are evaluated.
            eval_v = EVAL(v, env)
            new_map[k] = eval_v
        return mal_types.Map(new_map)
    else:
        # Numbers, strings, nil, bools return themselves
        return ast


def PRINT(form):
    return str(form)


def rep(source: str, repl_env: mal_types.Env) -> str:
    try:
        return PRINT(EVAL(READ(source), repl_env))
    except (mal_types.Error, reader.Error) as e:
        return f"Error: {e}"


def main() -> None:
    repl_env = mal_types.Env()
    # Add arithmetic functions as Python lambdas on integers
    repl_env.set(mal_types.Symbol("+"), mal_types.Fn(call=lambda args: mal_types.Number(sum(args))))
    repl_env.set(mal_types.Symbol("-"), mal_types.Fn(call=lambda args: mal_types.Number(args[0] - args[1])))
    repl_env.set(mal_types.Symbol("*"), mal_types.Fn(call=lambda args: mal_types.Number(args[0] * args[1])))
    repl_env.set(mal_types.Symbol("/"), mal_types.Fn(call=lambda args: mal_types.Number(args[0] // args[1])))

    while True:
        try:
            # REPL: input() NO prompt, do NOT skip blank lines.
            source = mal_readline.input_()
            print(rep(source, repl_env))
        except EOFError:
            break


if __name__ == '__main__':
    main()
