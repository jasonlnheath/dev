import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reader import read
from mal_types import Env

def eval_ast(ast, env):
    """Step 1: EVAL returns ast unchanged."""
    return ast

def rep(source, repl_env):
    """Read, Eval, Print cycle for a single source string."""
    try:
        ast = read(source)
        evaluated = eval_ast(ast, repl_env)
        print(str(evaluated))
    except Exception as e:
        print(f"Error: {e}")

def main():
    repl_env = Env()
    while True:
        try:
            source = input()
            if source:
                rep(source, repl_env)
        except EOFError:
            break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
