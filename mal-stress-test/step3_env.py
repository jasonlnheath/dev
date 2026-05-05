import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mal_readline
import reader
import mal_types


def READ(source: str):
    return reader.read(source)


def EVAL(ast, env=None):
    if env is None:
        raise mal_types.Error("Environment must be provided")

    # Check DEBUG-EVAL in the *current* environment before evaluating symbols or lists
    if mal_types.Symbol('DEBUG-EVAL') in env:
        debug_val = env[mal_types.Symbol('DEBUG-EVAL')]
        if debug_val is not mal_types.Boolean.FALSE and debug_val is not mal_types.Nil.NIL:
            print(f"EVAL: {ast}")

    if isinstance(ast, mal_types.Symbol):
        try:
            return env[ast]
        except KeyError:
            raise mal_types.Error(f"'{ast}' not found")
    elif isinstance(ast, mal_types.List):
        if len(ast) == 0:
            return ast
        
        first = ast[0]
        
        # def! special form
        if isinstance(first, mal_types.Symbol) and str(first) == "def!":
            if len(ast) != 3:
                raise mal_types.Error("def! requires exactly 2 arguments (name and value)")
            name = ast[1]
            if not isinstance(name, mal_types.Symbol):
                raise mal_types.Error("def! name must be a symbol")
            value_ast = ast[2]
            result = EVAL(value_ast, env)
            env.set(name, result)
            return result
        
        # let* special form
        elif isinstance(first, mal_types.Symbol) and str(first) == "let*":
            if len(ast) != 3:
                raise mal_types.Error("let* requires exactly 2 arguments (bindings and body)")
            bindings_ast = ast[1]
            body_ast = ast[2]
            
            if not isinstance(bindings_ast, (mal_types.List, mal_types.Vector)):
                raise mal_types.Error("let* bindings must be a list or vector")
            
            # Create new environment with outer=env
            new_env = mal_types.Env(mappings={}, outer=env)
            
            # Iterate over pairs in bindings
            bindings = bindings_ast
            if len(bindings) % 2 != 0:
                raise mal_types.Error("let* bindings must have even number of elements")
            
            for i in range(0, len(bindings), 2):
                symbol = bindings[i]
                value_ast = bindings[i + 1]
                if not isinstance(symbol, mal_types.Symbol):
                    raise mal_types.Error("let* binding key must be a symbol")
                # Evaluate value in the new environment so later bindings can reference earlier ones
                value = EVAL(value_ast, new_env)
                new_env.set(symbol, value)
            
            # Evaluate body in new environment
            return EVAL(body_ast, new_env)
        
        else:
            # Regular function call
            evaluated = [EVAL(e, env) for e in ast]
            func = evaluated[0]
            args = evaluated[1:]
            return func.call(list(args))
    elif isinstance(ast, mal_types.Vector):
        return mal_types.Vector(EVAL(e, env) for e in ast)
    else:
        # Other types (numbers, strings, booleans, etc.) return as-is
        return ast


def PRINT(form):
    return str(form)


def rep(source: str, repl_env: mal_types.Env) -> str:
    try:
        return PRINT(EVAL(READ(source), repl_env))
    except (mal_types.Error, reader.Error) as e:
        return f"Error: {e}"


def main() -> None:
    # Create the REPL environment with built-in functions
    repl_env = mal_types.Env()
    
    # Add arithmetic operations using mal_types.Fn
    repl_env.set(mal_types.Symbol("+"), mal_types.Fn(call=lambda args: mal_types.Number(sum(args))))
    
    def minus_fn(args):
        if len(args) == 0:
            raise mal_types.Error("Minus requires at least one argument")
        elif len(args) == 1:
            return mal_types.Number(-args[0])
        else:
            result = args[0]
            for arg in args[1:]:
                result = result - arg
            return mal_types.Number(result)
    
    repl_env.set(mal_types.Symbol("-"), mal_types.Fn(call=minus_fn))
    
    import functools
    
    def multiply_fn(args):
        if len(args) == 0:
            return mal_types.Number(1)
        return mal_types.Number(functools.reduce(lambda x, y: x * y, args))
    
    repl_env.set(mal_types.Symbol("*"), mal_types.Fn(call=multiply_fn))
    
    def divide_fn(args):
        if len(args) != 2:
            raise mal_types.Error("Division requires exactly two arguments")
        if args[1] == 0:
            raise mal_types.Error("Division by zero")
        return mal_types.Number(args[0] // args[1])
    
    repl_env.set(mal_types.Symbol("/"), mal_types.Fn(call=divide_fn))

    # Add core functions for step4
    # prn: print args and return nil
    def prn_fn(args):
        print(*args, sep=" ")
        return mal_types.Nil.NIL
    
    repl_env.set(mal_types.Symbol("prn"), mal_types.Fn(call=prn_fn))
    
    # list: return the args as a mal_types.List
    def list_fn(args):
        return mal_types.List(args)
    
    repl_env.set(mal_types.Symbol("list"), mal_types.Fn(call=list_fn))
    
    # list?: check if arg is a list
    def list_q_fn(args):
        if len(args) != 1:
            raise mal_types.Error("list? requires exactly one argument")
        return mal_types.Boolean(isinstance(args[0], mal_types.List))
    
    repl_env.set(mal_types.Symbol("list?"), mal_types.Fn(call=list_q_fn))
    
    # empty?: check if vector or list is empty
    def empty_q_fn(args):
        if len(args) != 1:
            raise mal_types.Error("empty? requires exactly one argument")
        arg = args[0]
        if isinstance(arg, (mal_types.Vector, mal_types.List)):
            return mal_types.Boolean(len(arg) == 0)
        else:
            raise mal_types.Error("empty? expects a vector or list")
    
    repl_env.set(mal_types.Symbol("empty?"), mal_types.Fn(call=empty_q_fn))
    
    # count: return the number of elements in a vector or list
    def count_fn(args):
        if len(args) != 1:
            raise mal_types.Error("count requires exactly one argument")
        arg = args[0]
        if isinstance(arg, mal_types.Nil):
            return mal_types.Number(0)
        if isinstance(arg, (mal_types.Vector, mal_types.List)):
            return mal_types.Number(len(arg))
        else:
            raise mal_types.Error("count expects a vector or list")
    
    repl_env.set(mal_types.Symbol("count"), mal_types.Fn(call=count_fn))
    
    # =: check equality of all arguments
    def eq_fn(args):
        if len(args) == 0:
            return mal_types.Boolean(True)
        first = args[0]
        for arg in args[1:]:
            if not (first == arg):
                return mal_types.Boolean(False)
        return mal_types.Boolean(True)
    
    repl_env.set(mal_types.Symbol("="), mal_types.Fn(call=eq_fn))
    
    # < : less than
    def lt_fn(args):
        if len(args) != 2:
            raise mal_types.Error("< requires exactly two arguments")
        return mal_types.Boolean(args[0] < args[1])
    
    repl_env.set(mal_types.Symbol("<"), mal_types.Fn(call=lt_fn))
    
    # <= : less than or equal
    def lte_fn(args):
        if len(args) != 2:
            raise mal_types.Error("<= requires exactly two arguments")
        return mal_types.Boolean(args[0] <= args[1])
    
    repl_env.set(mal_types.Symbol("<="), mal_types.Fn(call=lte_fn))
    
    # > : greater than
    def gt_fn(args):
        if len(args) != 2:
            raise mal_types.Error("> requires exactly two arguments")
        return mal_types.Boolean(args[0] > args[1])
    
    repl_env.set(mal_types.Symbol(">"), mal_types.Fn(call=gt_fn))
    
    # >= : greater than or equal
    def gte_fn(args):
        if len(args) != 2:
            raise mal_types.Error(">= requires exactly two arguments")
        return mal_types.Boolean(args[0] >= args[1])
    
    repl_env.set(mal_types.Symbol(">="), mal_types.Fn(call=gte_fn))

    # not: logical negation
    def not_fn(args):
        if args[0] is mal_types.Boolean.FALSE or args[0] is mal_types.Nil.NIL:
            return mal_types.Boolean.TRUE
        return mal_types.Boolean.FALSE
    repl_env.set(mal_types.Symbol("not"), mal_types.Fn(call=not_fn))

    # pr-str: readable string of args joined by space
    def pr_str_fn(args):
        return mal_types.String(' '.join(str(a) for a in args))
    repl_env.set(mal_types.Symbol("pr-str"), mal_types.Fn(call=pr_str_fn))

    # str: non-readable string of args concatenated
    def str_fn(args):
        return mal_types.String(''.join(a.__str__(readably=False) for a in args))
    repl_env.set(mal_types.Symbol("str"), mal_types.Fn(call=str_fn))

    # println: print args space-separated followed by newline, return nil
    def println_fn(args):
        print(' '.join(a.__str__(readably=False) for a in args))
        return mal_types.Nil.NIL
    repl_env.set(mal_types.Symbol("println"), mal_types.Fn(call=println_fn))

    while True:
        try:
            print(rep(mal_readline.input_('user> '), repl_env))
        except EOFError:
            break


if __name__ == '__main__':
    main()
