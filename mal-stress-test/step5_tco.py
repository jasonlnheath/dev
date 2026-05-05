import sys
import os

# Add the current directory to the path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mal_readline
import reader
import mal_types
from env import call_env


def is_truthy(val):
    """Returns True if val is truthy (not Nil.NIL and not Boolean.FALSE)."""
    if val is mal_types.Nil.NIL:
        return False
    if val is mal_types.Boolean.FALSE:
        return False
    return True


def eval_ast(ast, env):
    """Evaluate an AST node without TCO loop (used for non-tail positions)."""
    # Handle Keyword as a primitive literal
    if isinstance(ast, mal_types.Keyword):
        return ast
        
    if isinstance(ast, mal_types.Symbol):
        try:
            return env[ast]
        except KeyError:
            raise mal_types.Error(f"'{ast}' not found")
    elif isinstance(ast, mal_types.List):
        if len(ast) == 0:
            return ast
        
        head = ast[0]
        
        # Special Form: def!
        if isinstance(head, mal_types.Symbol) and str(head) == "def!":
            if len(ast) != 3:
                raise mal_types.Error("def! requires exactly 2 arguments (name and value)")
            name = ast[1]
            if not isinstance(name, mal_types.Symbol):
                raise mal_types.Error("def! name must be a symbol")
            value_ast = ast[2]
            result = eval_ast(value_ast, env)
            env.set(name, result)
            return result
        
        # Special Form: let*
        elif isinstance(head, mal_types.Symbol) and str(head) == "let*":
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
                value = eval_ast(value_ast, new_env)
                new_env.set(symbol, value)
            
            # Evaluate body in new environment
            return eval_ast(body_ast, new_env)

        # Special Form: if
        elif isinstance(head, mal_types.Symbol) and str(head) == "if":
            if len(ast) < 3 or len(ast) > 4:
                raise mal_types.Error("if requires 3 or 4 arguments")
            cond = eval_ast(ast[1], env)
            if is_truthy(cond):
                return eval_ast(ast[2], env)
            else:
                if len(ast) == 4:
                    return eval_ast(ast[3], env)
                else:
                    return mal_types.Nil.NIL
        
        # Special Form: fn*
        elif isinstance(head, mal_types.Symbol) and str(head) == "fn*":
            if len(ast) != 3:
                raise mal_types.Error("fn* requires exactly 3 arguments")
            params = ast[1]
            body = ast[2]
            
            if not isinstance(params, (mal_types.List, mal_types.Vector)):
                raise mal_types.Error("fn* parameters must be a list or vector")
            
            # Create closure capturing current env
            def make_fn_call(params_list, body_list, captured_env):
                def call(args):
                    new_env = call_env(captured_env, [str(p) for p in params_list], args)
                    return eval_loop(body_list, new_env)
                
                fenv_builder = lambda args: call_env(captured_env, [str(p) for p in params_list], args)
                tco_env = mal_types.TCOEnv(body=body_list, fenv=fenv_builder)
                
                return mal_types.Fn(call=call, tco_env=tco_env)

            return make_fn_call(params, body, env)
        
        # Special Form: do
        elif isinstance(head, mal_types.Symbol) and str(head) == "do":
            if len(ast) < 2:
                raise mal_types.Error("do requires at least one expression")
            result = mal_types.Nil.NIL
            for expr in ast[1:]:
                result = eval_ast(expr, env)
            return result
        
        # Function Call
        else:
            fn = eval_ast(head, env)
            if not isinstance(fn, mal_types.Fn):
                raise mal_types.Error(f"{fn} is not a function")
            
            args = [eval_ast(arg, env) for arg in ast[1:]]
            return fn.call(args)
    
    elif isinstance(ast, mal_types.Vector):
        return mal_types.Vector([eval_ast(item, env) for item in ast])
    
    elif isinstance(ast, mal_types.Map):
        new_map = mal_types.Map()
        for k, v in ast.items():
            new_map[eval_ast(k, env)] = eval_ast(v, env)
        return new_map
    
    else:
        # Primitives
        return ast


def eval_loop(ast, env):
    """Main evaluation loop with TCO support."""
    current_ast = ast
    current_env = env
    
    while True:
        if isinstance(current_ast, mal_types.Symbol):
            try:
                return current_env[current_ast]
            except KeyError:
                raise mal_types.Error(f"'{current_ast}' not found")
        
        elif isinstance(current_ast, (mal_types.String, mal_types.Number, 
                                     mal_types.Boolean, mal_types.Nil,
                                     mal_types.Keyword)):
            return current_ast
        
        elif isinstance(current_ast, mal_types.List):
            if len(current_ast) == 0:
                return current_ast
            
            head = current_ast[0]
            
            # Check if it's a special form symbol
            if isinstance(head, mal_types.Symbol):
                form_name = str(head)
                
                # Special Form: def!
                if form_name == "def!":
                    if len(current_ast) != 3:
                        raise mal_types.Error("def! requires exactly 2 arguments (name and value)")
                    name = current_ast[1]
                    if not isinstance(name, mal_types.Symbol):
                        raise mal_types.Error("def! name must be a symbol")
                    value_ast = current_ast[2]
                    result = eval_loop(value_ast, current_env)
                    current_env.set(name, result)
                    return result
                
                # Special Form: let*
                elif form_name == "let*":
                    if len(current_ast) != 3:
                        raise mal_types.Error("let* requires exactly 2 arguments (bindings and body)")
                    bindings_ast = current_ast[1]
                    body_ast = current_ast[2]
                    
                    if not isinstance(bindings_ast, (mal_types.List, mal_types.Vector)):
                        raise mal_types.Error("let* bindings must be a list or vector")
                    
                    # Create new environment with outer=current_env
                    new_env = mal_types.Env(mappings={}, outer=current_env)
                    
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
                        value = eval_loop(value_ast, new_env)
                        new_env.set(symbol, value)
                    
                    # Set current_ast to body and continue loop with new_env
                    current_ast = body_ast
                    current_env = new_env
                    continue
                
                # Special Form: do
                elif form_name == "do":
                    if len(current_ast) < 2:
                        raise mal_types.Error("do requires at least one expression")
                    # Evaluate all but last
                    for expr in current_ast[1:-1]:
                        eval_loop(expr, current_env)
                    # Tail call the last
                    current_ast = current_ast[-1]
                    continue
                
                # Special Form: if
                elif form_name == "if":
                    if len(current_ast) < 3 or len(current_ast) > 4:
                        raise mal_types.Error("if requires 3 or 4 arguments")
                    cond = eval_loop(current_ast[1], current_env)
                    if is_truthy(cond):
                        current_ast = current_ast[2]
                    else:
                        if len(current_ast) == 4:
                            current_ast = current_ast[3]
                        else:
                            return mal_types.Nil.NIL
                    continue
                
                # Special Form: fn*
                elif form_name == "fn*":
                    if len(current_ast) != 3:
                        raise mal_types.Error("fn* requires exactly 3 arguments")
                    params = current_ast[1]
                    body = current_ast[2]
                    
                    if not isinstance(params, (mal_types.List, mal_types.Vector)):
                        raise mal_types.Error("fn* parameters must be a list or vector")
                    
                    # Create the function object with TCO support
                    def make_fn_call(params_list, body_list, captured_env):
                        def call(args):
                            new_env = call_env(captured_env, [str(p) for p in params_list], args)
                            return eval_loop(body_list, new_env)
                        
                        fenv_builder = lambda args: call_env(captured_env, [str(p) for p in params_list], args)
                        tco_env = mal_types.TCOEnv(body=body_list, fenv=fenv_builder)
                        
                        return mal_types.Fn(call=call, tco_env=tco_env)

                    return make_fn_call(params, body, current_env)
                
                else:
                    # Function Call
                    # Evaluate the function expression first
                    fn = eval_loop(current_ast[0], current_env)
                    
                    # Evaluate all arguments
                    args = []
                    for arg in current_ast[1:]:
                        args.append(eval_loop(arg, current_env))
                    
                    # Check if it's a MalFunc and supports TCO
                    if isinstance(fn, mal_types.Fn):
                        # Use TCO mechanism if available
                        if fn.tco_env is not None:
                            current_ast = fn.tco_env.body
                            current_env = fn.tco_env.fenv(args)
                            continue
                        else:
                            # Standard call
                            return fn.call(args)
                    else:
                        raise mal_types.Error(f"{fn} is not a function")
            else:
                # First element is not a symbol, treat as function call? 
                # In MAL, lists are usually (func arg1 arg2). If func is not a symbol, 
                # we evaluate it. But standard MAL assumes first is symbol or evaluated form.
                # Let's evaluate the head as a form.
                fn = eval_loop(current_ast[0], current_env)
                
                if isinstance(fn, mal_types.Fn):
                    # Evaluate all arguments
                    args = [eval_loop(arg, current_env) for arg in current_ast[1:]]
                    
                    if fn.tco_env is not None:
                        current_ast = fn.tco_env.body
                        current_env = fn.tco_env.fenv(args)
                        continue
                    else:
                        return fn.call(args)
                else:
                    raise mal_types.Error(f"{fn} is not a function")
        
        elif isinstance(current_ast, mal_types.Vector):
            # Vectors are evaluated element-wise and returned as a new vector
            return mal_types.Vector([eval_loop(item, current_env) for item in current_ast])
        
        elif isinstance(current_ast, mal_types.Map):
            # Maps are evaluated key-value wise
            new_map = mal_types.Map()
            for k, v in current_ast.items():
                new_map[eval_loop(k, current_env)] = eval_loop(v, current_env)
            return new_map
        
        else:
            raise mal_types.Error(f"Unsupported AST type: {type(current_ast)}")


def rep(source: str, env: mal_types.Env) -> str:
    try:
        ast = reader.read(source.strip())
        result = eval_loop(ast, env)
        return str(result)
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
            line = mal_readline.input_()
            if not line:
                continue
            print(rep(line, repl_env))
        except EOFError:
            break


if __name__ == '__main__':
    main()
