import sys
import os
import core
import reader
import mal_readline
import mal_types
from env import call_env


def is_truthy(val):
    """Returns True if val is truthy (not Nil.NIL and not Boolean.FALSE)."""
    if val is mal_types.Nil.NIL:
        return False
    if val is mal_types.Boolean.FALSE:
        return False
    return True


def _quasiquote_expand(form):
    """Expand quasiquote into a form for EVAL to evaluate.

    Returns forms (not evaluated results):
    - Self-evaluating atoms (nil, numbers, strings, bools, keywords): as-is
    - Symbols: (quote sym)
    - (unquote x): x as-is (for EVAL to evaluate)
    - (splice-unquote x): ("SPLICE", x) marker
    - Lists: (cons elem1 (cons elem2 ... ()))
    - Vectors: (vec (cons elem1 ... ()))
    """
    # Self-evaluating atoms
    if isinstance(form, (mal_types.String, mal_types.Number,
                         mal_types.Boolean, mal_types.Keyword)) or \
       form is mal_types.Nil.NIL:
        return form

    # Symbols and maps wrap in quote
    if isinstance(form, mal_types.Symbol) or isinstance(form, mal_types.Map):
        return mal_types.List((mal_types.Symbol("quote"), form))

    if not isinstance(form, (mal_types.List, mal_types.Vector)):
        return form

    # (unquote x) -> return x (a form for EVAL to evaluate later)
    if isinstance(form, mal_types.List) and len(form) == 2:
        head = form[0]
        if isinstance(head, mal_types.Symbol) and str(head) == "unquote":
            return form[1]
        # (splice-unquote x) -> return marker (handled by parent)
        if isinstance(head, mal_types.Symbol) and str(head) == "splice-unquote":
            return ("SPLICE", form[1])

    # Process list/vector elements, building cons chain
    result = mal_types.List()
    for elem in reversed(form):
        expanded = _quasiquote_expand(elem)
        if isinstance(expanded, tuple) and len(expanded) == 2 and expanded[0] == "SPLICE":
            # splice-unquote: (concat x result)
            result = mal_types.List((mal_types.Symbol("concat"), expanded[1], result))
        else:
            result = mal_types.List((mal_types.Symbol("cons"), expanded, result))

    if isinstance(form, mal_types.Vector):
        return mal_types.List((mal_types.Symbol("vec"), result))

    return result


def eval_ast(ast, env):
    """Evaluate an AST node without TCO loop (used for non-tail positions)."""
    if 'DEBUG-EVAL' in env:
        print(f"EVAL: {ast}")
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
        
        # Special Form: quote
        elif isinstance(head, mal_types.Symbol) and str(head) == "quote":
            if len(ast) != 2:
                raise mal_types.Error("quote requires exactly 1 argument")
            return ast[1]
        
        # Special Form: quasiquote
        elif isinstance(head, mal_types.Symbol) and str(head) == "quasiquote":
            if len(ast) != 2:
                raise mal_types.Error("quasiquote requires exactly 1 argument")
            form = ast[1]
            return eval_ast(_quasiquote_expand(form), env)
        
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
        if 'DEBUG-EVAL' in current_env:
            print(f"EVAL: {current_ast}")
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
                
                # Special Form: quote
                elif form_name == "quote":
                    if len(current_ast) != 2:
                        raise mal_types.Error("quote requires exactly 1 argument")
                    return current_ast[1]
                
                # Special Form: quasiquote
                elif form_name == "quasiquote":
                    if len(current_ast) != 2:
                        raise mal_types.Error("quasiquote requires exactly 1 argument")
                    current_ast = _quasiquote_expand(current_ast[1])
                    continue
                
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
            return mal_types.Vector([eval_loop(item, current_ast) for item in current_ast])
        
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


def load_file(filename, env):
    """Reads a file, parses it into a list of forms, wraps in (do ...), and evaluates."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Helper to read multiple forms from a string
        def read_all_forms(source):
            forms = []
            lexer = reader.Lexer(source)
            while True:
                try:
                    form = reader.read_form(lexer, None)
                    forms.append(form)
                except reader.Error as e:
                    if "empty" in str(e):
                        break
                    raise
            return forms

        forms = read_all_forms(content)
        
        if not forms:
            return mal_types.Nil.NIL
            
        # Wrap in (do ...)
        do_ast = mal_types.List((mal_types.Symbol('do'), *forms))
        
        # Evaluate the do block
        result = eval_loop(do_ast, env)
        return mal_types.Nil.NIL
        
    except FileNotFoundError:
        raise mal_types.Error(f"File '{filename}' not found")
    except Exception as e:
        raise mal_types.Error(str(e))


def eval_string(source, env):
    """Evaluate a string as MAL code."""
    try:
        ast = reader.read(source.strip())
        result = eval_loop(ast, env)
        return result
    except (mal_types.Error, reader.Error) as e:
        raise mal_types.Error(str(e))


def main() -> None:
    # Create the REPL environment with built-in functions from core.py
    repl_env = mal_types.Env(core.ns)
    
    # Add *ARGV*
    argv_list = mal_types.List([mal_types.String(arg) for arg in sys.argv[1:]])
    repl_env.set(mal_types.Symbol("*ARGV*"), argv_list)

    # Add load-file to the environment
    def load_file_fn(args):
        if len(args) != 1:
            raise mal_types.Error("load-file requires exactly one argument (filename)")
        filename = args[0]
        if not isinstance(filename, mal_types.String):
            raise mal_types.Error("load-file expects a string argument")
        return load_file(filename.val, repl_env)

    repl_env.set(mal_types.Symbol("load-file"), mal_types.Fn(call=load_file_fn))

    # Add eval to the environment
    def eval_fn(args):
        if len(args) != 1:
            raise mal_types.Error("eval requires exactly one argument")
        ast = args[0]
        # If it's a string, parse it first
        if isinstance(ast, mal_types.String):
            try:
                parsed_ast = reader.read(ast.val.strip())
                result = eval_loop(parsed_ast, repl_env)
                return result
            except (mal_types.Error, reader.Error) as e:
                raise mal_types.Error(str(e))
        else:
            # It's already an AST
            result = eval_loop(ast, repl_env)
            return result

    repl_env.set(mal_types.Symbol("eval"), mal_types.Fn(call=eval_fn))

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
