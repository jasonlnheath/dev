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


def _quasiquote_expand(form, env):
    """Expand quasiquote, evaluating unquoted forms in env.

    Returns the expanded result directly (not a cons-chain form).
    """
    # Self-evaluating atoms
    if isinstance(form, (mal_types.String, mal_types.Number,
                         mal_types.Boolean, mal_types.Keyword)) or \
       form is mal_types.Nil.NIL:
        return form

    if not isinstance(form, (mal_types.List, mal_types.Vector)):
        return form

    # (unquote x) -> evaluate x in env
    if isinstance(form, mal_types.List) and len(form) == 2:
        head = form[0]
        if isinstance(head, mal_types.Symbol) and str(head) == "unquote":
            return eval_loop(form[1], env)
        # (splice-unquote x) -> evaluate x, return marker
        if isinstance(head, mal_types.Symbol) and str(head) == "splice-unquote":
            val = eval_loop(form[1], env)
            if not isinstance(val, (mal_types.List, mal_types.Vector)):
                raise mal_types.Error("splice-unquote requires a list or vector")
            return ("SPLICE", val)

    # Process list/vector elements
    py_list = list(form)
    result_py = []
    for elem in py_list:
        expanded = _quasiquote_expand(elem, env)
        if isinstance(expanded, tuple) and len(expanded) == 2 and expanded[0] == "SPLICE":
            result_py.extend(list(expanded[1]))
        else:
            result_py.append(expanded)

    if isinstance(form, mal_types.Vector):
        return mal_types.Vector(result_py)
    return mal_types.List(result_py)


def macroexpand(ast, env):
    """Expand macros in the AST. If ast is a list and first element is a symbol resolving to a macro, expand it."""
    if not isinstance(ast, mal_types.List) or len(ast) == 0:
        return ast
    
    # Check if the first element is a symbol that resolves to a macro
    first = ast[0]
    if isinstance(first, mal_types.Symbol):
        try:
            resolved = env[first]
            if isinstance(resolved, mal_types.Macro):
                # It's a macro! Call it with unevaluated arguments (rest of the list)
                args = ast[1:]
                result = resolved.call(args)
                # Recursively expand the result in case the expansion contains more macros
                return macroexpand(result, env)
        except KeyError:
            pass
        except mal_types.Error:
            pass
    
    return ast


def eval_ast(ast, env):
    """Evaluate an AST node without TCO loop (used for non-tail positions)."""
    if 'DEBUG-EVAL' in env:
        print(f"EVAL: {ast}")
    
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
            
            new_env = mal_types.Env(mappings={}, outer=env)
            
            bindings = bindings_ast
            if len(bindings) % 2 != 0:
                raise mal_types.Error("let* bindings must have even number of elements")
            
            for i in range(0, len(bindings), 2):
                symbol = bindings[i]
                value_ast = bindings[i + 1]
                if not isinstance(symbol, mal_types.Symbol):
                    raise mal_types.Error("let* binding key must be a symbol")
                value = eval_ast(value_ast, new_env)
                new_env.set(symbol, value)
            
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
            return eval_ast(_quasiquote_expand(form, env), env)
        
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
        return ast


def eval_loop(ast, env):
    """Main evaluation loop with TCO support."""
    current_ast = ast
    current_env = env

    while True:
        # Macro expansion before anything else (including DEBUG-EVAL print)
        if isinstance(current_ast, mal_types.List) and len(current_ast) > 0:
            head = current_ast[0]
            if isinstance(head, mal_types.Symbol):
                fn = current_env.get(head, None)
                if isinstance(fn, mal_types.Macro):
                    result = fn.call(current_ast[1:])
                    expanded = macroexpand(result, current_env)
                    current_ast = expanded
                    continue

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
                    
                    new_env = mal_types.Env(mappings={}, outer=current_env)
                    
                    bindings = bindings_ast
                    if len(bindings) % 2 != 0:
                        raise mal_types.Error("let* bindings must have even number of elements")
                    
                    for i in range(0, len(bindings), 2):
                        symbol = bindings[i]
                        value_ast = bindings[i + 1]
                        if not isinstance(symbol, mal_types.Symbol):
                            raise mal_types.Error("let* binding key must be a symbol")
                        value = eval_loop(value_ast, new_env)
                        new_env.set(symbol, value)
                    
                    current_ast = body_ast
                    current_env = new_env
                    continue
                
                # Special Form: do
                elif form_name == "do":
                    if len(current_ast) < 2:
                        raise mal_types.Error("do requires at least one expression")
                    for expr in current_ast[1:-1]:
                        eval_loop(expr, current_env)
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
                    return _quasiquote_expand(current_ast[1], current_env)
                
                # Special Form: defmacro!
                elif form_name == "defmacro!":
                    if len(current_ast) != 3:
                        raise mal_types.Error("defmacro! requires exactly 2 arguments (name and value)")
                    name = current_ast[1]
                    if not isinstance(name, mal_types.Symbol):
                        raise mal_types.Error("defmacro! name must be a symbol")
                    value_ast = current_ast[2]
                    result = eval_loop(value_ast, current_env)
                    if not isinstance(result, mal_types.Fn):
                        raise mal_types.Error("defmacro! value must be a function")
                    macro_fn = mal_types.Macro(call=result.call)
                    current_env.set(name, macro_fn)
                    return macro_fn
                
                # Special Form: cond
                elif form_name == "cond":
                    # Expand one level: (cond pred val ...rest) -> (if pred val (cond ...rest))
                    if len(current_ast) < 2:
                        return mal_types.Nil.NIL
                    args = list(current_ast[1:])
                    if len(args) >= 2:
                        pred = args[0]
                        cons_val = args[1]
                        rest = args[2:]
                        if rest:
                            next_cond = mal_types.List((mal_types.Symbol("cond"),) + tuple(rest))
                            current_ast = mal_types.List((mal_types.Symbol("if"), pred, cons_val, next_cond))
                        else:
                            current_ast = mal_types.List((mal_types.Symbol("if"), pred, cons_val))
                        continue
                    else:
                        return mal_types.Nil.NIL

                # Special Form: try*
                elif form_name == "try*":
                    if len(current_ast) < 3:
                        raise mal_types.Error("try* requires at least a body and one catch clause")
                    
                    body_forms = current_ast[1:-1]
                    catch_clause = current_ast[-1]
                    
                    if not isinstance(catch_clause, mal_types.List) or len(catch_clause) < 3:
                        raise mal_types.Error("catch* clause must be (catch* variable body...)")
                    
                    catch_var = catch_clause[1]
                    catch_body = catch_clause[2:]
                    
                    if not isinstance(catch_var, mal_types.Symbol):
                        raise mal_types.Error("catch* variable must be a symbol")
                    
                    try:
                        result = mal_types.Nil.NIL
                        for expr in body_forms:
                            result = eval_loop(expr, current_env)
                        return result
                    except Exception as e:
                        # Bind the caught exception object to the catch variable
                        new_env = mal_types.Env(mappings={}, outer=current_env)
                        new_env.set(catch_var, e)
                        catch_result = mal_types.Nil.NIL
                        for expr in catch_body:
                            catch_result = eval_loop(expr, new_env)
                        return catch_result

                # Special Form: throw
                elif form_name == "throw":
                    if len(current_ast) != 2:
                        raise mal_types.Error("throw requires exactly one argument")
                    value = eval_loop(current_ast[1], current_env)
                    raise mal_types.ThrownException(value)
                
                else:
                    # Function Call / Macro Call
                    # Check if the head is a macro BEFORE evaluating arguments
                    fn = current_env.get(head, None)
                    
                    if isinstance(fn, mal_types.Macro):
                        # Macro call: pass unevaluated arguments (rest of the list)
                        result = fn.call(current_ast[1:])
                        # Recursively expand the result
                        current_ast = macroexpand(result, current_env)
                        continue
                    
                    # If not a macro, treat as function call
                    fn_evaluated = eval_loop(current_ast[0], current_env)
                    
                    # Evaluate all arguments
                    args = []
                    for arg in current_ast[1:]:
                        args.append(eval_loop(arg, current_env))
                    
                    # Check if it's a MalFunc and supports TCO
                    if isinstance(fn_evaluated, mal_types.Fn):
                        # Use TCO mechanism if available
                        if fn_evaluated.tco_env is not None:
                            current_ast = fn_evaluated.tco_env.body
                            current_env = fn_evaluated.tco_env.fenv(args)
                            continue
                        else:
                            return fn_evaluated.call(args)
                    else:
                        raise mal_types.Error(f"{fn_evaluated} is not a function")
            else:
                # First element is not a symbol, treat as function call? 
                fn_evaluated = eval_loop(current_ast[0], current_env)
                
                if isinstance(fn_evaluated, mal_types.Fn):
                    args = [eval_loop(arg, current_env) for arg in current_ast[1:]]
                    
                    if fn_evaluated.tco_env is not None:
                        current_ast = fn_evaluated.tco_env.body
                        current_env = fn_evaluated.tco_env.fenv(args)
                        continue
                    else:
                        return fn_evaluated.call(args)
                else:
                    raise mal_types.Error(f"{fn_evaluated} is not a function")
        
        elif isinstance(current_ast, mal_types.Vector):
            return mal_types.Vector([eval_loop(item, current_env) for item in current_ast])
        
        elif isinstance(current_ast, mal_types.Map):
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
    except Exception as e:
        return f"Error: {e}"


def load_file(filename, env):
    """Reads a file, parses it into a list of forms, wraps in (do ...), and evaluates."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
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
            
        do_ast = mal_types.List((mal_types.Symbol('do'), *forms))
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
    repl_env = mal_types.Env(core.ns)
    
    argv_list = mal_types.List([mal_types.String(arg) for arg in sys.argv[1:]])
    repl_env.set(mal_types.Symbol("*ARGV*"), argv_list)

    def load_file_fn(args):
        if len(args) != 1:
            raise mal_types.Error("load-file requires exactly one argument (filename)")
        filename = args[0]
        if not isinstance(filename, mal_types.String):
            raise mal_types.Error("load-file expects a string argument")
        return load_file(filename.val, repl_env)

    repl_env.set(mal_types.Symbol("load-file"), mal_types.Fn(call=load_file_fn))

    def eval_fn(args):
        if len(args) != 1:
            raise mal_types.Error("eval requires exactly one argument")
        ast = args[0]
        if isinstance(ast, mal_types.String):
            try:
                parsed_ast = reader.read(ast.val.strip())
                result = eval_loop(parsed_ast, repl_env)
                return result
            except (mal_types.Error, reader.Error) as e:
                raise mal_types.Error(str(e))
        else:
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
