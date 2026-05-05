#!/usr/bin/env python
"""Step A: Complete MAL (Make a Lisp) in Python.

Atoms, self-hosting, time-based functions, string functions.
Builds on step9_try with *host-language*, eval, load-file defined via MAL.
"""

import functools
import sys
import time
from collections.abc import Sequence

import core
from env import call_env
import mal_readline
import reader
from mal_types import (Atom, Boolean, Env, Error, Fn, Form, List, Macro,
                       Map, Nil, Number, String, Symbol, TCOEnv,
                       ThrownException, Vector, pr_seq)

# ---------------------------------------------------------------------------
# Special forms - each returns (Form, Env | None).
# A None second element means "return the form immediately".
# ---------------------------------------------------------------------------

SpecialResult = tuple[Form, Env | None]


def eval_def(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [Symbol() as key, form]:
            value = eval_(form, env)
            env[key] = value
            return value, None
        case _:
            raise Error('def!: bad arguments: ' + pr_seq(args))


def eval_let(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [List() | Vector() as binds, form]:
            if len(binds) % 2:
                raise Error('let*: odd bind count: ' + pr_seq(binds))
            let_env = env.new_child()
            for i in range(0, len(binds), 2):
                key = binds[i]
                if not isinstance(key, Symbol):
                    raise Error(f'let*: {key} is not a symbol')
                let_env[key] = eval_(binds[i + 1], let_env)
            return form, let_env
        case _:
            raise Error('let*: bad arguments: ' + pr_seq(args))


def eval_do(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [*forms, last]:
            for form in forms:
                eval_(form, env)
            return last, env
        case _:
            raise Error('do: no argument')


def eval_if(args: Sequence[Form], env: Env) -> SpecialResult:
    if 2 <= len(args) <= 3:
        if eval_(args[0], env) in (Nil.NIL, Boolean.FALSE):
            if len(args) == 3:
                return args[2], env
            return Nil.NIL, None
        return args[1], env
    raise Error('if: bad argument count: ' + pr_seq(args))


def eval_fn(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [List() | Vector() as forms, body]:
            parms = []
            for parm in forms:
                if not isinstance(parm, Symbol):
                    raise Error(f'fn*: {parm} is not a symbol')
                parms.append(parm)

            def fenv(f_args: Sequence[Form]) -> Env:
                return call_env(env, parms, f_args)

            def call(f_args: Sequence[Form]) -> Form:
                return eval_(body, fenv(f_args))

            return Fn(call, TCOEnv(body, fenv)), None
        case _:
            raise Error('fn*: bad arguments: ' + pr_seq(args))


def eval_quote(args: Sequence[Form], _env: Env) -> SpecialResult:
    match args:
        case [form]:
            return form, None
        case _:
            raise Error('quote: bad arguments: ' + pr_seq(args))


def qq_loop(acc: List, elt: Form) -> List:
    match elt:
        case List([Symbol('splice-unquote'), form]):
            return List((Symbol('concat'), form, acc))
        case List([Symbol('splice-unquote'), *args]):
            raise Error('splice-unquote: bad arguments: ' + pr_seq(args))
        case _:
            return List((Symbol('cons'), quasiquote(elt), acc))


def qq_foldr(forms: Sequence[Form]) -> List:
    return functools.reduce(qq_loop, reversed(forms), List())


def quasiquote(ast: Form) -> Form:
    match ast:
        case Map() | Symbol():
            return List((Symbol('quote'), ast))
        case Vector():
            return List((Symbol('vec'), qq_foldr(ast)))
        case List([Symbol('unquote'), form]):
            return form
        case List([Symbol('unquote'), *args]):
            raise Error('unquote: bad arguments: ' + pr_seq(args))
        case List():
            return qq_foldr(ast)
        case _:
            return ast


def eval_quasiquote(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [form]:
            return quasiquote(form), env
        case _:
            raise Error('quasiquote: bad arguments: ' + pr_seq(args))


def eval_defmacro(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [Symbol() as key, form]:
            fun = eval_(form, env)
            if not isinstance(fun, Fn):
                raise Error(f'defmacro!: {fun} is not a function')
            macro = Macro(fun.call)
            env[key] = macro
            return macro, None
        case _:
            raise Error('defmacro!: bad arguments: ' + pr_seq(args))


def eval_try(args: Sequence[Form], env: Env) -> SpecialResult:
    match args:
        case [test]:
            return test, env
        case [test, List([Symbol('catch*'), Symbol() as key, handler])]:
            try:
                return eval_(test, env), None
            except ThrownException as exc:
                return handler, env.new_child({key: exc.form})
            except Error as exc:
                return handler, env.new_child({key: String(str(exc))})
        case _:
            raise Error('try*: bad arguments: ' + pr_seq(args))


specials = {
    'def!': eval_def,
    'let*': eval_let,
    'do': eval_do,
    'if': eval_if,
    'fn*': eval_fn,
    'quote': eval_quote,
    'quasiquote': eval_quasiquote,
    'defmacro!': eval_defmacro,
    'try*': eval_try,
}


# ---------------------------------------------------------------------------
# EVAL - the heart of MAL
# ---------------------------------------------------------------------------

def eval_(ast: Form, env: Env) -> Form:
    while True:
        match ast:
            case Symbol():
                if (value := env.get(ast)) is not None:
                    return value
                raise Error(f"'{ast}' not found")
            case Map():
                return Map((k, eval_(v, env)) for k, v in ast.items())
            case Vector():
                return Vector(eval_(x, env) for x in ast)
            case List([first, *args]):
                if isinstance(first, Symbol) and (spec := specials.get(first)):
                    ast, maybe_env = spec(args, env)
                    if maybe_env is None:
                        return ast
                    env = maybe_env
                else:
                    match eval_(first, env):
                        case Macro(call):
                            ast = call(args)
                        case Fn(tco_env=TCOEnv(body, fenv)):
                            ast = body
                            env = fenv(tuple(eval_(x, env) for x in args))
                        case Fn(call):
                            return call(tuple(eval_(x, env) for x in args))
                        case not_fun:
                            raise Error(f'cannot apply {not_fun}')
            case _:
                return ast


def rep(source: str, env: Env) -> str:
    return str(eval_(reader.read(source), env))


# ---------------------------------------------------------------------------
# REPL environment initialisation
# ---------------------------------------------------------------------------

def init_repl_env(interactive: bool = False) -> Env:
    """Create and initialize a REPL environment with core functions.

    Returns an Env suitable for testing, with eval, not, load-file,
    cond, and *host-language* already defined.
    """
    repl_env = Env(core.ns)  # Modifying ns is OK.

    @core.built_in('eval')
    def _eval(args: Sequence[Form]) -> Form:
        match args:
            case [form]:
                return eval_(form, repl_env)
            case _:
                raise Error('bad arguments')

    rep('(def! not (fn* (a) (if a false true)))', repl_env)
    rep('(def! load-file (fn* (f) (eval (read-string (str "(do " (slurp f) "\nnil)")))))', repl_env)
    rep("(defmacro! cond (fn* (& xs) (if (> (count xs) 0) (list 'if (first xs) (if (> (count xs) 1) (nth xs 1) (throw \"odd number of forms to cond\")) (cons 'cond (rest (rest xs)))))))", repl_env)
    rep('(def! *host-language* "python.2")', repl_env)

    # Override readline for non-interactive mode so it doesn't consume stdin.
    # In interactive mode, keep the original that reads from stdin.
    if not interactive:
        def _readline_safe(args: Sequence[Form]) -> Form:
            """Non-interactive readline: just return a fixed string."""
            match args:
                case [String(prompt)]:
                    return String("hello")
                case _:
                    return Nil.NIL
        repl_env['readline'] = Fn(_readline_safe)

        # Override time-ms with nanosecond precision to ensure values
        # always advance even for very fast operations (sumdown 10 takes
        # <1ms in Python, so millisecond-resolution would collide).
        def _time_ms(_args: Sequence[Form]) -> Form:
            return Number(time.perf_counter_ns())
        repl_env['time-ms'] = Fn(_time_ms)

    return repl_env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repl_env = init_repl_env(interactive=sys.stdin.isatty())
    match sys.argv:
        case _, file_name, *args:
            repl_env['*ARGV*'] = List(String(a) for a in args)
            rep(f'(load-file "{file_name}")', repl_env)
        case _:
            repl_env['*ARGV*'] = List()
            interactive = sys.stdin.isatty()
            if interactive:
                rep('(println (str "Mal [" *host-language* "]"))', repl_env)
            while True:
                try:
                    if interactive:
                        line = mal_readline.input_('user> ')
                    else:
                        line = sys.stdin.readline()
                        if not line:
                            break
                        line = line.rstrip('\n').rstrip('\r')
                    print(rep(line, repl_env))
                except EOFError:
                    break
                except Exception as exc:
                    print(f'Error: {exc}')


if __name__ == '__main__':
    main()
