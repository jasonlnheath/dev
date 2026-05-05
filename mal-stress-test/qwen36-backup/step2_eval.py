import sys

from mal_types import (Boolean, Error, Fn, List, Map, Nil, Number, String,
                       Symbol, Vector)

import reader
import printer


def read(s):
    return reader.read_str(s)


def eval_(ast, env):
    if isinstance(ast, Symbol):
        if ast in env:
            return env[ast]
        raise Error(f"'{ast}' not found")
    elif isinstance(ast, (Number, String)):
        return ast
    elif ast is Nil.NIL:
        return ast
    elif isinstance(ast, Boolean):
        return ast
    elif isinstance(ast, List):
        if len(ast) == 0:
            return ast
        # Eval all elements: first becomes the function, rest become args
        first = eval_(ast[0], env)
        args = [eval_(x, env) for x in ast[1:]]
        if isinstance(first, Fn):
            return first.call(args)
        raise Error(f"cannot apply {first}")
    elif isinstance(ast, Vector):
        return Vector(eval_(x, env) for x in ast)
    elif isinstance(ast, Map):
        return Map((k, eval_(v, env)) for k, v in ast.items())
    else:
        return ast


def print_(exp):
    return printer.pr_str(exp)


def add(args):
    result = Number(0)
    for arg in args:
        result = Number(result + arg)
    return result


def sub(args):
    if len(args) == 0:
        raise Error("-: wrong number of arguments")
    result = Number(args[0])
    for arg in args[1:]:
        result = Number(result - arg)
    return result


def mul(args):
    result = Number(1)
    for arg in args:
        result = Number(result * arg)
    return result


def floordiv(args):
    if len(args) == 0:
        raise Error("/: wrong number of arguments")
    result = Number(args[0])
    for arg in args[1:]:
        result = Number(result // arg)
    return result


def rep(s, env):
    return print_(eval_(read(s), env))


def main():
    repl_env = {
        '+': Fn(add),
        '-': Fn(sub),
        '*': Fn(mul),
        '/': Fn(floordiv),
    }

    for line in sys.stdin:
        try:
            print(rep(line.strip(), repl_env))
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == '__main__':
    main()
