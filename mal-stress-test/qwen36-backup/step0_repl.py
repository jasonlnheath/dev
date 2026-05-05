import sys

def read(s):
    return s

def eval(ast, env):
    return ast

def print_(exp):
    return str(exp)

def rep(s):
    return print_(eval(read(s), None))

for line in sys.stdin:
    print(rep(line.strip()))
