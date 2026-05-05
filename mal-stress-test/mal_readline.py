import sys


def input_(prompt: str = '') -> str:
    try:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("EOF")
        if line.endswith('\n'):
            line = line[:-1]
        return line
    except EOFError:
        raise
