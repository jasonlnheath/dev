import sys

import reader
import printer


def read(s):
    return reader.read_str(s)


def eval_(ast, env):
    return ast


def print_(exp):
    return printer.pr_str(exp)


def rep(s):
    return print_(eval_(read(s), None))


def main():
    for line in sys.stdin:
        try:
            print(rep(line.strip()))
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == '__main__':
    main()
