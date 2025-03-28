#!/usr/bin/env python3

import sys

class Var(int):
    __match_args__ = ("as_var",)

    def __repr__(self):
        return f"Var({int(self)})"

    @property
    def as_var(self):
        return self

code = []

for line in sys.stdin:
    if line.startswith("#"):
        continue
    [_, op, *args] = line.split()
    match op:
        case "var-x" | "var-y":
            code.append((op,))
        case "const":
            code.append((op, float(args[0])))
        case "add" | "sub" | "mul" | "max" | "min" | "neg" | "square" | "sqrt":
            code.append((op, *(Var(int(arg.lstrip("_"), 16)) for arg in args)))
        case _:
            raise

code.append(("le_imm", Var(len(code) - 1), 0.0))
code.append(("return", Var(len(code) - 1)))

def substitute_vars(inst, f):
    out = []
    for x in inst:
        if isinstance(x, Var):
            out.append(f(x))
        else:
            out.append(x)
    return tuple(out)

def push(x, y):
    n = len(x)
    x.append(y)
    return n

def emit(out, cse, inst):
    if inst in cse:
        return cse[inst]
    i = Var(push(out, inst))
    cse[inst] = i
    return i

def simplify(out, cse, inst):
    match inst:
        case "var-x" | "var-y",:
            return emit(out, cse, inst)
        case "const", float(_):
            return emit(out, cse, inst)
        case op, Var(x):
            match (op, out[x]):
                case "sqrt", ("hypot2", x, y):
                    return emit(out, cse, ("hypot", x, y))
                case "square", ("neg", x):
                    return emit(out, cse, ("square", x))
                case _:
                    return emit(out, cse, (op, x))
        case op, Var(x), float(c):
            match (op, out[x]):
                case "ge_imm", ("neg", x):
                    return simplify(out, cse, ("le_imm", x, - c))
                case "ge_imm", ("add_imm", x, d):
                    return simplify(out, cse, ("ge_imm", x, c - d))
                case "ge_imm", ("min", x, y):
                    x = simplify(out, cse, ("ge_imm", x, c))
                    y = simplify(out, cse, ("ge_imm", y, c))
                    return simplify(out, cse, ("and", x, y))
                case "ge_imm", ("max", x, y):
                    x = simplify(out, cse, ("ge_imm", x, c))
                    y = simplify(out, cse, ("ge_imm", y, c))
                    return simplify(out, cse, ("or", x, y))
                case "le_imm", ("const", x):
                    return emit(out, cse, (("true",) if x <= c else ("false",)))
                case "le_imm", ("neg", x):
                    return simplify(out, cse, ("ge_imm", x, - c))
                case "le_imm", ("add_imm", x, d):
                    return simplify(out, cse, ("le_imm", x, c - d))
                case "le_imm", ("min", x, y):
                    x = simplify(out, cse, ("le_imm", x, c))
                    y = simplify(out, cse, ("le_imm", y, c))
                    return simplify(out, cse, ("or", x, y))
                case "le_imm", ("max", x, y):
                    x = simplify(out, cse, ("le_imm", x, c))
                    y = simplify(out, cse, ("le_imm", y, c))
                    return simplify(out, cse, ("and", x, y))
                case _:
                    return emit(out, cse, (op, x, c))
        case op, Var(x), Var(y):
            match (op, out[x], out[y]):
                case "add", ("square", x), ("square", y):
                    return emit(out, cse, ("hypot2", x, y))
                case "add", ("const", x), _:
                    return emit(out, cse, ("add_imm", y, x))
                case "add", _, ("const", y):
                    return emit(out, cse, ("add_imm", x, y))
                case "add", ("add_imm", x, c), _:
                    return emit(out, cse, ("add_imm", emit(out, cse, ("add", x, y)), c))
                case "add", _, ("add_imm", y, c):
                    return emit(out, cse, ("add_imm", emit(out, cse, ("add", x, y)), c))
                case "sub", ("const", x), _:
                    return emit(out, cse, ("neg", emit(out, cse, ("add_imm", y, - x))))
                case "sub", _, ("const", y):
                    return emit(out, cse, ("add_imm", x, - y))
                case "sub", ("add_imm", x, c), _:
                    return emit(out, cse, ("add_imm", emit(out, cse, ("sub", x, y)), c))
                case "sub", _, ("add_imm", y, c):
                    return emit(out, cse, ("add_imm", emit(out, cse, ("sub", x, y)), - c))
                case "mul", ("const", x), _:
                    return emit(out, cse, ("mul_imm", y, x))
                case "mul", _, ("const", y):
                    return emit(out, cse, ("mul_imm", x, y))
                case "and", _, _:
                    return emit(out, cse, ("and", min(x, y), max(x, y)))
                case "or", _, ("false",):
                    return x
                case "or", _, _:
                    return emit(out, cse, ("or", min(x, y), max(x, y)))
                case _:
                    return emit(out, cse, (op, x, y))
        case _:
            raise

def lower(code):
    # Simplification and Common Subexpression Elimination

    out = []
    cse = {}
    map = [] # old var -> new var

    for inst in code:
        inst = substitute_vars(inst, lambda i: map[i])
        map.append(simplify(out, cse, inst))

    # Dead Code Elimination

    code = out
    used = [False for _ in code]
    used[-1] = True

    for i, inst in reversed(list(enumerate(code))):
        if used[i]:
            for x in inst:
                if isinstance(x, Var):
                    used[x] = True

    out = []
    map = [] # old var -> new var

    for i, inst in enumerate(code):
        if used[i]:
            inst = substitute_vars(inst, lambda i: map[i])
            map.append(Var(push(out, inst)))
        else:
            map.append(None)

    return out

code = lower(code)

from collections import defaultdict

counts = defaultdict(lambda: 0)

for i, inst in enumerate(code):
    counts[inst[0]] += 1
    inst = substitute_vars(inst, lambda i: code[i])
    print(Var(i), "=", inst)

for x, y in counts.items():
    print(x, y)
