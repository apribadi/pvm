#!/usr/bin/env python3

import sys

from math import copysign

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

code.append(("le", Var(len(code) - 1), 0.0))
code.append(("ret", Var(len(code) - 1)))

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
        case "var-x",:
            return emit(out, cse, ("affine", 1.0, 0.0, 0.0))
        case "var-y",:
            return emit(out, cse, ("affine", 0.0, 1.0, 0.0))
        case "const", float(_):
            return emit(out, cse, inst)
        case op, Var(x):
            match (op, out[x]):
                case "neg", ("affine", a, b, c):
                    return emit(out, cse, ("affine", - a, - b, - c))
                case "square", ("affine", a, b, c):
                    if a + b < 0:
                        return emit(out, cse, ("square", emit(out, cse, ("affine", - a + 0.0, - b + 0.0, - c + 0.0))))
                    else:
                        return emit(out, cse, ("square", emit(out, cse, ("affine", a + 0.0, b + 0.0, c + 0.0))))
                case _:
                    return emit(out, cse, (op, x))
        case op, Var(x), float(c):
            match (op, out[x]):
                case "ge", ("affine", a, b, d):
                    if a + b < 0:
                        return emit(out, cse, ("le", emit(out, cse, ("affine", - a + 0.0, - b + 0.0, 0.0)), - (c - d) + 0.0))
                    else:
                        return emit(out, cse, ("ge", emit(out, cse, ("affine", a + 0.0, b + 0.0, 0.0)), (c - d) + 0.0))
                case "ge", ("sqrt", x):
                    if c <= 0.0:
                        return emit(out, cse, ("true",))
                    return emit(out, cse, ("ge", x, c * c))
                case "ge", ("add_imm", x, d):
                    return simplify(out, cse, ("ge", x, c - d))
                case "ge", ("min", x, y):
                    x = simplify(out, cse, ("ge", x, c))
                    y = simplify(out, cse, ("ge", y, c))
                    return simplify(out, cse, ("and", x, y))
                case "ge", ("max", x, y):
                    x = simplify(out, cse, ("ge", x, c))
                    y = simplify(out, cse, ("ge", y, c))
                    return simplify(out, cse, ("or", x, y))
                case "le", ("affine", a, b, d):
                    if a + b < 0:
                        return emit(out, cse, ("ge", emit(out, cse, ("affine", - a + 0.0, - b + 0.0, 0.0)), - (c - d) + 0.0))
                    else:
                        return emit(out, cse, ("le", emit(out, cse, ("affine", a + 0.0, b + 0.0, 0.0)), (c - d) + 0.0))
                case "le", ("sqrt", x):
                    if c < 0.0:
                        return emit(out, cse, ("false",))
                    return emit(out, cse, ("le", x, c * c))
                case "le", ("const", x):
                    return emit(out, cse, (("true",) if x <= c else ("false",)))
                case "le", ("neg", x):
                    return simplify(out, cse, ("ge", x, - c))
                case "le", ("add_imm", x, d):
                    return simplify(out, cse, ("le", x, c - d))
                case "le", ("min", x, y):
                    x = simplify(out, cse, ("le", x, c))
                    y = simplify(out, cse, ("le", y, c))
                    return simplify(out, cse, ("or", x, y))
                case "le", ("max", x, y):
                    x = simplify(out, cse, ("le", x, c))
                    y = simplify(out, cse, ("le", y, c))
                    return simplify(out, cse, ("and", x, y))
                case _:
                    return emit(out, cse, (op, x, c))
        case op, Var(x), Var(y):
            match (op, out[x], out[y]):
                case "add", ("affine", a, b, c), ("affine", d, e, f):
                    return emit(out, cse, ("affine", a + d, b + e, c + f))
                case "add", ("affine", a, b, c), ("const", d):
                    return emit(out, cse, ("affine", a, b, c + d))
                case "add", ("const", d), ("affine", a, b, c):
                    return emit(out, cse, ("affine", a, b, c + d))
                case "add", ("square", x), ("square", y):
                    return emit(out, cse, ("hypot2", x, y))
                case "sub", ("affine", a, b, c), ("affine", d, e, f):
                    return emit(out, cse, ("affine", a - d, b - e, c - f))
                case "sub", ("affine", a, b, c), ("const", d):
                    return emit(out, cse, ("affine", a, b, c - d))
                case "sub", ("const", d), ("affine", a, b, c):
                    return emit(out, cse, ("affine", - a, - b, - c + d))
                case "sub", ("const", x), _:
                    return emit(out, cse, ("neg", emit(out, cse, ("add_imm", y, - x))))
                case "sub", _, ("const", y):
                    return emit(out, cse, ("add_imm", x, - y))
                case "mul", ("affine", a, b, c), ("const", d):
                    return emit(out, cse, ("affine", a * d, b * d, c * d))
                case "mul", ("const", d), ("affine", a, b, c):
                    return emit(out, cse, ("affine", a * d, b * d, c * d))
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
