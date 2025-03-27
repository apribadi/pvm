#!/usr/bin/env python3

import sys
from dataclasses import dataclass
from collections import defaultdict

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
        case ("var-x" | "var-y"):
            code.append((op,))
        case "const":
            code.append((op, float(args[0])))
        case ("add" | "sub" | "mul" | "max" | "min" | "neg" | "square" | "sqrt"):
            code.append((op, *(Var(int(arg.lstrip("_"), 16)) for arg in args)))
        case _:
            raise

code.append(("le_imm", Var(len(code) - 1), 0.0))
code.append(("return", Var(len(code) - 1)))

def square(x):
    return x * x

def substitute_vars(inst, f):
    out = []
    for x in inst:
        if isinstance(x, Var):
            out.append(f(x))
        else:
            out.append(x)
    return tuple(out)

def iterate_vars(inst):
    for x in inst:
        if isinstance(x, Var):
            yield x

def push(x, y):
    n = len(x)
    x.append(y)
    return n

def emit(out, cse, inst):
    if inst in cse:
        return cse[inst]
    else:
        i = Var(push(out, inst))
        cse[inst] = i
        return i

def simplify(out, cse, inst):
    match inst:
        case (("var-x" | "var-y"),):
            return emit(out, cse, inst)
        case ("const", float(_)):
            return emit(out, cse, inst)
        case (op, Var(x)):
            match (op, out[x]):
                case ("square", ("neg", x)):
                    return emit(out, cse, ("square", x))
                case _:
                    return emit(out, cse, (op, x))
        case (op, Var(x), Var(y)):
            match (op, out[x], out[y]):
                case ("add", ("const", x), _):
                    return emit(out, cse, ("add_imm", y, x))
                case ("add", _, ("const", y)):
                    return emit(out, cse, ("add_imm", x, y))
                case ("add", ("square", x), ("square", y)):
                    return emit(out, cse, ("hypot2", x, y))
                case ("sub", ("const", x), _):
                    return emit(out, cse, ("neg", emit(out, cse, ("sub_imm", y, x))))
                case ("sub", _, ("const", y)):
                    return emit(out, cse, ("sub_imm", x, y))
                case ("mul", ("const", x), _):
                    return emit(out, cse, ("mul_imm", y, x))
                case ("mul", _, ("const", y)):
                    return emit(out, cse, ("mul_imm", x, y))
                case ("and", _, _):
                    return emit(out, cse, ("and", min(x, y), max(x, y)))
                case ("or", _, ("false",)):
                    return x
                case ("or", _, _):
                    return emit(out, cse, ("or", min(x, y), max(x, y)))
                case _:
                    return emit(out, cse, (op, x, y))
        case (op, Var(x), float(c)):
            match (op, out[x]):
                case ("ge_imm", ("neg", x)):
                    return simplify(out, cse, ("le_imm", x, - c))
                case ("ge_imm", ("sqrt", x)):
                    return simplify(out, cse, ("ge_imm", x, c * c))
                case ("ge_imm", ("add_imm", x, d)):
                    return simplify(out, cse, ("ge_imm", x, c - d))
                case ("ge_imm", ("sub_imm", x, d)):
                    return simplify(out, cse, ("ge_imm", x, c + d))
                case ("ge_imm", ("min", x, y)):
                    x = simplify(out, cse, ("ge_imm", x, c))
                    y = simplify(out, cse, ("ge_imm", y, c))
                    return simplify(out, cse, ("and", x, y))
                case ("ge_imm", ("max", x, y)):
                    x = simplify(out, cse, ("ge_imm", x, c))
                    y = simplify(out, cse, ("ge_imm", y, c))
                    return simplify(out, cse, ("or", x, y))
                case ("le_imm", ("const", x)):
                    if x <= c:
                        return emit(out, cse, ("true",))
                    else:
                        return emit(out, cse, ("false",))
                case ("le_imm", ("neg", x)):
                    return simplify(out, cse, ("ge_imm", x, - c))
                case ("le_imm", ("sqrt", x)):
                    return simplify(out, cse, ("le_imm", x, c * c))
                case ("le_imm", ("add_imm", x, d)):
                    return simplify(out, cse, ("le_imm", x, c - d))
                case ("le_imm", ("sub_imm", x, d)):
                    return simplify(out, cse, ("le_imm", x, c + d))
                case ("le_imm", ("min", x, y)):
                    x = simplify(out, cse, ("le_imm", x, c))
                    y = simplify(out, cse, ("le_imm", y, c))
                    return simplify(out, cse, ("or", x, y))
                case ("le_imm", ("max", x, y)):
                    x = simplify(out, cse, ("le_imm", x, c))
                    y = simplify(out, cse, ("le_imm", y, c))
                    return simplify(out, cse, ("and", x, y))
                case _:
                    return emit(out, cse, (op, x, c))
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
            for var in iterate_vars(inst):
                used[var] = True

    out = []
    cse = None
    map = [] # old var -> new var

    for i, inst in enumerate(code):
        if used[i]:
            inst = substitute_vars(inst, lambda i: map[i])
            map.append(Var(push(out, inst)))
        else:
            map.append(None)

    return out

code = lower(code)

counts = defaultdict(lambda: 0)

for i, inst in enumerate(code):
    counts[inst[0]] += 1
    # inst = substitute_vars(inst, lambda i: code[i])
    print(Var(i), "=", inst)

for x, y in counts.items():
    print(x, y)
