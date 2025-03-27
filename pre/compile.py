#!/usr/bin/env python3

import sys
from dataclasses import dataclass

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
code.append(("return", Var(len(code)- 1)))

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

def emit(out, cse, inst):
    if cse is None:
        i = Var(len(out))
        out.append(inst)
        return i
    if inst in cse:
        return cse[inst]
    else:
        i = Var(len(out))
        out.append(inst)
        cse[inst] = i
        return i

def simplify(out, cse, inst):
    match inst:
        case (("var-x" | "var-y" | "const"), *_):
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
                case ("sub", ("const", x), _):
                    return emit(out, cse, ("neg", emit(out, cse, ("sub_imm", y, x))))
                case ("sub", _, ("const", y)):
                    return emit(out, cse, ("sub_imm", x, y))
                case ("mul", ("const", x), _):
                    return emit(out, cse, ("mul_imm", y, x))
                case ("mul", _, ("const", y)):
                    return emit(out, cse, ("mul_imm", x, y))
                case ("and", _, _) if x > y:
                    return emit(out, cse, ("and", y, x))
                case ("or", _, ("false",)):
                    return x
                case ("or", _, _) if x > y:
                    return emit(out, cse, ("or", y, x))
                case _:
                    return emit(out, cse, (op, x, y))
        case (op, Var(x), float(c)):
            match (op, out[x]):
                case ("ge_imm", ("neg", x)):
                    return simplify(out, cse, ("le_imm", x, - c))
                case ("ge_imm", ("min", x, y)):
                    return simplify(out, cse, ("and", simplify(out, cse, ("ge_imm", x, c)), simplify(out, cse, ("ge_imm", y, c))))
                case ("ge_imm", ("max", x, y)):
                    return simplify(out, cse, ("or", simplify(out, cse, ("ge_imm", x, c)), simplify(out, cse, ("ge_imm", y, c))))
                case ("ge_imm", ("sub_imm", x, d)) if out[x][0] == "sqrt":
                    # sqrt(y) - d >= c <=> y >= square(c + d)
                    y = out[x][1]
                    return simplify(out, cse, ("ge_imm", y, (c + d) * (c + d)))
                case ("ge_imm", ("add_imm", x, d)):
                    return simplify(out, cse, ("ge_imm", x, c - d))
                case ("le_imm", ("const", x)):
                    return emit(out, cse, (("true",) if x <= c else ("false",)))
                case ("le_imm", ("neg", x)):
                    return simplify(out, cse, ("ge_imm", x, - c))
                case ("le_imm", ("min", x, y)):
                    return simplify(out, cse, ("or", simplify(out, cse, ("le_imm", x, c)), simplify(out, cse, ("le_imm", y, c))))
                case ("le_imm", ("max", x, y)):
                    return simplify(out, cse, ("and", simplify(out, cse, ("le_imm", x, c)), simplify(out, cse, ("le_imm", y, c))))
                case ("le_imm", ("sub_imm", x, d)) if out[x][0] == "sqrt":
                    # sqrt(y) - d <= c <=> y <= square(c + d)
                    y = out[x][1]
                    return simplify(out, cse, ("le_imm", y, (c + d) * (c + d)))
                case ("le_imm", ("add_imm", x, d)):
                    return simplify(out, cse, ("le_imm", x, c - d))
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
            map.append(emit(out, cse, inst))
        else:
            map.append(None)

    return out

code = lower(code)

from collections import defaultdict

counts = defaultdict(lambda: 0)

for i, x in enumerate(code):
    counts[x[0]] += 1
    print(Var(i), "=", x)

for x, y in counts.items():
    print(x, y)
