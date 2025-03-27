#!/usr/bin/env python3

import sys
from dataclasses import dataclass

class Var(int):
    __match_args__ = ("index",)

    def __repr__(self):
        return f"Var({self.index})"

    @property
    def index(self):
        return int(self)

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

code.append(("le0", Var(len(code) - 1)))
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
        case (op, Var(_) as x):
            match (op, out[x]):
                case ("ge0", ("neg", x)):
                    return simplify(out, cse, ("le0", x))
                case ("ge0", ("min", x, y)):
                    return simplify(out, cse, ("and", simplify(out, cse, ("ge0", x)), simplify(out, cse, ("ge0", y))))
                case ("ge0", ("max", x, y)):
                    return simplify(out, cse, ("or", simplify(out, cse, ("ge0", x)), simplify(out, cse, ("ge0", y))))
                case ("ge0", ("sub_imm", y, c)):
                    match out[y]:
                        case ("sqrt", z):
                            return emit(out, cse, ("ge0", emit(out, cse, ("sub_imm", z, c * c))))
                        case _:
                            return emit(out, cse, ("ge0", x))
                case ("le0", ("const", x)):
                    if x <= 0:
                        return emit(out, cse, ("true",))
                    else:
                        return emit(out, cse, ("false",))
                case ("le0", ("neg", x)):
                    return simplify(out, cse, ("ge0", x))
                case ("le0", ("min", x, y)):
                    return simplify(out, cse, ("or", simplify(out, cse, ("le0", x)), simplify(out, cse, ("le0", y))))
                case ("le0", ("max", x, y)):
                    return simplify(out, cse, ("and", simplify(out, cse, ("le0", x)), simplify(out, cse, ("le0", y))))
                case ("le0", ("sub_imm", y, c)):
                    match out[y]:
                        case ("sqrt", z):
                            return emit(out, cse, ("le0", emit(out, cse, ("sub_imm", z, c * c))))
                        case _:
                            return emit(out, cse, ("le0", x))
                case ("neg", ("neg", x)):
                    return x
                case ("square", ("neg", x)):
                    return simplify(out, cse, ("square", x))
                case _:
                    return emit(out, cse, (op, x))
        case (op, Var(_) as x, Var(_) as y):
            match (op, out[x], out[y]):
                case ("add", ("const", x), ("const", y)):
                    return emit(out, cse, ("const", x + y))
                case ("add", ("const", x), ("add_imm", y, z)):
                    return emit(out, cse, ("add_imm", y, x + z))
                case ("add", ("const", x), _):
                    return emit(out, cse, ("add_imm", y, x))
                case ("add", ("add_imm", x, y), ("const", z)):
                    return emit(out, cse, ("add_imm", x, y + z))
                case ("add", _, ("const", y)):
                    return emit(out, cse, ("add_imm", x, y))
                case ("add", ("neg", x), ("neg", y)):
                    return emit(out, cse, ("neg", simplify(out, cse, ("add", x, y))))
                case ("add", ("neg", x), _):
                    return emit(out, cse, ("sub", y, x))
                case ("add", _, ("neg", y)):
                    return emit(out, cse, ("sub", x, y))
                case ("add", _, _) if x > y:
                    return emit(out, cse, ("add", y, x))
                case ("and", _, _) if x > y:
                    return emit(out, cse, ("and", y, x))
                case ("sub", ("const", x), ("const", y)):
                    return emit(out, cse, ("const", x - y))
                case ("sub", ("const", x), _):
                    return emit(out, cse, ("neg", emit(out, cse, ("sub_imm", y, x))))
                case ("sub", ("sub_imm", x, y), ("const", z)):
                    return emit(out, cse, ("sub_imm", x, y + z))
                case ("sub", _, ("const", y)):
                    return emit(out, cse, ("sub_imm", x, y))
                case ("sub", _, ("neg", y)):
                    return emit(out, cse, ("add", x, y))
                case ("mul", ("const", x), ("const", y)):
                    return emit(out, cse, ("const", x * y))
                case ("mul", ("const", x), _):
                    return emit(out, cse, ("mul_imm", y, x))
                case ("mul", _, ("const", y)):
                    return emit(out, cse, ("mul_imm", x, y))
                case ("mul", _, _) if x == y:
                    return emit(out, cse, ("square", x))
                case ("mul", _, _) if x > y:
                    return emit(out, cse, ("mul", y, x))
                case ("or", _, ("false",)):
                    return x
                case ("or", _, _) if x > y:
                    return emit(out, cse, ("or", y, x))
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
