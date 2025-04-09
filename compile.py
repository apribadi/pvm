#!/usr/bin/env python3

import sys
import numpy

from math import copysign

class Var(int):
    __match_args__ = ("as_var",)

    def __repr__(self):
        return f"{int(self)}"

    @property
    def as_var(self):
        return self

# Read input from stdin and generate instructions line by line

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
            raise RuntimeError("cant't parse input instruction: {line}")

code.append(("le_imm", Var(len(code) - 1), 0.0))
code.append(("result", Var(len(code) - 1)))

def push(x, y):
    n = len(x)
    x.append(y)
    return n

def emit(out, cse, ins):
    if ins in cse:
        return cse[ins]
    i = Var(push(out, ins))
    cse[ins] = i
    return i

def simplify(out, cse, ins):
    match ins:
        case "var-x",:
            return emit(out, cse, ("affine", 1.0, 0.0, 0.0))
        case "var-y",:
            return emit(out, cse, ("affine", 0.0, 1.0, 0.0))
        case "const", float(_):
            return emit(out, cse, ins)
        case op, Var(x):
            match (op, out[x]):
                case "neg", ("affine", a, b, c):
                    return emit(out, cse, ("affine", - a, - b, - c))
                case "square", ("affine", a, b, c):
                    if a + b < 0:
                        x = emit(out, cse, ("affine", - a + 0.0, - b + 0.0, - c + 0.0))
                        return emit(out, cse, ("square", x))
                    else:
                        x = emit(out, cse, ("affine", a + 0.0, b + 0.0, c + 0.0))
                        return emit(out, cse, ("square", x))
                case _:
                    return emit(out, cse, (op, x))
        case op, Var(x), float(c):
            match (op, out[x]):
                case "ge_imm", ("affine", a, b, d):
                    if a + b < 0:
                        x = emit(out, cse, ("affine", - a + 0.0, - b + 0.0, 0.0))
                        return emit(out, cse, ("le_imm", x, - (c - d) + 0.0))
                    else:
                        x = emit(out, cse, ("affine", a + 0.0, b + 0.0, 0.0))
                        return emit(out, cse, ("ge_imm", x, (c - d) + 0.0))
                case "ge_imm", ("sqrt", x):
                    if c <= 0.0:
                        return emit(out, cse, ("true",))
                    return emit(out, cse, ("ge_imm", x, c * c))
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
                case "le_imm", ("affine", a, b, d):
                    if a + b < 0:
                        x = emit(out, cse, ("affine", - a + 0.0, - b + 0.0, 0.0))
                        return emit(out, cse, ("ge_imm", x, - (c - d) + 0.0))
                    else:
                        x = emit(out, cse, ("affine", a + 0.0, b + 0.0, 0.0))
                        return emit(out, cse, ("le_imm", x, (c - d) + 0.0))
                case "le_imm", ("sqrt", x):
                    if c < 0.0:
                        return emit(out, cse, ("false",))
                    return emit(out, cse, ("le_imm", x, c * c))
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
            raise RuntimeError("can't simplify instruction: {ins}")

def substitute_vars(ins, f):
    out = []
    for x in ins:
        if isinstance(x, Var):
            out.append(f(x))
        else:
            out.append(x)
    return tuple(out)

# Simplification and Common Subexpression Elimination

out = []
cse = {}
map = [] # old var -> new var

for ins in code:
    ins = substitute_vars(ins, lambda i: map[i])
    map.append(simplify(out, cse, ins))

code = out

# Dead Code Elimination

used = [False for _ in code]
used[-1] = True

for i, ins in reversed(list(enumerate(code))):
    if used[i]:
        for x in ins:
            if isinstance(x, Var):
                used[x] = True

out = []
map = [] # old var -> new var

for i, ins in enumerate(code):
    if used[i]:
        ins = substitute_vars(ins, lambda i: map[i])
        map.append(Var(push(out, ins)))
    else:
        map.append(None)

code = out

print(f"static Ins PROSPERO[{len(code)}] = {{")

for i, ins in enumerate(code):
    match ins:
        case "affine", a, b, c:
            print(f"  [{i}] = {{ AFFINE, .affine = {{ {a:.9}f, {b:.9}f, {c:.9}f }} }},")
        case "hypot2", x, y:
            print(f"  [{i}] = {{ HYPOT2, .hypot2 = {{ {x}, {y} }} }},")
        case "le_imm", x, t:
            print(f"  [{i}] = {{ LE_IMM, .le_imm = {{ {x}, {t:.9}f }} }},")
        case "ge_imm", x, t:
            print(f"  [{i}] = {{ GE_IMM, .ge_imm = {{ {x}, {t:.9}f }} }},")
        case "and", x, y:
            print(f"  [{i}] = {{ AND, .and = {{ {x}, {y} }} }},")
        case "or", x, y:
            print(f"  [{i}] = {{ OR, .or = {{ {x}, {y} }} }},")
        case "result", x:
            print(f"  [{i}] = {{ RESULT, .result = {{ {x} }} }}")
        case _:
            raise RuntimeError(f"can't lower instruction: {ins}")

print(f"}};")

'''
for i, ins in enumerate(code):
    match ins:
        case "affine", a, b, c:
            print(f"  [{i}] = {{ AFFINE, .affine = {{ {a:.9}f, {b:.9}f, {c:.9}f }} }},")
        case "hypot2", x, y:
            print(f"  [{i}] = {{ HYPOT2, .hypot2 = {{ {code[x]}, {code[y]} }} }},")
        case "le_imm", x, t:
            print(f"  [{i}] = {{ LE_IMM, .le_imm = {{ {code[x]}, {t:.9}f }} }},")
        case "ge_imm", x, t:
            print(f"  [{i}] = {{ GE_IMM, .ge_imm = {{ {code[x]}, {t:.9}f }} }},")
        case "and", x, y:
            print(f"  [{i}] = {{ AND, .and = {{ {code[x]}, {code[y]} }} }},")
        case "and3", x, y, z:
            print(f"  [{i}] = {{ AND3, .and3 = {{ {code[x]}, {code[y]}, {code[z]} }} }},")
        case "or", x, y:
            print(f"  [{i}] = {{ OR, .or = {{ {code[x]}, {code[y]} }} }},")
        case "or3", x, y, z:
            print(f"  [{i}] = {{ OR3, .or3 = {{ {code[x]}, {code[y]}, {code[z]} }} }},")
        case "result", x:
            print(f"  [{i}] = {{ RESULT, .result = {{ {code[x]} }} }}")
        case _:
            raise RuntimeError(f"can't lower instruction: {ins}")
'''
