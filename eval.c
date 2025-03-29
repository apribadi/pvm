#include <stddef.h>
#include <stdint.h>

#include "eval.h"

float do_add(float * env, Inst inst) {
    return env[inst.add.x] + env[inst.add.y];
}

float do_add_imm(float * env, Inst inst) {
    return env[inst.add_imm.x] + inst.add_imm.c;
}

float (*TBL[2])(float *, Inst) = { do_add, do_add_imm };

float foo(float * env, float (*tbl[2])(float *, Inst), Inst inst) {
    return tbl[inst.tag](env, inst);
}
