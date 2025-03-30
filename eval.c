#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <arm_neon.h>

#include "eval.h"

struct Tbl;

typedef void (*Op)(struct Env *, Inst *, struct Tbl *, size_t, Inst);

struct Tbl { Op ops[7]; };

static inline void DISPATCH(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip) {
  Inst inst = code[ip];
  tbl->ops[inst.tag](env, code, tbl, ip, inst);
}

static inline float32x4x4_t vmul(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vmulq_f32(x.val[0], y.val[0]);
  z.val[1] = vmulq_f32(x.val[1], y.val[1]);
  z.val[2] = vmulq_f32(x.val[2], y.val[2]);
  z.val[3] = vmulq_f32(x.val[3], y.val[3]);
  return z;
}

static inline float32x4x4_t vadd(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vaddq_f32(x.val[0], y.val[0]);
  z.val[1] = vaddq_f32(x.val[1], y.val[1]);
  z.val[2] = vaddq_f32(x.val[2], y.val[2]);
  z.val[3] = vaddq_f32(x.val[3], y.val[3]);
  return z;
}

static inline float32x4x4_t vdup(float x) {
  float32x4_t y = vdupq_n_f32(x);
  float32x4x4_t z;
  z.val[0] = y;
  z.val[1] = y;
  z.val[2] = y;
  z.val[3] = y;
  return z;
}

static inline float32x4x4_t vle(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vcleq_f32(x.val[0], y.val[0]);
  z.val[1] = vcleq_f32(x.val[1], y.val[1]);
  z.val[2] = vcleq_f32(x.val[2], y.val[2]);
  z.val[3] = vcleq_f32(x.val[3], y.val[3]);
  return z;
}

static inline float32x4x4_t vge(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vcgeq_f32(x.val[0], y.val[0]);
  z.val[1] = vcgeq_f32(x.val[1], y.val[1]);
  z.val[2] = vcgeq_f32(x.val[2], y.val[2]);
  z.val[3] = vcgeq_f32(x.val[3], y.val[3]);
  return z;
}

static inline float32x4x4_t vbitand(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0])));
  z.val[1] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1])));
  z.val[2] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2])));
  z.val[3] = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])));
  return z;
}

static inline float32x4x4_t vbitor(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0])));
  z.val[1] = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1])));
  z.val[2] = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2])));
  z.val[3] = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])));
  return z;
}

static void op_affine(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  float32x4x4_t a = vdup(inst.affine.a);
  float32x4x4_t b = vdup(inst.affine.b);
  float32x4x4_t c = vdup(inst.affine.c);
  float32x4x4_t x = vld1q_f32_x4(&env->x[0]);
  float32x4x4_t y = vld1q_f32_x4(&env->y[0]);
  vst1q_f32_x4(&env->v[ip][0], vadd(vadd(vmul(a, x), vmul(b, y)), c));
  DISPATCH(env, code, tbl, ip + 1);
}

static void op_hypot2(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  float32x4x4_t x = vld1q_f32_x4(&env->v[inst.hypot2.x][0]);
  float32x4x4_t y = vld1q_f32_x4(&env->v[inst.hypot2.y][0]);
  vst1q_f32_x4(&env->v[ip][0], vadd(vmul(x, x), vmul(y, y)));
  DISPATCH(env, code, tbl, ip + 1);
}

static void op_le(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  float32x4x4_t x = vld1q_f32_x4(&env->v[inst.le.x][0]);
  float32x4x4_t t = vdup(inst.le.t);
  vst1q_f32_x4(&env->v[ip][0], vle(x, t));
  DISPATCH(env, code, tbl, ip + 1);
}

static void op_ge(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  float32x4x4_t x = vld1q_f32_x4(&env->v[inst.ge.x][0]);
  float32x4x4_t t = vdup(inst.ge.t);
  vst1q_f32_x4(&env->v[ip][0], vge(x, t));
  DISPATCH(env, code, tbl, ip + 1);
}

static void op_and(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  float32x4x4_t x = vld1q_f32_x4(&env->v[inst.and.x][0]);
  float32x4x4_t y = vld1q_f32_x4(&env->v[inst.and.y][0]);
  vst1q_f32_x4(&env->v[ip][0], vbitand(x, y));
  DISPATCH(env, code, tbl, ip + 1);
}

static void op_or(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  float32x4x4_t x = vld1q_f32_x4(&env->v[inst.or.x][0]);
  float32x4x4_t y = vld1q_f32_x4(&env->v[inst.or.y][0]);
  vst1q_f32_x4(&env->v[ip][0], vbitor(x, y));
  DISPATCH(env, code, tbl, ip + 1);
}

static void op_ret(struct Env * env, Inst * code, struct Tbl * tbl, size_t ip, Inst inst) {
  (void) code;
  (void) tbl;
  (void) ip;
  vst1q_f32_x4(&env->z[0], vld1q_f32_x4(&env->v[inst.ret.x][0]));
}

static struct Tbl TBL = {
  .ops = {
    op_affine,
    op_hypot2,
    op_le,
    op_ge,
    op_and,
    op_or,
    op_ret
  }
};

void eval(struct Env * env, Inst * code) {
  DISPATCH(env, code, &TBL, 0);
}
