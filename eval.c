#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

#include "eval.h"

struct Tbl;

typedef __uint128_t (*Op)(Env *, Ins *, uint8_t out[16], struct Tbl *, size_t, Ins);

struct Tbl { Op ops[7]; };

static inline __uint128_t DISPATCH(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip) {
  Ins ins = code[ip];
  return tbl->ops[ins.tag](env, code, out, tbl, ip, ins);
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

static inline float32x4_t vle(float32x4x4_t x, float32x4x4_t y) {
  float32x4_t a = vcleq_f32(x.val[0], y.val[0]);
  float32x4_t b = vcleq_f32(x.val[1], y.val[1]);
  float32x4_t c = vcleq_f32(x.val[2], y.val[2]);
  float32x4_t d = vcleq_f32(x.val[3], y.val[3]);
  uint16x8_t e = vuzp1q_u16(vreinterpretq_u16_f32(a), vreinterpretq_u16_f32(b));
  uint16x8_t f = vuzp1q_u16(vreinterpretq_u16_f32(c), vreinterpretq_u16_f32(d));
  uint8x16_t g = vuzp1q_u8(vreinterpretq_u8_u16(e), vreinterpretq_u8_u16(f));
  return vreinterpretq_f32_u8(g);
}

static inline float32x4_t vge(float32x4x4_t x, float32x4x4_t y) {
  float32x4_t a = vcgeq_f32(x.val[0], y.val[0]);
  float32x4_t b = vcgeq_f32(x.val[1], y.val[1]);
  float32x4_t c = vcgeq_f32(x.val[2], y.val[2]);
  float32x4_t d = vcgeq_f32(x.val[3], y.val[3]);
  uint16x8_t e = vuzp1q_u16(vreinterpretq_u16_f32(a), vreinterpretq_u16_f32(b));
  uint16x8_t f = vuzp1q_u16(vreinterpretq_u16_f32(c), vreinterpretq_u16_f32(d));
  uint8x16_t g = vuzp1q_u8(vreinterpretq_u8_u16(e), vreinterpretq_u8_u16(f));
  return vreinterpretq_f32_u8(g);
}

static inline float32x4_t vbitand(float32x4_t x, float32x4_t y) {
  return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
}

static inline float32x4_t vbitor(float32x4_t x, float32x4_t y) {
  return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
}

static __uint128_t op_affine(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  float32x4x4_t a = vdup(ins.affine.a);
  float32x4x4_t b = vdup(ins.affine.b);
  float32x4x4_t c = vdup(ins.affine.c);
  float32x4x4_t x = vld1q_f32_x4(env->x);
  float32x4x4_t y = vld1q_f32_x4(env->y);
  vst1q_f32_x4(env->v[ip], vadd(vadd(vmul(a, x), vmul(b, y)), c));
  return DISPATCH(env, code, out, tbl, ip + 1);
}

static __uint128_t op_hypot2(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  float32x4x4_t x = vld1q_f32_x4(env->v[ins.hypot2.x]);
  float32x4x4_t y = vld1q_f32_x4(env->v[ins.hypot2.y]);
  vst1q_f32_x4(env->v[ip], vadd(vmul(x, x), vmul(y, y)));
  return DISPATCH(env, code, out, tbl, ip + 1);
}

static __uint128_t op_le_imm(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  float32x4x4_t x = vld1q_f32_x4(env->v[ins.le_imm.x]);
  float32x4x4_t t = vdup(ins.le_imm.t);
  vst1q_f32(env->v[ip], vle(x, t));
  return DISPATCH(env, code, out, tbl, ip + 1);
}

static __uint128_t op_ge_imm(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  float32x4x4_t x = vld1q_f32_x4(env->v[ins.ge_imm.x]);
  float32x4x4_t t = vdup(ins.ge_imm.t);
  vst1q_f32(env->v[ip], vge(x, t));
  return DISPATCH(env, code, out, tbl, ip + 1);
}

static __uint128_t op_and(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  float32x4_t x = vld1q_f32(env->v[ins.and.x]);
  float32x4_t y = vld1q_f32(env->v[ins.and.y]);
  vst1q_f32(env->v[ip], vbitand(x, y));
  return DISPATCH(env, code, out, tbl, ip + 1);
}

static __uint128_t op_or(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  float32x4_t x = vld1q_f32(env->v[ins.or.x]);
  float32x4_t y = vld1q_f32(env->v[ins.or.y]);
  vst1q_f32(env->v[ip], vbitor(x, y));
  return DISPATCH(env, code, out, tbl, ip + 1);
}

static __uint128_t op_result(Env * env, Ins * code, uint8_t out[16], struct Tbl * tbl, size_t ip, Ins ins) {
  (void) code;
  (void) tbl;
  (void) ip;
  uint8x16_t x = vreinterpretq_u8_f32(vld1q_f32(env->v[ins.result.x]));
  __uint128_t y;
  memcpy(&y, &x, 16);
  return y;
}

static struct Tbl TBL = {
  .ops = {
    op_affine,
    op_hypot2,
    op_le_imm,
    op_ge_imm,
    op_and,
    op_or,
    op_result
  }
};

void eval(Env * env, Ins * code, uint8_t out[16]) {
  size_t i = 0;
  while (1) {
    Ins ins = code[i];
  }
  __uint128_t x = DISPATCH(env, code, out, &TBL, 0);
  memcpy(out, &x, 16);
  /*
  for (size_t i = 0; i < n; ++ i) {
    DISPATCH(env, code, out, &TBL, i);
  }
  */
}
