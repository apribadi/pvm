#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>

#include "render.h"

typedef struct {
  float32x4x4_t val[2];
} vfloat;

struct Tbl;

typedef size_t (*Op)(Env *, Ins *, struct Tbl *, size_t, Ins);

struct Tbl { Op ops[7]; };

static inline size_t DISPATCH(Env * env, Ins * code, struct Tbl * tbl, size_t ip) {
  Ins ins = code[ip];
  return tbl->ops[ins.tag](env, code, tbl, ip, ins);
}

static inline vfloat vload(float src[32]) {
  vfloat x;
  x.val[0] = vld1q_f32_x4(&src[0]);
  x.val[1] = vld1q_f32_x4(&src[16]);
  return x;
}

static inline void vstore(float dst[32], vfloat x) {
  vst1q_f32_x4(&dst[0], x.val[0]);
  vst1q_f32_x4(&dst[16], x.val[1]);
}

static inline vfloat vmul(vfloat x, vfloat y) {
  vfloat z;
  z.val[0].val[0] = vmulq_f32(x.val[0].val[0], y.val[0].val[0]);
  z.val[0].val[1] = vmulq_f32(x.val[0].val[1], y.val[0].val[1]);
  z.val[0].val[2] = vmulq_f32(x.val[0].val[2], y.val[0].val[2]);
  z.val[0].val[3] = vmulq_f32(x.val[0].val[3], y.val[0].val[3]);
  z.val[1].val[0] = vmulq_f32(x.val[1].val[0], y.val[1].val[0]);
  z.val[1].val[1] = vmulq_f32(x.val[1].val[1], y.val[1].val[1]);
  z.val[1].val[2] = vmulq_f32(x.val[1].val[2], y.val[1].val[2]);
  z.val[1].val[3] = vmulq_f32(x.val[1].val[3], y.val[1].val[3]);
  return z;
}

static inline vfloat vadd(vfloat x, vfloat y) {
  vfloat z;
  z.val[0].val[0] = vaddq_f32(x.val[0].val[0], y.val[0].val[0]);
  z.val[0].val[1] = vaddq_f32(x.val[0].val[1], y.val[0].val[1]);
  z.val[0].val[2] = vaddq_f32(x.val[0].val[2], y.val[0].val[2]);
  z.val[0].val[3] = vaddq_f32(x.val[0].val[3], y.val[0].val[3]);
  z.val[1].val[0] = vaddq_f32(x.val[1].val[0], y.val[1].val[0]);
  z.val[1].val[1] = vaddq_f32(x.val[1].val[1], y.val[1].val[1]);
  z.val[1].val[2] = vaddq_f32(x.val[1].val[2], y.val[1].val[2]);
  z.val[1].val[3] = vaddq_f32(x.val[1].val[3], y.val[1].val[3]);
  return z;
}

static inline vfloat vdup(float x) {
  float32x4_t a = vdupq_n_f32(x);
  vfloat y;
  y.val[0].val[0] = a;
  y.val[0].val[1] = a;
  y.val[0].val[2] = a;
  y.val[0].val[3] = a;
  y.val[1].val[0] = a;
  y.val[1].val[1] = a;
  y.val[1].val[2] = a;
  y.val[1].val[3] = a;
  return y;
}

static inline uint8x16x2_t vle(vfloat x, vfloat y) {
  vfloat z;
  z.val[0].val[0] = vcleq_f32(x.val[0].val[0], y.val[0].val[0]);
  z.val[0].val[1] = vcleq_f32(x.val[0].val[1], y.val[0].val[1]);
  z.val[0].val[2] = vcleq_f32(x.val[0].val[2], y.val[0].val[2]);
  z.val[0].val[3] = vcleq_f32(x.val[0].val[3], y.val[0].val[3]);
  z.val[1].val[0] = vcleq_f32(x.val[1].val[0], y.val[1].val[0]);
  z.val[1].val[1] = vcleq_f32(x.val[1].val[1], y.val[1].val[1]);
  z.val[1].val[2] = vcleq_f32(x.val[1].val[2], y.val[1].val[2]);
  z.val[1].val[3] = vcleq_f32(x.val[1].val[3], y.val[1].val[3]);
  uint8x16x2_t u;
  uint16x8_t a = vuzp1q_u16(vreinterpretq_u16_f32(z.val[0].val[0]), vreinterpretq_u16_f32(z.val[0].val[1]));
  uint16x8_t b = vuzp1q_u16(vreinterpretq_u16_f32(z.val[0].val[2]), vreinterpretq_u16_f32(z.val[0].val[3]));
  uint16x8_t c = vuzp1q_u16(vreinterpretq_u16_f32(z.val[1].val[0]), vreinterpretq_u16_f32(z.val[1].val[1]));
  uint16x8_t d = vuzp1q_u16(vreinterpretq_u16_f32(z.val[1].val[2]), vreinterpretq_u16_f32(z.val[1].val[3]));
  u.val[0] = vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
  u.val[1] = vuzp1q_u8(vreinterpretq_u8_u16(c), vreinterpretq_u8_u16(d));
  return u;
}

static inline uint8x16x2_t vbitand(uint8x16x2_t x, uint8x16x2_t y) {
  uint8x16x2_t z;
  z.val[0] = vandq_u8(x.val[0], y.val[0]);
  z.val[1] = vandq_u8(x.val[1], y.val[1]);
  return z;
}

static inline uint8x16x2_t vbitor(uint8x16x2_t x, uint8x16x2_t y) {
  uint8x16x2_t z;
  z.val[0] = vorrq_u8(x.val[0], y.val[0]);
  z.val[1] = vorrq_u8(x.val[1], y.val[1]);
  return z;
}

static size_t op_affine(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  vfloat a = vdup(ins.affine.a);
  vfloat b = vdup(ins.affine.b);
  vfloat c = vdup(ins.affine.c);
  vfloat x = vload(env->x);
  vfloat y = vload(env->y);
  vstore(env->v[ip], vadd(vadd(vmul(a, x), vmul(b, y)), c));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_hypot2(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  vfloat x = vload(env->v[ins.hypot2.x]);
  vfloat y = vload(env->v[ins.hypot2.y]);
  vstore(env->v[ip], vadd(vmul(x, x), vmul(y, y)));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_le_imm(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  vfloat x = vload(env->v[ins.le_imm.x]);
  vfloat t = vdup(ins.le_imm.t);
  vst1q_u8_x2((uint8_t *) env->v[ip], vle(x, t));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_ge_imm(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  vfloat x = vload(env->v[ins.ge_imm.x]);
  vfloat t = vdup(ins.ge_imm.t);
  vst1q_u8_x2((uint8_t *) env->v[ip], vle(t, x));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_and(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  uint8x16x2_t x = vld1q_u8_x2((uint8_t *) env->v[ins.and.x]);
  uint8x16x2_t y = vld1q_u8_x2((uint8_t *) env->v[ins.and.y]);
  vst1q_u8_x2((uint8_t *) env->v[ip], vbitand(x, y));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_or(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  uint8x16x2_t x = vld1q_u8_x2((uint8_t *) env->v[ins.or.x]);
  uint8x16x2_t y = vld1q_u8_x2((uint8_t *) env->v[ins.or.y]);
  vst1q_u8_x2((uint8_t *) env->v[ip], vbitor(x, y));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_result(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  (void) env;
  (void) code;
  (void) tbl;
  (void) ip;
  return ins.result.x;
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

void render(Env env[NUM_THREADS], Ins * code, uint8_t image[RESOLUTION][RESOLUTION]) {
  float side = 2.0;
  float xmin = -1.0;
  float ymin = -1.0;
  float ymax = ymin + side;
  float step = side / RESOLUTION;
  float half = step * 0.5f;

#pragma omp parallel for num_threads(NUM_THREADS)
  for (size_t i = 0; i < RESOLUTION; i ++) {
    Env * tenv = &env[omp_get_thread_num()];
    float y = ymax - half - step * (float) i;
    for (size_t j = 0; j < RESOLUTION; j += 32) {
      float x = xmin + half + step * (float) j;
      for (size_t k = 0; k < 32; k ++) {
        tenv->x[k] = x + step * (float) k;
        tenv->y[k] = y;
      }
      size_t result = DISPATCH(tenv, code, &TBL, 0);
      memcpy(&image[i][j], tenv->v[result], 32);
    }
  }
}
