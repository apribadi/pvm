#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>

#include "render.h"

struct Tbl;

typedef size_t (*Op)(Env *, Ins *, struct Tbl *, size_t, Ins);

struct Tbl { Op ops[7]; };

static inline size_t DISPATCH(Env * env, Ins * code, struct Tbl * tbl, size_t ip) {
  Ins ins = code[ip];
  return tbl->ops[ins.tag](env, code, tbl, ip, ins);
}

static inline float32x4x4_t vmulx(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t r;
  r.val[0] = vmulq_f32(x.val[0], y.val[0]);
  r.val[1] = vmulq_f32(x.val[1], y.val[1]);
  r.val[2] = vmulq_f32(x.val[2], y.val[2]);
  r.val[3] = vmulq_f32(x.val[3], y.val[3]);
  return r;
}

static inline float32x4x4_t vmulx_n(float32x4x4_t x, float y) {
  float32x4x4_t r;
  r.val[0] = vmulq_n_f32(x.val[0], y);
  r.val[1] = vmulq_n_f32(x.val[1], y);
  r.val[2] = vmulq_n_f32(x.val[2], y);
  r.val[3] = vmulq_n_f32(x.val[3], y);
  return r;
}

static inline float32x4x4_t vaddx(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t r;
  r.val[0] = vaddq_f32(x.val[0], y.val[0]);
  r.val[1] = vaddq_f32(x.val[1], y.val[1]);
  r.val[2] = vaddq_f32(x.val[2], y.val[2]);
  r.val[3] = vaddq_f32(x.val[3], y.val[3]);
  return r;
}

static inline float32x4x4_t vdupx_n(float x) {
  float32x4_t y = vdupq_n_f32(x);
  float32x4x4_t r;
  r.val[0] = y;
  r.val[1] = y;
  r.val[2] = y;
  r.val[3] = y;
  return r;
}

static inline uint8x16_t vclex(float32x4x4_t x, float32x4x4_t y) {
  float32x4x4_t z;
  z.val[0] = vcleq_f32(x.val[0], y.val[0]);
  z.val[1] = vcleq_f32(x.val[1], y.val[1]);
  z.val[2] = vcleq_f32(x.val[2], y.val[2]);
  z.val[3] = vcleq_f32(x.val[3], y.val[3]);
  uint16x8_t a = vuzp1q_u16(vreinterpretq_u16_f32(z.val[0]), vreinterpretq_u16_f32(z.val[1]));
  uint16x8_t b = vuzp1q_u16(vreinterpretq_u16_f32(z.val[2]), vreinterpretq_u16_f32(z.val[3]));
  return vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
}

static inline uint8x16x4_t vandx(uint8x16x4_t x, uint8x16x4_t y) {
  uint8x16x4_t r;
  r.val[0] = vandq_u8(x.val[0], y.val[0]);
  r.val[1] = vandq_u8(x.val[1], y.val[1]);
  r.val[2] = vandq_u8(x.val[2], y.val[2]);
  r.val[3] = vandq_u8(x.val[3], y.val[3]);
  return r;
}

static inline uint8x16x4_t vorrx(uint8x16x4_t x, uint8x16x4_t y) {
  uint8x16x4_t r;
  r.val[0] = vorrq_u8(x.val[0], y.val[0]);
  r.val[1] = vorrq_u8(x.val[1], y.val[1]);
  r.val[2] = vorrq_u8(x.val[2], y.val[2]);
  r.val[3] = vorrq_u8(x.val[3], y.val[3]);
  return r;
}

static size_t op_affine(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  float a = ins.affine.a;
  float b = ins.affine.b;
  float c = ins.affine.c;
  float32x4x4_t x = vld1q_f32_x4(env->x);
  float32x4_t y = vld1q_f32(env->y);
  float32x4x4_t u = vmulx_n(x, a);
  float32x4_t v = vaddq_f32(vmulq_n_f32(y, b), vdupq_n_f32(c));
  for (size_t k = 0; k < 4; k ++) {
    vst1q_f32_x4(&env->v[ip][16 * k],  vaddx(u, vdupx_n(v[k])));
  }
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_hypot2(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  for (size_t k = 0; k < 4; k ++) {
    float32x4x4_t x = vld1q_f32_x4(&env->v[ins.hypot2.x][16 * k]);
    float32x4x4_t y = vld1q_f32_x4(&env->v[ins.hypot2.y][16 * k]);
    vst1q_f32_x4(&env->v[ip][16 * k], vaddx(vmulx(x, x), vmulx(y, y)));
  }
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_le_imm(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  uint8x16x4_t r;
  float32x4x4_t t = vdupx_n(ins.le_imm.t);
  for (size_t k = 0; k < 4; k ++) {
    float32x4x4_t x = vld1q_f32_x4(&env->v[ins.le_imm.x][16 * k]);
    r.val[k] = vclex(x, t);
  }
  vst1q_u8_x4((uint8_t *) env->v[ip], r);
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_ge_imm(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  uint8x16x4_t r;
  float32x4x4_t t = vdupx_n(ins.ge_imm.t);
  for (size_t k = 0; k < 4; k ++) {
    float32x4x4_t x = vld1q_f32_x4(&env->v[ins.ge_imm.x][16 * k]);
    r.val[k] = vclex(t, x);
  }
  vst1q_u8_x4((uint8_t *) env->v[ip], r);
  return DISPATCH(env, code, tbl, ip + 1);
}


static size_t op_and(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  uint8x16x4_t x = vld1q_u8_x4((uint8_t *) env->v[ins.and.x]);
  uint8x16x4_t y = vld1q_u8_x4((uint8_t *) env->v[ins.and.y]);
  vst1q_u8_x4((uint8_t *) env->v[ip], vandx(x, y));
  return DISPATCH(env, code, tbl, ip + 1);
}

static size_t op_or(Env * env, Ins * code, struct Tbl * tbl, size_t ip, Ins ins) {
  uint8x16x4_t x = vld1q_u8_x4((uint8_t *) env->v[ins.or.x]);
  uint8x16x4_t y = vld1q_u8_x4((uint8_t *) env->v[ins.or.y]);
  vst1q_u8_x4((uint8_t *) env->v[ip], vorrx(x, y));
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
  for (size_t i = 0; i < RESOLUTION; i += 4) {
    Env * tenv = &env[omp_get_thread_num()];
    float y = ymax - half - step * (float) i;
    for (size_t k = 0; k < 4; k ++) {
      tenv->y[k] = y - step * (float) k;
    }
    for (size_t j = 0; j < RESOLUTION; j += 16) {
      float x = xmin + half + step * (float) j;
      for (size_t k = 0; k < 16; k ++) {
        tenv->x[k] = x + step * (float) k;
      }
      size_t result = DISPATCH(tenv, code, &TBL, 0);
      for (size_t k = 0; k < 4; k ++) {
        memcpy(&image[i + k][j], &tenv->v[result][4 * k], 16);
      }
    }
  }
}
