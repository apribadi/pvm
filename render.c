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
  float side = 2.0f;
  float step = side /  RESOLUTION;
  float half = side /  RESOLUTION / 2.0f;
  float xmin0 = -1.0f;
  float ymax0 = 1.0f;
  uint8_t * p0 = (uint8_t *) image;

#pragma omp parallel for num_threads(NUM_THREADS)
  for (size_t t1 = 0; t1 < 16; t1 ++) {
    Env * e = &env[omp_get_thread_num()];
    size_t i1 = t1 & 3;
    size_t j1 = t1 >> 2;
    float xmin1 = xmin0 + (side / 4.0f) * (float) i1;
    float ymax1 = ymax0 - (side / 4.0f) * (float) j1;
    uint8_t * p1 = p0 + (RESOLUTION / 4) * i1 + (RESOLUTION / 4) * RESOLUTION * j1;
    for (size_t t2 = 0; t2 < 16; t2 ++) {
      size_t i2 = t2 & 3;
      size_t j2 = t2 >> 2;
      float xmin2 = xmin1 + (side / 16.0f) * (float) i2;
      float ymax2 = ymax1 - (side / 16.0f) * (float) j2;
      uint8_t * p2 = p1 + (RESOLUTION / 16) * i2 + (RESOLUTION / 16) * RESOLUTION * j2;
      for (size_t t3 = 0; t3 < 16; t3 ++) {
        size_t i3 = t3 & 3;
        size_t j3 = t3 >> 2;
        float xmin3 = xmin2 + (side / 64.0f) * (float) i3;
        float ymax3 = ymax2 - (side / 64.0f) * (float) j3;
        uint8_t * p3 = p2 + (RESOLUTION / 64) * i3 + (RESOLUTION / 64) * RESOLUTION * j3;
        for (size_t k = 0; k < 16; k ++) {
          e->x[k] = (xmin3 + half) + step * (float) k;
        }
        for (size_t t4 = 0; t4 < 4; t4 ++) {
          float ymax4 = ymax3 - (side / 256.0f) * (float) t4;
          uint8_t * p4 = p3 + (RESOLUTION / 256) * RESOLUTION * t4;
          for (size_t k = 0; k < 4; k ++) {
            e->y[k] = (ymax4 - half) - step * (float) k;
          }
          size_t result = DISPATCH(e, code, &TBL, 0);
          for (size_t k = 0; k < 4; k ++) {
            uint8_t * p5 = p4 + RESOLUTION * k;
            memcpy(p5, &e->v[result][4 * k], 16);
          }
        }
      }
    }
  }
}
