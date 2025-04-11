#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>

#include "render.h"

// -------- UTILITY FUNCTIONS --------

static inline float32x4x4_t vmulx(float32x4x4_t x, float32x4x4_t y) {
  return (float32x4x4_t) {{
    vmulq_f32(x.val[0], y.val[0]),
    vmulq_f32(x.val[1], y.val[1]),
    vmulq_f32(x.val[2], y.val[2]),
    vmulq_f32(x.val[3], y.val[3])
  }};
}

static inline float32x4x4_t vmulx_n(float32x4x4_t x, float y) {
  return (float32x4x4_t) {{
    vmulq_n_f32(x.val[0], y),
    vmulq_n_f32(x.val[1], y),
    vmulq_n_f32(x.val[2], y),
    vmulq_n_f32(x.val[3], y)
  }};
}

static inline float32x4x4_t vaddx(float32x4x4_t x, float32x4x4_t y) {
  return (float32x4x4_t) {{
    vaddq_f32(x.val[0], y.val[0]),
    vaddq_f32(x.val[1], y.val[1]),
    vaddq_f32(x.val[2], y.val[2]),
    vaddq_f32(x.val[3], y.val[3])
  }};
}

static inline float32x4x4_t vdupx_n(float x) {
  float32x4_t y = vdupq_n_f32(x);
  return (float32x4x4_t) {{ y, y, y, y }};
}

static inline uint8x16_t vclex(float32x4x4_t x, float32x4x4_t y) {
  float32x4_t z0 = vcleq_f32(x.val[0], y.val[0]);
  float32x4_t z1 = vcleq_f32(x.val[1], y.val[1]);
  float32x4_t z2 = vcleq_f32(x.val[2], y.val[2]);
  float32x4_t z3 = vcleq_f32(x.val[3], y.val[3]);
  uint16x8_t a = vuzp1q_u16(vreinterpretq_u16_f32(z0), vreinterpretq_u16_f32(z1));
  uint16x8_t b = vuzp1q_u16(vreinterpretq_u16_f32(z2), vreinterpretq_u16_f32(z3));
  return vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
}

static inline uint8x16x4_t vandx(uint8x16x4_t x, uint8x16x4_t y) {
  return (uint8x16x4_t) {{
    vandq_u8(x.val[0], y.val[0]),
    vandq_u8(x.val[1], y.val[1]),
    vandq_u8(x.val[2], y.val[2]),
    vandq_u8(x.val[3], y.val[3])
  }};
}

static inline uint8x16x4_t vorrx(uint8x16x4_t x, uint8x16x4_t y) {
  return (uint8x16x4_t) {{
    vorrq_u8(x.val[0], y.val[0]),
    vorrq_u8(x.val[1], y.val[1]),
    vorrq_u8(x.val[2], y.val[2]),
    vorrq_u8(x.val[3], y.val[3])
  }};
}

// -------- RASTERIZE --------

struct ra_Tbl { size_t (*ops[7])(Inst *, ra_X *, ra_V *, struct ra_Tbl *, size_t, Inst); };

static inline size_t ra_dispatch(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip) {
  Inst inst = cp[ip];
  return tp->ops[inst.tag](cp, xp, vp, tp, ip, inst);
}

static size_t ra_affine(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  float32x4x4_t x = vld1q_f32_x4(xp->x);
  float32x4_t y = vld1q_f32(xp->y);
  float32x4x4_t u = vaddx(vmulx_n(x, inst.affine.a), vdupx_n(inst.affine.c));
  float32x4_t v = vmulq_n_f32(y, inst.affine.b);
  for (size_t k = 0; k < 4; k ++) {
    vst1q_f32_x4(&vp[ip].f32x64[16 * k],  vaddx(u, vdupx_n(v[k])));
  }
  return ra_dispatch(cp, xp, vp, tp, ip + 1);
}

static size_t ra_hypot2(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  for (size_t k = 0; k < 4; k ++) {
    float32x4x4_t x = vld1q_f32_x4(&vp[inst.hypot2.x].f32x64[16 * k]);
    float32x4x4_t y = vld1q_f32_x4(&vp[inst.hypot2.y].f32x64[16 * k]);
    vst1q_f32_x4(&vp[ip].f32x64[16 * k], vaddx(vmulx(x, x), vmulx(y, y)));
  }
  return ra_dispatch(cp, xp, vp, tp, ip + 1);
}

static size_t ra_le_imm(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  float32x4x4_t t = vdupx_n(inst.le_imm.t);
  uint8x16x4_t r;
  for (size_t k = 0; k < 4; k ++) {
    r.val[k] = vclex(vld1q_f32_x4(&vp[inst.le_imm.x].f32x64[16 * k]), t);
  }
  vst1q_u8_x4(vp[ip].u8x64, r);
  return ra_dispatch(cp, xp, vp, tp, ip + 1);
}

static size_t ra_ge_imm(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  uint8x16x4_t r;
  float32x4x4_t t = vdupx_n(inst.ge_imm.t);
  for (size_t k = 0; k < 4; k ++) {
    r.val[k] = vclex(t, vld1q_f32_x4(&vp[inst.ge_imm.x].f32x64[16 * k]));
  }
  vst1q_u8_x4(vp[ip].u8x64, r);
  return ra_dispatch(cp, xp, vp, tp, ip + 1);
}

static size_t ra_and(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  uint8x16x4_t x = vld1q_u8_x4(vp[inst.and.x].u8x64);
  uint8x16x4_t y = vld1q_u8_x4(vp[inst.and.y].u8x64);
  vst1q_u8_x4(vp[ip].u8x64, vandx(x, y));
  return ra_dispatch(cp, xp, vp, tp, ip + 1);
}

static size_t ra_or(Inst * cp, ra_X * xp, ra_V * vp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  uint8x16x4_t x = vld1q_u8_x4(vp[inst.or.x].u8x64);
  uint8x16x4_t y = vld1q_u8_x4(vp[inst.or.y].u8x64);
  vst1q_u8_x4(vp[ip].u8x64, vorrx(x, y));
  return ra_dispatch(cp, xp, vp, tp, ip + 1);
}

static size_t ra_result(Inst *, ra_X *, ra_V *, struct ra_Tbl *, size_t, Inst inst) {
  return inst.result.x;
}

static struct ra_Tbl ra_TBL = {{
  ra_affine,
  ra_hypot2,
  ra_le_imm,
  ra_ge_imm,
  ra_and,
  ra_or,
  ra_result
}};

static size_t rasterize(Inst * cp, ra_X * xp, ra_V * vp) {
  return ra_dispatch(cp, xp, vp, &ra_TBL, 0);
}

// -------- RENDER FUNCTION --------

void render(
    size_t num_threads,
    ra_V env[num_threads][PROGRAM_MAX_LEN],
    Inst * code,
    uint8_t image[RESOLUTION][RESOLUTION]
  )
{
  float side = 2.0f;
  float step = side / RESOLUTION;
  float xmin0 = -1.0f;
  float ymax0 = 1.0f;
  uint8_t * p0 = &image[0][0];

#pragma omp parallel for num_threads(num_threads)
  for (size_t t1 = 0; t1 < 16; t1 ++) {
    ra_V * vp = env[omp_get_thread_num()];
    ra_X * xp = &(ra_X) {};
    size_t i1 = t1 & 3;
    size_t j1 = t1 >> 2;
    float xmin1 = xmin0 + (side / 4.0f) * (float) i1;
    float ymax1 = ymax0 - (side / 4.0f) * (float) j1;
    uint8_t * p1 = p0 + (RESOLUTION / 4) * i1 + (RESOLUTION / 4) * RESOLUTION * j1;

    // TODO: specialize here

    for (size_t j2 = 0; j2 < 4; j2 ++)
    for (size_t i2 = 0; i2 < 4; i2 ++) {
      float xmin2 = xmin1 + (side / 16.0f) * (float) i2;
      float ymax2 = ymax1 - (side / 16.0f) * (float) j2;
      uint8_t * p2 = p1 + (RESOLUTION / 16) * i2 + (RESOLUTION / 16) * RESOLUTION * j2;
      for (size_t j3 = 0; j3 < 4; j3 ++)
      for (size_t i3 = 0; i3 < 4; i3 ++) {
        float xmin3 = xmin2 + (side / 64.0f) * (float) i3;
        float ymax3 = ymax2 - (side / 64.0f) * (float) j3;
        uint8_t * p3 = p2 + (RESOLUTION / 64) * i3 + (RESOLUTION / 64) * RESOLUTION * j3;
        for (size_t k = 0; k < 16; k ++) {
          xp->x[k] = (xmin3 + step / 2.0f) + step * (float) k;
        }
        for (size_t t4 = 0; t4 < 4; t4 ++) {
          float ymax4 = ymax3 - (side / 256.0f) * (float) t4;
          uint8_t * p4 = p3 + (RESOLUTION / 256) * RESOLUTION * t4;
          for (size_t k = 0; k < 4; k ++) {
            xp->y[k] = (ymax4 - step / 2.0f) - step * (float) k;
          }
          size_t result = rasterize(code, xp, vp);
          for (size_t k = 0; k < 4; k ++) {
            uint8_t * p5 = p4 + RESOLUTION * k;
            memcpy(p5, &vp[result].u8x64[16 * k], 16);
          }
        }
      }
    }
  }
}
