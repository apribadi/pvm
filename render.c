#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>
#include <stdio.h>

#include "simd.h"
#include "render.h"

// -------- SPECIALIIZE --------

typedef struct {
  float x[5];
  float y[5];
} sp_X;

typedef union {
  struct {
    float min[16];
    float max[16];
  } interval;
} sp_R;

struct sp_Tbl { size_t (*ops[7])(Inst *, sp_X *, sp_R *, struct sp_Tbl *, size_t, Inst); };

static inline size_t sp_dispatch(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip) {
  Inst inst = cp[ip];
  return tp->ops[inst.op](cp, xp, rp, tp, ip, inst);
}

static size_t sp_affine(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip, Inst inst) {
  float32x4_t xmin = vld1q_f32(&xp->x[0]);
  float32x4_t xmax = vld1q_f32(&xp->x[1]);
  float32x4_t ymin = vld1q_f32(&xp->y[1]);
  float32x4_t ymax = vld1q_f32(&xp->y[0]);
  float32x4_t u0 = vmulq_n_f32(xmin, inst.affine.a);
  float32x4_t u1 = vmulq_n_f32(xmax, inst.affine.a);
  float32x4_t v0 = vmulq_n_f32(ymin, inst.affine.b);
  float32x4_t v1 = vmulq_n_f32(ymax, inst.affine.b);
  float32x4_t umin = vminnmq_f32(u0, u1);
  float32x4_t umax = vmaxnmq_f32(u0, u1);
  float32x4_t vmin = vaddq_f32(vminnmq_f32(v0, v1), vdupq_n_f32(inst.affine.c));
  float32x4_t vmax = vaddq_f32(vmaxnmq_f32(v0, v1), vdupq_n_f32(inst.affine.c));
  float32x4x4_t wmin;
  float32x4x4_t wmax;
  for (size_t k = 0; k < 4; k ++) {
    wmin.val[k] = vaddq_f32(umin, vdupq_n_f32(vmin[k]));
    wmax.val[k] = vaddq_f32(umax, vdupq_n_f32(vmax[k]));
  }
  vx_store_f32(rp[ip].interval.min, wmin);
  vx_store_f32(rp[ip].interval.max, wmax);
  /*
  printf("ip = %d\n", (int) ip);
  for (size_t k = 0; k < 4; k ++) {
    printf("min = %f\n", (double) wmin.val[0][k]);
    printf("max = %f\n", (double) wmax.val[0][k]);
  }
  */
  return sp_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t sp_hypot2(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip, Inst inst) {
  return sp_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t sp_le_imm(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip, Inst inst) {
  return sp_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t sp_ge_imm(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip, Inst inst) {
  return sp_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t sp_and(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip, Inst inst) {
  return sp_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t sp_or(Inst * cp, sp_X * xp, sp_R * rp, struct sp_Tbl * tp, size_t ip, Inst inst) {
  return sp_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t sp_result(Inst *, sp_X *, sp_R *, struct sp_Tbl *, size_t, Inst inst) {
  return inst.result.x;
}

static struct sp_Tbl sp_TBL = {{
  sp_affine,
  sp_hypot2,
  sp_le_imm,
  sp_ge_imm,
  sp_and,
  sp_or,
  sp_result
}};

static size_t specialize(Inst * cp, sp_X * xp, sp_R * rp) {
  return sp_dispatch(cp, xp, rp, &sp_TBL, 0);
}

// -------- RASTERIZE --------

typedef struct {
  float x[16];
  float y[4];
} ra_X;

typedef union {
  float f32x64[64];
  uint8_t u8x64[64];
} ra_R;

struct ra_Tbl { size_t (*ops[7])(Inst *, ra_X *, ra_R *, struct ra_Tbl *, size_t, Inst); };

static inline size_t ra_dispatch(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip) {
  Inst inst = cp[ip];
  return tp->ops[inst.op](cp, xp, rp, tp, ip, inst);
}

static size_t ra_affine(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 x = vx_load_f32(xp->x);
  v128 y = vq_load_f32(xp->y);
  v512 u = vx_add_n_f32(vx_mul_n_f32(x, inst.affine.a), inst.affine.c);
  v128 v = vq_mul_n_f32(y, inst.affine.b);
  for (size_t k = 0; k < 4; k ++) {
    vx_store_f32(&rp[ip].f32x64[16 * k],  vx_add_n_f32(u, vq_get_f32(v, k)));
  }
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_hypot2(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  for (size_t k = 0; k < 4; k ++) {
    v512 x = vx_load_f32(&rp[inst.hypot2.x].f32x64[16 * k]);
    v512 y = vx_load_f32(&rp[inst.hypot2.y].f32x64[16 * k]);
    vx_store_f32(&rp[ip].f32x64[16 * k], vx_add_f32(vx_mul_f32(x, x), vx_mul_f32(y, y)));
  }
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_le_imm(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 t = vx_dup_f32(inst.le_imm.t);
  v512 r;
  for (size_t k = 0; k < 4; k ++) {
    vx_set_v128(&r, k, vx_truncate_i8_i32(vx_le_f32(vx_load_f32(&rp[inst.le_imm.x].f32x64[16 * k]), t)));
  }
  vx_store_u8(rp[ip].u8x64, r);
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_ge_imm(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 r;
  v512 t = vx_dup_f32(inst.ge_imm.t);
  for (size_t k = 0; k < 4; k ++) {
    vx_set_v128(&r, k, vx_truncate_i8_i32(vx_le_f32(t, vx_load_f32(&rp[inst.ge_imm.x].f32x64[16 * k]))));
  }
  vx_store_u8(rp[ip].u8x64, r);
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_and(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 x = vx_load_u8(rp[inst.and.x].u8x64);
  v512 y = vx_load_u8(rp[inst.and.y].u8x64);
  vx_store_u8(rp[ip].u8x64, vx_and(x, y));
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_or(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 x = vx_load_u8(rp[inst.or.x].u8x64);
  v512 y = vx_load_u8(rp[inst.or.y].u8x64);
  vx_store_u8(rp[ip].u8x64, vx_or(x, y));
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_result(Inst *, ra_X *, ra_R *, struct ra_Tbl *, size_t, Inst inst) {
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

static size_t rasterize(Inst * cp, ra_X * xp, ra_R * rp) {
  return ra_dispatch(cp, xp, rp, &ra_TBL, 0);
}

// -------- RENDER FUNCTION --------

void render(size_t num_insts, Inst code[num_insts], uint8_t image[RES][RES]) {
  float side = 2.0f;
  float step = side / RES;
  float xmin = -1.0f;
  float ymax = 1.0f;
  uint8_t * p0 = &image[0][0];

#pragma omp parallel for
  for (size_t t1 = 0; t1 < 16; t1 ++) {
    size_t i1 = t1 & 3;
    size_t j1 = t1 >> 2;
    float xmin1 = xmin + side / 4.0f * (float) i1;
    float ymax1 = ymax - side / 4.0f * (float) j1;
    uint8_t * p1 = p0 + RES / 4 * i1 + RES / 4 * RES * j1;

    // TODO: specialize here
    {
      sp_X * xp = &(sp_X) {};
      sp_R * rp = malloc(sizeof(sp_R) * num_insts);
      if (! rp) abort();

      for (size_t k = 0; k < 5; k ++) {
        xp->x[k] = xmin1 + side / 16.0f * (float) k;
        xp->y[k] = ymax1 - side / 16.0f * (float) k;
      }

      specialize(code, xp, rp);
    }

    {
      ra_X * xp = &(ra_X) {};
      ra_R * rp = malloc(sizeof(ra_R) * num_insts);
      if (! rp) abort();

      for (size_t j2 = 0; j2 < 4; j2 ++)
      for (size_t i2 = 0; i2 < 4; i2 ++) {
        float xmin2 = xmin1 + side / 16.0f * (float) i2;
        float ymax2 = ymax1 - side / 16.0f * (float) j2;
        uint8_t * p2 = p1 + RES / 16 * i2 + RES / 16 * RES * j2;
        for (size_t j3 = 0; j3 < 4; j3 ++)
        for (size_t i3 = 0; i3 < 4; i3 ++) {
          float xmin3 = xmin2 + side / 64.0f * (float) i3;
          float ymax3 = ymax2 - side / 64.0f * (float) j3;
          uint8_t * p3 = p2 + RES / 64 * i3 + RES / 64 * RES * j3;
          for (size_t k = 0; k < 16; k ++) {
            xp->x[k] = xmin3 + step / 2.0f + step * (float) k;
          }
          for (size_t t4 = 0; t4 < 4; t4 ++) {
            float ymax4 = ymax3 - side / 256.0f * (float) t4;
            uint8_t * p4 = p3 + RES / 256 * RES * t4;
            for (size_t k = 0; k < 4; k ++) {
              xp->y[k] = ymax4 - step / 2.0f - step * (float) k;
            }
            size_t result = rasterize(code, xp, rp);
            for (size_t k = 0; k < 4; k ++) {
              uint8_t * p5 = p4 + RES * k;
              memcpy(p5, &rp[result].u8x64[16 * k], 16);
            }
          }
        }
      }
      free(rp);
    }
  }
}
