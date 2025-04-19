#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

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
  float32x4_t xmin = v128_load_f32(&xp->x[0]);
  float32x4_t xmax = v128_load_f32(&xp->x[1]);
  float32x4_t ymin = v128_load_f32(&xp->y[1]);
  float32x4_t ymax = v128_load_f32(&xp->y[0]);
  float32x4_t u0 = v128_mul_n_f32(xmin, inst.affine.a);
  float32x4_t u1 = v128_mul_n_f32(xmax, inst.affine.a);
  float32x4_t v0 = v128_mul_n_f32(ymin, inst.affine.b);
  float32x4_t v1 = v128_mul_n_f32(ymax, inst.affine.b);
  float32x4_t umin = v128_min_f32(u0, u1);
  float32x4_t umax = v128_max_f32(u0, u1);
  float32x4_t vmin = v128_add_n_f32(v128_min_f32(v0, v1), inst.affine.c);
  float32x4_t vmax = v128_add_n_f32(v128_max_f32(v0, v1), inst.affine.c);
  float32x4x4_t wmin;
  float32x4x4_t wmax;
  for (size_t k = 0; k < 4; k ++) {
    wmin.val[k] = v128_add_n_f32(umin, v128_get_f32(vmin, k));
    wmax.val[k] = v128_add_n_f32(umax, v128_get_f32(vmax, k));
  }
  v512_store_f32(rp[ip].interval.min, wmin);
  v512_store_f32(rp[ip].interval.max, wmax);
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
  v512 x = v512_load_f32(xp->x);
  v128 y = v128_load_f32(xp->y);
  v512 u = v512_add_n_f32(v512_mul_n_f32(x, inst.affine.a), inst.affine.c);
  v128 v = v128_mul_n_f32(y, inst.affine.b);
  for (size_t k = 0; k < 4; k ++) {
    v512_store_f32(&rp[ip].f32x64[16 * k],  v512_add_n_f32(u, v128_get_f32(v, k)));
  }
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_hypot2(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  for (size_t k = 0; k < 4; k ++) {
    v512 x = v512_load_f32(&rp[inst.hypot2.x].f32x64[16 * k]);
    v512 y = v512_load_f32(&rp[inst.hypot2.y].f32x64[16 * k]);
    v512_store_f32(&rp[ip].f32x64[16 * k], v512_add_f32(v512_mul_f32(x, x), v512_mul_f32(y, y)));
  }
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_le_imm(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 t = v512_dup_f32(inst.le_imm.t);
  v128 r[4];
  for (size_t k = 0; k < 4; k ++) {
    r[k] = v512_narrow_i8_i32(v512_le_f32(v512_load_f32(&rp[inst.le_imm.x].f32x64[16 * k]), t));
  }
  v512_store_u8(rp[ip].u8x64, v512_from_v128x4(r));
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_ge_imm(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 t = v512_dup_f32(inst.ge_imm.t);
  v128 r[4];
  for (size_t k = 0; k < 4; k ++) {
    r[k] = v512_narrow_i8_i32(v512_le_f32(t, v512_load_f32(&rp[inst.ge_imm.x].f32x64[16 * k])));
  }
  v512_store_u8(rp[ip].u8x64, v512_from_v128x4(r));
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_and(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 x = v512_load_u8(rp[inst.and.x].u8x64);
  v512 y = v512_load_u8(rp[inst.and.y].u8x64);
  v512_store_u8(rp[ip].u8x64, v512_and(x, y));
  return ra_dispatch(cp, xp, rp, tp, ip + 1);
}

static size_t ra_or(Inst * cp, ra_X * xp, ra_R * rp, struct ra_Tbl * tp, size_t ip, Inst inst) {
  v512 x = v512_load_u8(rp[inst.or.x].u8x64);
  v512 y = v512_load_u8(rp[inst.or.y].u8x64);
  v512_store_u8(rp[ip].u8x64, v512_or(x, y));
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

static void rasterize(Inst * cp, ra_X * xp, ra_R * rp, uint8_t * tile, size_t stride) {
  size_t result = ra_dispatch(cp, xp, rp, &ra_TBL, 0);
  for (size_t k = 0; k < 4; k ++) {
    memcpy(tile + stride * k, &rp[result].u8x64[16 * k], 16);
  }
}

// -------- RENDER --------

void render_tile(
    size_t num_insts,
    Inst code[num_insts],
    float xmin,
    float ymax,
    float side,
    size_t resolution,
    size_t stride,
    uint8_t * tile
  )
{
  if (resolution == 64) {
    ra_X * xp = &(ra_X) {};
    ra_R * rp = malloc(sizeof(ra_R) * num_insts);
    if (! rp) abort();

    float step = 0.015625f * side;

    for (size_t i = 0; i < 64; i += 4) {
      for (size_t k = 0; k < 4; k ++) {
        xp->y[k] = ymax - 0.5f * step - step * (float) (i + k);
      }
      for (size_t j = 0; j < 64; j += 16) {
        for (size_t k = 0; k < 16; k ++) {
          xp->x[k] = xmin + 0.5f * step + step * (float) (j + k);
        }
        rasterize(code, xp, rp, tile + i * stride + j, stride);
      }
    }

    free(rp);

    return;
  }

  for (size_t i = 0; i < 4; i ++) {
    for (size_t j = 0; j < 4; j ++) {
      render_tile(
          num_insts,
          code,
          xmin + 0.25f * side * (float) j,
          ymax - 0.25f * side * (float) i,
          0.25f * side,
          resolution / 4,
          stride,
          tile + resolution / 4 * stride * i + resolution / 4 * j
        );
    }
  }
}

void render(
    size_t num_insts,
    Inst code[num_insts],
    size_t resolution,
    uint8_t image[resolution][resolution]
  )
{
  // resolution = 256, 1024, 4096, ...
  assert(resolution >= 256);
  assert(__builtin_popcountll(resolution) == 1);
  assert(__builtin_ctzll(resolution) % 2 == 0);

  float side = 2.0f;
  float xmin = -1.0f;
  float ymax = 1.0f;

#pragma omp parallel for
  for (size_t t = 0; t < 16; t ++) {
    size_t i = t / 4;
    size_t j = t % 4;
    render_tile(
        num_insts,
        code,
        xmin + 0.25f * side * (float) j,
        ymax - 0.25f * side * (float) i,
        0.25f * side,
        resolution / 4,
        resolution,
        &image[resolution / 4 * i][resolution / 4 * j]
      );
  }
}
