#pragma once

#include <arm_neon.h>

typedef float32x4x4_t v512;

static inline v512 vx_load_f32(float p[16]) {
  return vld1q_f32_x4(p);
}

static inline void vx_store_f32(float p[16], v512 x) {
  return vst1q_f32_x4(p, x);
}

static inline v512 vx_load_u8(uint8_t p[64]) {
  return vld1q_f32_x4((float *) p);
}

static inline void vx_store_u8(uint8_t p[64], v512 x) {
  return vst1q_f32_x4((float *) p, x);
}

static inline v512 vx_dup_f32(float x) {
  return (v512) {{
    vdupq_n_f32(x),
    vdupq_n_f32(x),
    vdupq_n_f32(x),
    vdupq_n_f32(x)
  }};
}

static inline v512 vx_and(v512 x, v512 y) {
  return (v512) {{
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0]))),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1]))),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2]))),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])))
  }};
}

static inline v512 vx_or(v512 x, v512 y) {
  return (v512) {{
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])))
  }};
}

static inline v512 vx_add_f32(v512 x, v512 y) {
  return (v512) {{
    vaddq_f32(x.val[0], y.val[0]),
    vaddq_f32(x.val[1], y.val[1]),
    vaddq_f32(x.val[2], y.val[2]),
    vaddq_f32(x.val[3], y.val[3])
  }};
}

static inline v512 vx_add_n_f32(v512 x, float y) {
  return (v512) {{
    vaddq_f32(x.val[0], vdupq_n_f32(y)),
    vaddq_f32(x.val[1], vdupq_n_f32(y)),
    vaddq_f32(x.val[2], vdupq_n_f32(y)),
    vaddq_f32(x.val[3], vdupq_n_f32(y))
  }};
}

static inline v512 vx_mul_f32(v512 x, v512 y) {
  return (v512) {{
    vmulq_f32(x.val[0], y.val[0]),
    vmulq_f32(x.val[1], y.val[1]),
    vmulq_f32(x.val[2], y.val[2]),
    vmulq_f32(x.val[3], y.val[3])
  }};
}

static inline v512 vx_mul_n_f32(v512 x, float y) {
  return (v512) {{
    vmulq_n_f32(x.val[0], y),
    vmulq_n_f32(x.val[1], y),
    vmulq_n_f32(x.val[2], y),
    vmulq_n_f32(x.val[3], y)
  }};
}

static inline v512 vx_le_f32(v512 x, v512 y) {
  return (v512) {{
    vcleq_f32(x.val[0], y.val[0]),
    vcleq_f32(x.val[1], y.val[1]),
    vcleq_f32(x.val[2], y.val[2]),
    vcleq_f32(x.val[3], y.val[3])
  }};
}
