#pragma once

#include <arm_neon.h>

typedef float32x4x4_t v512;

typedef float32x4x2_t v256;

typedef float32x4_t v128;

static inline v128 v128_load_f32(float p[4]) {
  return vld1q_f32(p);
}

static inline void v128_store_f32(float p[4], v128 x) {
  return vst1q_f32(p, x);
}

static inline float v128_get_f32(v128 x, size_t i) {
  return x[i];
}

static inline v128 v128_add_n_f32(v128 x, float y) {
  return vaddq_f32(x, vdupq_n_f32(y));
}

static inline v128 v128_mul_n_f32(v128 x, float y) {
  return vmulq_n_f32(x, y);
}

static inline v128 v128_min_f32(v128 x, v128 y) {
  return vminq_f32(x, y);
}

static inline v128 v128_max_f32(v128 x, v128 y) {
  return vmaxq_f32(x, y);
}

static inline v128 v256_narrow_i8_i16(v256 x) {
  return vreinterpretq_f32_u8(vuzp1q_u8(vreinterpretq_u8_f32(x.val[0]), vreinterpretq_u8_f32(x.val[1])));
}

static inline v512 v512_load_f32(float p[16]) {
  return vld1q_f32_x4(p);
}

static inline void v512_store_f32(float p[16], v512 x) {
  return vst1q_f32_x4(p, x);
}

static inline v512 v512_load_u8(uint8_t p[64]) {
  uint8x16x4_t x = vld1q_u8_x4(p);
  return (float32x4x4_t) {{
    vreinterpretq_f32_u8(x.val[0]),
    vreinterpretq_f32_u8(x.val[1]),
    vreinterpretq_f32_u8(x.val[2]),
    vreinterpretq_f32_u8(x.val[3])
  }};
}

static inline void v512_store_u8(uint8_t p[64], v512 x) {
  uint8x16x4_t y = {{
    vreinterpretq_u8_f32(x.val[0]),
    vreinterpretq_u8_f32(x.val[1]),
    vreinterpretq_u8_f32(x.val[2]),
    vreinterpretq_u8_f32(x.val[3])
  }};
  vst1q_u8_x4(p, y);
}

static inline v512 v512_from_v128x4(v128 x[4]) {
  return (float32x4x4_t) {{ x[0], x[1], x[2], x[3] }};
}

static inline v512 v512_dup_f32(float x) {
  return (float32x4x4_t) {{
    vdupq_n_f32(x),
    vdupq_n_f32(x),
    vdupq_n_f32(x),
    vdupq_n_f32(x)
  }};
}

static inline v256 v512_narrow_i16_i32(v512 x) {
  return (float32x4x2_t) {{
    vreinterpretq_f32_u16(vuzp1q_u16(vreinterpretq_u16_f32(x.val[0]), vreinterpretq_u16_f32(x.val[1]))),
    vreinterpretq_f32_u16(vuzp1q_u16(vreinterpretq_u16_f32(x.val[2]), vreinterpretq_u16_f32(x.val[3])))
  }};
}

static inline v128 v512_narrow_i8_i32(v512 x) {
  return v256_narrow_i8_i16(v512_narrow_i16_i32(x));
}

static inline v512 v512_and(v512 x, v512 y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0]))),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1]))),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2]))),
    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])))
  }};
}

static inline v512 v512_or(v512 x, v512 y) {
  return (float32x4x4_t) {{
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[0]), vreinterpretq_u32_f32(y.val[0]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[1]), vreinterpretq_u32_f32(y.val[1]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[2]), vreinterpretq_u32_f32(y.val[2]))),
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x.val[3]), vreinterpretq_u32_f32(y.val[3])))
  }};
}

static inline v512 v512_add_f32(v512 x, v512 y) {
  return (float32x4x4_t) {{
    vaddq_f32(x.val[0], y.val[0]),
    vaddq_f32(x.val[1], y.val[1]),
    vaddq_f32(x.val[2], y.val[2]),
    vaddq_f32(x.val[3], y.val[3])
  }};
}

static inline v512 v512_add_n_f32(v512 x, float y) {
  return (float32x4x4_t) {{
    vaddq_f32(x.val[0], vdupq_n_f32(y)),
    vaddq_f32(x.val[1], vdupq_n_f32(y)),
    vaddq_f32(x.val[2], vdupq_n_f32(y)),
    vaddq_f32(x.val[3], vdupq_n_f32(y))
  }};
}

static inline v512 v512_mul_f32(v512 x, v512 y) {
  return (float32x4x4_t) {{
    vmulq_f32(x.val[0], y.val[0]),
    vmulq_f32(x.val[1], y.val[1]),
    vmulq_f32(x.val[2], y.val[2]),
    vmulq_f32(x.val[3], y.val[3])
  }};
}

static inline v512 v512_mul_n_f32(v512 x, float y) {
  return (float32x4x4_t) {{
    vmulq_n_f32(x.val[0], y),
    vmulq_n_f32(x.val[1], y),
    vmulq_n_f32(x.val[2], y),
    vmulq_n_f32(x.val[3], y)
  }};
}

static inline v512 v512_le_f32(v512 x, v512 y) {
  return (float32x4x4_t) {{
    vcleq_f32(x.val[0], y.val[0]),
    vcleq_f32(x.val[1], y.val[1]),
    vcleq_f32(x.val[2], y.val[2]),
    vcleq_f32(x.val[3], y.val[3])
  }};
}
