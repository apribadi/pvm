#pragma once

#include <arm_neon.h>

typedef float32x4x4_t v512;

static inline v512 vx_fadd_32(v512 x, v512 y) {
  return (v512) {{
    vaddq_f32(x.val[0], y.val[0]),
    vaddq_f32(x.val[1], y.val[1]),
    vaddq_f32(x.val[2], y.val[2]),
    vaddq_f32(x.val[3], y.val[3])
  }};
}

static inline v512 vx_fmul_32(v512 x, v512 y) {
  return (v512) {{
    vmulq_f32(x.val[0], y.val[0]),
    vmulq_f32(x.val[1], y.val[1]),
    vmulq_f32(x.val[2], y.val[2]),
    vmulq_f32(x.val[3], y.val[3])
  }};
}

static inline v512 vx_fmul_n_32(v512 x, float y) {
  return (v512) {{
    vmulq_n_f32(x.val[0], y),
    vmulq_n_f32(x.val[1], y),
    vmulq_n_f32(x.val[2], y),
    vmulq_n_f32(x.val[3], y)
  }};
}
