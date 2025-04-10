#define PROGRAM_MAX_LEN 8192
#define RESOLUTION 1024

typedef enum : uint8_t {
  OP_AFFINE,
  OP_HYPOT2,
  OP_LE_IMM,
  OP_GE_IMM,
  OP_AND,
  OP_OR,
  OP_RESULT,
} Op;

typedef struct {
  Op op;
  union {
    struct { float a; float b; float c; } affine;
    struct { uint16_t x; uint16_t y; } hypot2;
    struct { uint16_t x; float t; } le_imm;
    struct { uint16_t x; float t; } ge_imm;
    struct { uint16_t x; uint16_t y; } and;
    struct { uint16_t x; uint16_t y; } or;
    struct { uint16_t x; } result;
  };
} Inst;

typedef struct {
  float x[16];
  float y[4];
} ra_X;

typedef union {
  float f32x64[64];
  uint8_t u8x64[64];
} ra_R;

void render(
    size_t num_threads,
    ra_R regs[num_threads][PROGRAM_MAX_LEN],
    Inst *,
    uint8_t image[RESOLUTION][RESOLUTION]
  );
