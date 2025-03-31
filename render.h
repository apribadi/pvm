#define NUM_THREADS 4
#define PROGRAM_MAX_LEN 8192
#define RESOLUTION 1024

typedef enum {
  AFFINE,
  HYPOT2,
  LE_IMM,
  GE_IMM,
  AND,
  OR,
  RESULT,
} Tag;

typedef struct {
  Tag tag;
  union {
    struct { float a; float b; float c; } affine;
    struct { uint32_t x; uint32_t y; } hypot2;
    struct { uint32_t x; float t; } le_imm;
    struct { uint32_t x; float t; } ge_imm;
    struct { uint32_t x; uint32_t y; } and;
    struct { uint32_t x; uint32_t y; } or;
    struct { uint32_t x; } result;
  };
} Ins;

typedef struct {
  float v[PROGRAM_MAX_LEN][32];
  float x[32];
  float y[32];
} __attribute__((aligned(64))) Env;

void render(Env env[NUM_THREADS], Ins *, uint8_t image[RESOLUTION][RESOLUTION]);
