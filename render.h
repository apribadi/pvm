#define NUM_THREADS 6
#define MAX_PROGRAM_LENGTH 8192
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
    struct { uint16_t x; uint16_t y; } hypot2;
    struct { uint16_t x; float t; } le_imm;
    struct { uint16_t x; float t; } ge_imm;
    struct { uint16_t x; uint16_t y; } and;
    struct { uint16_t x; uint16_t y; } or;
    struct { uint16_t x; } result;
  };
} Ins;

typedef struct {
  float v[MAX_PROGRAM_LENGTH][16];
  float x[16];
  float y[16];
} __attribute__((aligned(64))) Env;

void render(Env env[NUM_THREADS], Ins *, uint8_t image[RESOLUTION][RESOLUTION]);
