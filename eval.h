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

#define MAX_INS 8192
#define RES 1024

typedef struct {
  float v[16][MAX_INS];
  float x[16];
  float y[16];
} Env;

void eval(Env *, Ins *, uint8_t[16]);
