#define RES 1024

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

void render(size_t num_insts, Inst code[num_insts], uint8_t image[RES][RES]);
