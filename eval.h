typedef enum {
  AFFINE,
  HYPOT2,
  LE,
  GE,
  AND,
  OR,
  RET,
} Tag;

typedef struct {
  Tag tag;
  union {
    struct { float a; float b; float c; } affine;
    struct { uint16_t x; uint16_t y; } hypot2;
    struct { uint16_t x; float t; } le;
    struct { uint16_t x; float t; } ge;
    struct { uint16_t x; uint16_t y; } and;
    struct { uint16_t x; uint16_t y; } or;
    struct { uint16_t x; } ret;
  };
} Inst;

#define MAX_INST 8192

struct Env {
  float v[16][MAX_INST];
  float x[16];
  float y[16];
  float z[16];
};

void eval(struct Env *, Inst *);
