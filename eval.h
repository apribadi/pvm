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
} Ins;

#define MAX_INS 8192
#define RES 1024

struct Env {
  float v[16][MAX_INS];
  float x[16];
  float y[16];
};

void eval(struct Env *, Ins *, uint8_t[16]);
