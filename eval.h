typedef enum : uint8_t {
  ADD,
  ADD_IMM,
} Tag;

typedef struct {
  Tag tag;
  union {
    struct {
      uint16_t x;
      uint16_t y;
    } add;
    struct {
      uint16_t x;
      float c;
    } add_imm;
  } __attribute__((packed, aligned(2)));
} Inst;

