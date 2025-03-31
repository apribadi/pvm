#include <stddef.h>
#include <stdint.h>

#include "render.h"
#include "prospero.c"

static uint8_t IMAGE[RESOLUTION][RESOLUTION];

static Env ENV;

int main(int argc, char ** argv) {
  (void) argc;
  (void) argv;

  for (size_t i = 0; i < 16; ++ i) {
    ENV.x[i] = 0;
    ENV.y[i] = 0;
  }

  render(&ENV, PROSPERO, IMAGE);

  return 0;
}
