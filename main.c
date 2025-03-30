#include <stddef.h>
#include <stdint.h>

#include "eval.h"
#include "prospero.c"

#define RES 1024

static uint8_t IMAGE[RES][RES];

static Env ENV;

int main(int argc, char ** argv) {
  (void) argc;
  (void) argv;

  for (size_t i = 0; i < 16; ++ i) {
    ENV.x[i] = 0;
    ENV.y[i] = 0;
  }

  eval(&ENV, PROSPERO, &IMAGE[0][0]);

  return 0;
}
