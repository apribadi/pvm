#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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

  FILE * file = fopen("prospero.pgm", "w+");
  if (! file) return 1;

  // we don't check fprintf's return value ...
  fprintf(file, "P5\n");
  fprintf(file, "%d\n", RESOLUTION);
  fprintf(file, "%d\n", RESOLUTION);
  fprintf(file, "255\n");

  if (fwrite(IMAGE, sizeof(IMAGE), 1, file) != 1) return 1;
  if (fclose(file) != 0) return 1;

  return 0;
}
