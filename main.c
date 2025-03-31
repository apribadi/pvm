#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "render.h"
#include "prospero.c"

static uint8_t IMAGE[RESOLUTION][RESOLUTION];

static Env ENV[NUM_THREADS];

int main(int argc, char ** argv) {
  (void) argc;
  (void) argv;

  uint64_t start = clock_gettime_nsec_np(CLOCK_REALTIME); // which clock ???
  render(ENV, PROSPERO, IMAGE);
  uint64_t stop = clock_gettime_nsec_np(CLOCK_REALTIME);
  double elapsed_ms = ((double) stop - (double) start) / 1000000.;

  printf("rendered in %f ms ...", elapsed_ms);

  FILE * file = fopen("prospero.pgm", "w+");
  if (! file) return 1;

  // we don't check fprintf's return values ...
  fprintf(file, "P5\n");
  fprintf(file, "%d\n", RESOLUTION);
  fprintf(file, "%d\n", RESOLUTION);
  fprintf(file, "255\n");

  if (fwrite(IMAGE, sizeof(IMAGE), 1, file) != 1) return 1;
  if (fclose(file) != 0) return 1;

  return 0;
}
