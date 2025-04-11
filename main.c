#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "render.h"
#include "prospero.c"

#define ITERATION_COUNT 5
#define NUM_THREADS 4

static uint8_t IMAGE[RESOLUTION][RESOLUTION];

static ra_R REGS[NUM_THREADS][PROGRAM_MAX_LEN];

int main(int, char **) {
  uint64_t start = clock_gettime_nsec_np(CLOCK_REALTIME); // which clock ???

  for (size_t i = 0; i < ITERATION_COUNT; i ++) {
    render(NUM_THREADS, REGS, PROSPERO, IMAGE);
  }

  uint64_t stop = clock_gettime_nsec_np(CLOCK_REALTIME);
  double ms_per_frame = (double) (stop - start) / 1000000.0 / ITERATION_COUNT;

  printf("rendered in %f ms / frame ...\n", ms_per_frame);

  FILE * file = fopen("prospero.pgm", "w");
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
