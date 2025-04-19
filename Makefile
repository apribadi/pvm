.PHONY: default clean

default: go

CC = clang

CFLAGS = \
	-std=c2x \
	-O2 \
	-march=native \
	-Wall \
	-Wconversion \
	-Wdouble-promotion \
	-Wextra \
	-Wno-unused-function \
	-ffp-contract=off \
	-fno-math-errno \
	-fno-omit-frame-pointer \
	-fno-slp-vectorize

#  -fsanitize=undefined \

go: main.c render.h render.o prospero.c
	$(CC) -o $@ $< render.o $(CFLAGS) -lomp -L/opt/homebrew/opt/libomp/lib

render.o: render.c render.h simd.h
	$(CC) -c -o $@ $< $(CFLAGS) -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include

prospero.c: compile.py prospero.vm
	./compile.py < prospero.vm > prospero.c

clean:
	rm -f go
	rm -f prospero.c
	rm -f render.o
