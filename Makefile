.PHONY: default clean

default: go

#  -fsanitize=undefined \

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
	-fno-slp-vectorize \
	-pedantic \
	-fopenmp=libomp

CC = \
	/opt/homebrew/Cellar/llvm/19.1.7_1/bin/clang

go: main.c render.h render.o
	$(CC) -o $@ $< render.o $(CFLAGS)

render.o: render.c render.h prospero.c
	$(CC) -c -o $@ $< $(CFLAGS)

prospero.c: compile.py prospero.vm
	./compile.py < prospero.vm > prospero.c

clean:
	rm -f go
	rm -f prospero.c
	rm -f render.o
