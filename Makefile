.PHONY: default

default: go

CFLAGS = \
	-std=c2x \
	-O2 \
	-march=native \
	-Wall \
	-Wconversion \
	-Wdouble-promotion \
	-Wextra \
	-ffp-contract=off \
	-fno-math-errno \
	-fno-omit-frame-pointer \
	-fno-slp-vectorize \
	-pedantic

prospero.c: compile.py prospero.vm
	./compile.py < prospero.vm > prospero.c

go: main.c eval.h eval.o
	clang -o $@ $< eval.o $(CFLAGS)

eval.o: eval.c eval.h prospero.c
	clang -c -o $@ $< $(CFLAGS)
