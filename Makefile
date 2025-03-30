.PHONY: clean default

default: out/go

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

clean:
	rm -f out/*

out/go: main.c eval.h out/eval.o
	clang -o $@ $< out/eval.o $(CFLAGS)

out/eval.o: eval.c eval.h
	clang -c -o $@ $< $(CFLAGS)
