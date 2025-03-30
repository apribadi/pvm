.PHONY: default clean

default: go


CFLAGS = \
  -fsanitize=undefined \
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

go: main.c eval.h eval.o
	clang -o $@ $< eval.o $(CFLAGS)

prospero.c: compile.py prospero.vm
	./compile.py < prospero.vm > prospero.c

eval.o: eval.c eval.h prospero.c
	clang -c -o $@ $< $(CFLAGS)

clean:
	rm -f go
	rm -f prospero.c
	rm -f eval.o
