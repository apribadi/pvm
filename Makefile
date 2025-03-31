.PHONY: default clean

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

go: main.c render.h render.o
	clang -o $@ $< render.o $(CFLAGS)

prospero.c: compile.py prospero.vm
	./compile.py < prospero.vm > prospero.c

render.o: render.c render.h prospero.c
	clang -c -o $@ $< $(CFLAGS)

clean:
	rm -f go
	rm -f prospero.c
	rm -f render.o
