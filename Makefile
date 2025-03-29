go: main.c eval.o
	clang -o go main.c eval.o -O2 -Wall -Werror -pedantic -march=native

eval.o: eval.c
	clang -o eval.o -c eval.c -O2 -Wall -Werror -pedantic -march=native
