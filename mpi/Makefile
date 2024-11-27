all: main

main: main.cpp ../config.hpp
	mpicxx main.cpp -o main

run: main
	for np in 1 8 16 32 64; do \
		for i in {1..20}; do \
			echo "Running with $$np processes, iteration $$i"; \
			mpiexec --bind-to core -np $$np ./main; \
		done \
	done
