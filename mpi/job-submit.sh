#!/bin/bash
#SBATCH --job-name=demo_mpi
#SBATCH --tasks-per-node=32
#SBATCH --nodelist=node[4-7]

module load openmpi

cd $SLURM_SUBMIT_DIR

# Compile the MPI program
mpic++ -o main main.cpp

# Run the MPI program with different numbers of processes
for np in 1 8 16 32 64; do
	for i in {1..20}; do
		echo "Running with $np processes, iteration $i"
		mpiexec --bind-to core -np $np ./main
	done
done
