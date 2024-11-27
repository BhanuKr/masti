#!/bin/bash
#SBATCH --job-name demo_openmp_prefix_sum
#SBATCH --tasks-per-node 32
#SBATCH --nodelist node4

cd $SLURM_SUBMIT_DIR
make
./prefix_sum
