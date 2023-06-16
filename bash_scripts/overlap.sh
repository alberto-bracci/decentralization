#!/bin/bash
#$ -cwd  
#$ -t 63-63
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=15G # covid required ~ 8 GB
# #$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../../scripts

python overlap.py -i ${SGE_TASK_ID}