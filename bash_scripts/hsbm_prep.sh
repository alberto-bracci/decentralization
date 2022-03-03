#!/bin/bash
#$ -cwd  
#$ -t 1
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=50G # covid required ~ 8 GB
# #$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

python hsbm.py --dataset_path "data/2022-01-01/decentralization/"  --do_analysis 0  -i ${SGE_TASK_ID} -prep 1

# Keywords: decentralization, covid, social_media, internet, wireless
