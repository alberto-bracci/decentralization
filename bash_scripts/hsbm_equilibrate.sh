#!/bin/bash
#$ -cwd  
#$ -t 2-100
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=500G # check 2175659
# #$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

python hsbm.py --dataset_path "data/2022-01-01/covid/"  --do_analysis 0  -i ${SGE_TASK_ID} --stop_at_fit 0

# Keywords: decentralization, covid, social_media, internet, wireless

# If you want to submit a job to wait for the previous one to finish you can do so as follows:
# qsub -hold_jid 2178832 job_two.sh ### where 2178832 is the id of the previous job to finish (covid fits 2-100)
# 2178837 social_media fits 1-100
# 2178838 internet fits 1-100
# 2178839 wireless fits 1-100