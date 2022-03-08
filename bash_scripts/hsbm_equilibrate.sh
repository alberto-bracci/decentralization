#!/bin/bash
#$ -cwd  
#$ -t 1-100
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=100G # covid required ~ 42.5GB
# #$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

python hsbm.py --dataset_path "data/2022-01-01/decentralization/"  --do_analysis 0  -i ${SGE_TASK_ID} --stop_at_fit 0

# Keywords: decentralization, covid, social_media, internet, wireless

# If you want to submit a job to wait for the previous one to finish you can do so as follows:
# qsub -hold_jid 500 job_two.sh ### where 500 is the id of the previous job to finish