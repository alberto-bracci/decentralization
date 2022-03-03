#!/bin/bash
#$ -cwd  
#$ -t 1
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=300G
#$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

python hsbm.py --dataset_path "data/2022-01-01/decentralization/"  --do_analysis 1 -ID_list "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]" --analysis_results_subfolder "consensus_all_100_iter" -lev_kf 0

# Keywords: decentralization, covid, social_media, internet, wireless

# If you want to submit a job to wait for the previous one to finish you can do so as follows:
# qsub -hold_jid 500 job_two.sh ### where 500 is the id of the previous job to finish