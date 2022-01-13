#!/bin/bash
#$ -cwd  
#$ -t 1
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=600G
#$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

# python hsbm.py --dataset_path "data/2021-09-01/decentralization/"  --do_analysis 0  -i ${SGE_TASK_ID}
# python hsbm.py --dataset_path "data/2022-01-01/decentralization/"  --do_analysis 1 -id_list "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" # "[1,2,3,4,5,6,7,8,9,10]" 
python hsbm.py --dataset_path "data/2022-01-01/decentralization/"  --do_analysis 1 -id_list "[1,2,3,4,6,8,10,11,13,14,16,17,19,20,22,26,29,30,32,33,35,38,41,42,45,47,49,50,54,56,59,60,62,65,68,70,71,76,77,81,82,83,87,89,92,95,96,97]" --analysis_results_subfolder "consensus_all_5_levels_100_iter"

# qsub -hold_jid 500 job_two.sh ### where 500 is the id of the previous job to finish


# qlogin
# source source
# conda activate gt
# Latora
# cd decentralization/github/scripts
# rm -r /data/Maths-LatoraLab/GDB/decentralization/github/data/2021-09-01/sample_test/0_min_inCitations_5_min_word_occurrences_titles
# python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 1
# python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 2
# python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 3
# python hsbm.py --dataset_path "data/2021-09-01/sample_test/" -NoIterMC 10 -analysis 1 -id_list "[1,2,3]"

