#!/bin/bash
#$ -cwd  
#$ -t 1
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=1G
# #$ -l highmem
#$ -l h_rt=1:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

rm -r /data/Maths-LatoraLab/GDB/decentralization/github/data/2021-09-01/sample_test/0_min_inCitations_5_min_word_occurrences_titles
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  --do_analysis 0  -i ${SGE_TASK_ID} -prep 1
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 1
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 2
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 3
python hsbm.py --dataset_path "data/2021-09-01/sample_test/" -NoIterMC 10 -analysis 1 -ID_list "[1,2,3]" -lev_kf 0