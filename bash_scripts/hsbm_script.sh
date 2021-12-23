#!/bin/bash
#$ -cwd  
#$ -t 1#-10
#$ -j y    
#$ -pe smp 1
#$ -l h_vmem=150G
#$ -l highmem
#$ -l h_rt=240:0:0  
# #$ -m bae

module load anaconda3
export OMP_NUM_THREADS=1
conda activate gt
cd ../scripts

# python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  --do_analysis 0  -i ${SGE_TASK_ID}
# -analysis 1 -id_list "[1,2,3,4,5,6,7,8,9,10]"

rm -r /data/Maths-LatoraLab/GDB/decentralization/github/data/2021-09-01/sample_test/0cits_5_occ_titles
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 1
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 2
python hsbm.py --dataset_path "data/2021-09-01/sample_test/"  -NoIterMC 10 --do_analysis 0  -i 3
python hsbm.py --dataset_path "data/2021-09-01/sample_test/" -NoIterMC 10 -analysis 1 -id_list "[1,2,3]"
