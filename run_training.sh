#!/bin/sh

# DATASET_PATH=/mnt/public/home/220462/datasets/DATASET_Acdc

# export PYTHONPATH=/mnt/public/home/220462/3d_seg_2paper/2UNETR++/
# export RESULTS_FOLDER=/mnt/public/home/220462/3d_seg_2paper/train_logs
# export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
# export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw
# export CUDA_VISIBLE_DEVICES=0,1

CUDA_VISIBLE_DEVICES=0 /mnt/public/home/10431220/miniconda3/envs/pytorch12/bin/python /mnt/public/home/220462/LKDA_net/main_train.py

# csub -cwd log -n 2 -R "rusage[gpu=1] span[hosts=1]" -q "train" -o %J.txt -sp 10 /mnt/public/home/220462/miniconda3/envs/pytorch1.8/bin/python /mnt/public/home/220462/LKDA_net/main_train.py

csub -cwd log -n 4,4 -R "rusage[gpu=0.5]" -q "train" -o %J.txt -sp 1 /mnt/public/home/220462/miniconda3/envs/pytorch1.8/bin/python /mnt/public/home/220462/LKDA_net/main_train.py