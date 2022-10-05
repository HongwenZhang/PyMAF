#!/bin/bash
export PATH="/root/miniconda3/bin:"$PATH
source /root/.bashrc
source activate
python3 train.py --regressor pymaf_net --pretrained_checkpoint logs/pymaf_res50/pymaf_res50_as_lp3_mlp256-128-64-5_Sep21-21-10-01-CWR/checkpoints/model_best.pt --misc TRAIN.BATCH_SIZE 64