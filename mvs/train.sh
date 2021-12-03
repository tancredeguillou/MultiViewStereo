




#!/usr/bin/python3 bash

MVS_TRAINING="/mnt/c/Users/hugol/OneDrive/Documents/ETHZ/ComputerVision/A_4/code/dtu_dataset/"
python3 train.py --dataset=dtu --batch_size=2 --epochs 4 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt --numdepth=192 --logdir ./checkpoints $@
