#!/usr/bin/env bash
DTU_TESTING="/Users/tancrede/Desktop/eth/first/cv/a4/codes/mvs/dtu_dataset"
CKPT_FILE="/Users/tancrede/Desktop/eth/first/cv/a4/codes/mvs/checkpoints/model_000000.ckpt"
python eval.py --dataset=dtu_eval --batch_size=1 --testpath=$DTU_TESTING \
--testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
