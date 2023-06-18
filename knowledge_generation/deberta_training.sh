#!/bin/bash

# set path to dataset here
lm="deberta-large"
task="rel_tail"
eval_set="val"
num_gpus=4
visible=0,1,2,3

CUDA_VISIBLE_DEVICES=${visible} python -m torch.distributed.launch \
       --nproc_per_node ${num_gpus} baseline.py \
       --params_file baseline/configs/params-${lm}.json \
       --dataroot data_persona_gen/ \
       --exp_name ${lm}-${task}-${eval_set} \
       --eval_dataset ${eval_set}