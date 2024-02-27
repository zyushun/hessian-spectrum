#!/bin/bash


set -e #print
set -x

export CUDA_VISIBLE_DEVICES="1"
for o in  0 
do
  python -um main \
    --data 'imagenet/'\
    --lr 0.1\
    --beta1 0.9\
    --beta2  0.999\
    --wd 1e-4\
    --workers 48\
    --epochs 90\
    --arch 'vit'\
    --opt 'adamw'\
    --seed 32 \
    --batchsize 256\
    --epsilon 1e-8\
    --resume 'checkpoint/'\
    --load_iter $o\
    --comment '' \
done

