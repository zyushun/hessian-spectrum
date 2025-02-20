#!/bin/bash

python -um main \
  --data 'imagenet'\
  --lr 0.1\
  --beta1 0.9\
  --beta2  0.999\
  --wd 1e-4\
  --workers 48\
  --epochs 90\
  --arch 'vit_base'\
  --opt 'adamw'\
  --seed 32 \
  --batchsize 256\
  --epsilon 1e-8\
  --resume ''\
  --load_iter 0\
  --comment '' \

