#!/bin/bash

export DATASET=mosi

CUDA_VISIBLE_DEVICES=0 python run.py \
    --dataset ${DATASET} \
    --batch_size 24 \
    --max_len 128 \
    --seed 0 \
    --seeds 0 1 2 3 10 11 42 66 100 111 222 333 520 666 888 1000 1111 2022 2023 \
    --do_train \
    --do_predict \
    --save_path ./saved_results/${DATASET}/ \
    --device_ids 0 \
    --epoch 20 \
    --lr_bert 1e-5 \
    --lr_other 1e-3 \
    --weight_decay_bert 1e-5 \
    --weight_decay_other 1e-3 \
    --hidden_size 128 \
    --num_lstm_layers 1 \
    --num_gnn_layers 1 \
    --num_gnn_heads 1 \
    --dropout 0.1 \
    --dropout_gnn 0.1 \
    --aug_ratio 0.2 \
    --sup_cl_weight 0.1 \
    --self_cl_weight 0.1 \
    --capsule_dim 128 \
    --capsule_num 40 \
    --routing 3 \
    --multi_dim 128 \
    --Tt 128 \
    --Ta 375 \
    --Tv 500
