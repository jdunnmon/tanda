#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ddsm_train.py --generator gru --gen_lr 0.001 --disc_lr 0.05 --mse_term 0.0 --seq_len 10 --batch_size 25 --n_epochs 10 --end_epochs 10 --run_name ddsm_test --is_test True --debug_every 40 --plot_every 40 --save_model False