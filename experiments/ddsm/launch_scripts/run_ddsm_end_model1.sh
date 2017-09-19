#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ddsm_train.py --generator gru --mse_term .001 --seq_len 15 --gen_lr 1e-3 --disc_lr 1e-4 --end_discriminator dcnn --end_epochs 25 --end_lr .001 --end_batch_size 25 --end_model_only True --run_name ddsm_end_model --is_test True --run_index 1 --log_path /home/zeshanmh/tanda/experiments/ddsm/experiments/log/2017_04_12/mammo_end_run_11_40_48