#!/bin/bash
declare -r launch_path=/home/jdunnmon/repos/tanda-experimental/experiments/launch_end_models.py
declare -r train_script=/home/jdunnmon/repos/tanda-experimental/experiments/ddsm/ddsm_train.py 
declare -r config_file=/home/jdunnmon/repos/tanda-experimental/experiments/ddsm/config/end_model_only_general.json 
declare -r tan_log_root=/home/zeshanmh/tanda/experiments/ddsm/experiments/log/2017_05_18/tan_only_mammo_0517_tfs17_05_59_07
declare -r output_file=/home/zeshanmh/tanda/experiments/ddsm/output/end_model_benign_malignant_tfs17_lstm_mf_mse_output.txt

CUDA_VISIBLE_DEVICES=0 python $launch_path --script $train_script --end_model_config $config_file --tan_log_root $tan_log_root --model_indexes 0 --procs_lim 1 | tee $output_file

# 4 0 1 5 8 9
