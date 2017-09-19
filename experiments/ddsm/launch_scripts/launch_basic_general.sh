declare -r launch_path=/home/jdunnmon/repos/tanda-experimental/experiments/launch_run.py
declare -r train_script=/home/jdunnmon/repos/tanda-experimental/experiments/ddsm/ddsm_train.py 
declare -r config_file=/home/jdunnmon/repos/tanda-experimental/experiments/ddsm/config/end_model_only_general.json
declare -r log_root=/home/jdunnmon/repos/tanda-experimental/experiments/ddsm/experiments/log
declare -r output_file=/home/jdunnmon/repos/tanda-experimental/experiments/ddsm/output/test_basic_output.txt

CUDA_VISIBLE_DEVICES=0 python $launch_path --script $train_script --config $config_file --procs_lim 1 --log_root $log_root | tee $output_file
