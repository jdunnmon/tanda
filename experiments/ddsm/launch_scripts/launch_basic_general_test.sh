declare -r launch_path=/home/jdunnmon/repos/tanda/experiments/launch_run.py
declare -r train_script=/home/jdunnmon/repos/tanda/experiments/ddsm/ddsm_train.py 
declare -r config_file=/home/jdunnmon/repos/tanda/experiments/ddsm/config/mammo_tan_only_test.json
declare -r log_root=/home/jdunnmon/repos/tanda/experiments/ddsm/experiments/log
declare -r output_file=/home/jdunnmon/repos/tanda/experiments/ddsm/output/test_basic_output.txt

source set_env.sh
CUDA_VISIBLE_DEVICES=0 python $launch_path --script $train_script --config $config_file --procs_lim 1 --log_root $log_root | tee $output_file
