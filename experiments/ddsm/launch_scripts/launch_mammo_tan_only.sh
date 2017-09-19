declare -r launch_path=/mnt/repos/tanda/experiments/launch_run.py
declare -r train_script=/mnt/repos/tanda/experiments/ddsm/ddsm_train.py 
declare -r config_file=/mnt/repos/tanda/experiments/ddsm/config/mammo_tan_only_0517.json 
declare -r log_root=/mnt/repos/tanda/experiments/ddsm/experiments/log
declare -r output_file=/mnt/repos/tanda/experiments/ddsm/output/mammo_tan_only_tfs15_output.txt

python $launch_path --script $train_script --config $config_file --procs_lim 12 --log_root $log_root | tee $output_file
