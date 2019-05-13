#!/usr/bin/env bash
base_path='../../sp_rl_new_experiments/dataset_2d_2/lsp_selectors_test/'
files=(${base_path}'select_prior_200.json' ${base_path}'select_posterior_200.json')
	  # ${base_path}'select_lookahead_len_200.json' ${base_path}'select_lookahead_prog_200.json'
	  # ${base_path}'select_posterior_delta_len_200.json' ${base_path}'length_oracle_200.json')
echo ${files[@]}
python analyze_results.py --files ${files[@]} 

dagger_file1='../../sp_rl_new_experiments/dataset_2d_1/dagger_linear_mix/test_results.json'
dagger_file2='../../sp_rl_new_experiments/dataset_2d_2/dagger_linear_mix/test_results.json'
dagger_file3='../../sp_rl_new_experiments/dataset_2d_3/dagger_linear_mix/test_results.json'
dagger_file4='../../sp_rl_new_experiments/dataset_2d_4/dagger_linear_mix/test_results.json'
dagger_file5='../../sp_rl_new_experiments/dataset_2d_5/dagger_linear_mix/test_results.json'
dagger_file6='../../sp_rl_new_experiments/dataset_2d_6/dagger_linear_mix/test_results.json'
dagger_file7='../../sp_rl_new_experiments/dataset_2d_7/dagger_linear_mix/test_results.json'

echo ${dagger_file}
python analyze_results.py --files ${dagger_file3}
