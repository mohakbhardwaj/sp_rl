#!/usr/bin/env bash
base_path='../../sp_rl_new_experiments/dataset_2d_4/lsp_selectors_test/'
files=(${base_path}'select_prior_200.json' ${base_path}'select_posterior_200.json'
	   ${base_path}'select_lookahead_len_200.json' ${base_path}'select_lookahead_prog_200.json'
	   ${base_path}'select_posterior_delta_len_200.json' ${base_path}'length_oracle_200.json')
echo ${files[@]}
python analyze_results.py --files ${files[@]} 

dagger_file='../../sp_rl_new_experiments/dataset_2d_4/dagger_linear/test_results.json'
echo ${dagger_file}
python analyze_results.py --files ${dagger_file} 
