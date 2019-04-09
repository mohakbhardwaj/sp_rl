#!/usr/bin/env bash
base_path='../../sp_rl_new_experiments/dataset_2d_7/lsp_selectors_test/'
# files=(${base_path}'select_forward_200.json' ${base_path}'select_backward_200.json' 
# 	   ${base_path}'select_alternate_200.json' ${base_path}'select_prior_200.json' 
# 	   ${base_path}'select_posterior_200.json')
files=(${base_path}'select_prior_200.json' ${base_path}'select_posterior_200.json'
	   ${base_path}'select_lookahead_len_200.json' ${base_path}'select_lookahead_prog_200.json'
	   ${base_path}'select_posterior_delta_len_200.json')
echo ${files[@]}
python analyze_results.py --files ${files[@]} 