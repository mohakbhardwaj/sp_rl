#!/usr/bin/env bash

files=('../../sp_rl_new_experiments/dataset_2d_5/dagger_linear_mix/weights_per_iter.json' \
'../../sp_rl_new_experiments/dataset_2d_5/bc_linear/weights_per_iter.json') # \
#'../../sp_rl_new_experiments/dataset_2d_2/dagger_linear_extraf/train_results.csv' \
#'../../sp_rl_new_experiments/dataset_2d_3/dagger_linear_extraf/train_results.csv' \
#'../../sp_rl_new_experiments/dataset_2d_4/dagger_linear_extraf/train_results.csv' \
#'../../sp_rl_new_experiments/dataset_2d_6/dagger_linear_extraf/train_results.csv' \
#'../../sp_rl_new_experiments/dataset_2d_7/dagger_linear_extraf/train_results.csv' \
#'../../sp_rl_new_experiments/dataset_herb_5/dagger_linear_strollr/train_results.csv')

#'../../sp_rl_new_experiments/dataset_2d_5/dagger_linear_extraf/train_results.csv' \


echo ${files[@]}
python print_weights.py --files ${files[@]} 
