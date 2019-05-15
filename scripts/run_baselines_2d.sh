#!/usr/bin/env bash
envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
            'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')


base_folder='../../sp_rl_new_experiments/'
folders=(${base_folder}'dataset_2d_1/lsp_selectors_valid' ${base_folder}'dataset_2d_2/lsp_selectors_valid' ${base_folder}'dataset_2d_3/lsp_selectors_valid'\
         ${base_folder}'dataset_2d_4/lsp_selectors_valid' ${base_folder}'dataset_2d_5/lsp_selectors_valid' ${base_folder}'dataset_2d_6/lsp_selectors_valid'\
         ${base_folder}'dataset_2d_7/lsp_selectors_valid' ${base_folder}'dataset_2d_8/lsp_selectors_valid')
selectors=('select_forward' 'select_backward' 'select_prior' 'select_posterior' 'select_delta_len' 'select_delta_prog' 'length_oracle' 'select_alternate', 'select_posterior_delta_len')

run_idxs=(0 1 2 3 4 5 6)

num_episodes=200
seed_val=0

echo `pwd`
cd ../examples
echo `pwd`

for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  echo "=== Environment ${envs[idx]} ==="
  python2.7 example_heuristic_selector.py --env ${envs[idx]} --heuristics ${selectors[@]} --num_episodes ${num_episodes} --folder ${folders[idx]} --seed_val ${seed_val}

done


