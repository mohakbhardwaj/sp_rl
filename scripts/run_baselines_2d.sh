#!/usr/bin/env bash
envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
            'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')


base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
folders=(${base_folder}'dataset_2d_1/lsp_selectors_validation' ${base_folder}'dataset_2d_2/lsp_selectors_validation' ${base_folder}'dataset_2d_3/lsp_selectors_validation'\
         ${base_folder}'dataset_2d_4/lsp_selectors_validation' ${base_folder}'dataset_2d_5/lsp_selectors_validation' ${base_folder}'dataset_2d_6/lsp_selectors_validation'\
         ${base_folder}'dataset_2d_7/lsp_selectors_validation' ${base_folder}'dataset_2d_8/lsp_selectors_validation')

run_idxs=(4 5 6 7)

num_episodes=200
seed_val=0




echo `pwd`
cd ../examples
echo `pwd`

for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  echo "=== Environment ${envs[idx]} ==="
  python example_run_all_baselines.py --env ${envs[idx]} --num_episodes ${num_episodes} --folder ${folders[idx]} --seed_val ${seed_val}

done


