#!/usr/bin/env bash
# envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
#             'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')
envs=('graphEnv2DTest-v1' 'graphEnv2DTest-v2' 'graphEnv2DTest-v3' 'graphEnv2DTest-v4' 'graphEnv2DTest-v5'\
            'graphEnv2DTest-v6' 'graphEnv2DTest-v7' 'graphEnv2DTest-v8')


base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
folders=(${base_folder}'dataset_2d_1/lsp_selectors_test' ${base_folder}'dataset_2d_2/lsp_selectors_test' ${base_folder}'dataset_2d_3/lsp_selectors_test'\
         ${base_folder}'dataset_2d_4/lsp_selectors_test' ${base_folder}'dataset_2d_5/lsp_selectors_test' ${base_folder}'dataset_2d_6/lsp_selectors_test'\
         ${base_folder}'dataset_2d_7/lsp_selectors_test' ${base_folder}'dataset_2d_8/lsp_selectors_test')

run_idxs=(0 1 2 3 5 6 7)

num_episodes=200
seed_val=0




echo `pwd`
cd ../examples
echo `pwd`

for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  echo "=== Environment ${envs[idx]} ==="
  python2.7 example_run_all_baselines.py --env ${envs[idx]} --num_episodes ${num_episodes} --folder ${folders[idx]} --seed_val ${seed_val}

done


