#!/usr/bin/env bash
envs=('graphEnvHerbValidation-v1' 'graphEnvHerbValidation-v2' 'graphEnvHerbValidation-v3' 'graphEnvHerbValidation-v4' 'graphEnvHerbValidation-v5')


base_folder='../../rss_sp_rl/experiments/graph_collision_checking_datasets/'
folders=(${base_folder}'dataset_herb_1/lsp_selectors' ${base_folder}'dataset_herb_2/lsp_selectors' ${base_folder}'dataset_herb_3/lsp_selectors'\
         ${base_folder}'dataset_herb_4/lsp_selectors' ${base_folder}'dataset_herb_5/lsp_selectors')


run_idxs=(3 4)

num_episodes=100
seed_val=0


echo `pwd`
cd ../examples
echo `pwd`

for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  echo "=== Environment ${envs[idx]} ==="
  python example_run_all_baselines.py --env ${envs[idx]} --num_episodes ${num_episodes} --folder ${folders[idx]} --seed_val ${seed_val}

done
