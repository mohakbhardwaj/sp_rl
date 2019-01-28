#!/usr/bin/env bash
baselines=('select_forward' 'select_backward' 'select_alternate' 'select_prior' 'length_oracle')
envs=('graphEnvHerbValidation-v1' 'graphEnvHerbValidation-v2' 'graphEnvHerbValidation-v3' 'graphEnvHerbValidation-v4' 'graphEnvHerbValidation-v5')


base_folder='../../rss_sp_rl/experiments/graph_collision_checking_datasets/'
folders=(${base_folder}'dataset_herb_1/lsp_selectors' ${base_folder}'dataset_herb_2/lsp_selectors' ${base_folder}'dataset_herb_3/lsp_selectors'\
         ${base_folder}'dataset_herb_4/lsp_selectors' ${base_folder}'dataset_herb_5/lsp_selectors')
         # ${base_folder}'dataset_2d_8/lsp_selectors')


num_episodes=200
seed_val=0




echo `pwd`
cd ../examples
echo `pwd`

for base in ${baselines[@]}; do
  for((i=0;i<${#envs[@]};++i)); do
  echo "=== Baseline ${base}, Environment ${envs[i]} ==="
  python example_heuristic_selector.py --env ${envs[i]} --heuristic ${base} --num_episodes ${num_episodes} --folder ${folders[i]} --seed_val ${seed_val}
  done
done


