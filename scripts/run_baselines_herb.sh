#!/usr/bin/env bash
envs=('graphEnvHerbValidation-v1' 'graphEnvHerbValidation-v2' 'graphEnvHerbValidation-v3' 'graphEnvHerbValidation-v4' 'graphEnvHerbValidation-v5')
#envs=('graphEnvHerbTest-v1' 'graphEnvHerbTest-v2' 'graphEnvHerbTest-v3' 'graphEnvHerbTest-v4' 'graphEnvHerbTest-v5')


base_folder='../../sp_rl_new_experiments'
folders=(${base_folder}'dataset_herb_1/lsp_selectors_test' ${base_folder}'dataset_herb_2/lsp_selectors_test' ${base_folder}'dataset_herb_3/lsp_selectors_test'\
         ${base_folder}'dataset_herb_4/lsp_selectors_test' ${base_folder}'dataset_herb_5/lsp_selectors_test')

selectors=('select_forward' 'select_backward' 'select_prior' 'select_posterior' 'select_delta_len' 'select_delta_prog' 'length_oracle' 'select_alternate')
run_idxs=(3 4)

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
