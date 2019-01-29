#!/usr/bin/env bash

train_envs=('graphEnvHerb-v1' 'graphEnvHerb-v2' 'graphEnvHerb-v3' 'graphEnvHerb-v4' 'graphEnvHerb-v5')
valid_envs=('graphEnvHerbValidation-v1' 'graphEnvHerbValidation-v2' 'graphEnvHerbValidation-v3' 'graphEnvHerbValidation-v4' 'graphEnvHerbValidation-v5')


base_folder='../../rss_sp_rl/experiments/graph_collision_checking_datasets/'
folders=(${base_folder}'dataset_herb_1/dagger_nn_full' ${base_folder}'dataset_herb_2/dagger_nn_full' ${base_folder}'dataset_herb_3/dagger_nn_full'\
         ${base_folder}'dataset_herb_4/dagger_nn_full' ${base_folder}'dataset_herb_5/dagger_nn_full')

run_idxs=(0 1 2 3 4)

printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`


num_iters=10
epsiodes_per_iter=50
num_valid_episodes=0
num_test_episodes=30

model='mlp'
expert='length_oracle'
beta0=0.7
alpha=0.01
batch_size=32
epochs=7
weight_decay=0.2
seed_val=0

# #
for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf idx
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_dagger.py --env ${train_envs[idx]} --valid_env ${valid_envs[idx]} --folder ${folders[idx]} --num_iters ${num_iters}\
         --num_episodes_per_iter ${epsiodes_per_iter} --num_valid_episodes ${num_valid_episodes} --num_test_episodes ${num_test_episodes}\
         --model ${model} --expert ${expert} --beta0 ${beta0} --alpha ${alpha} --batch_size ${batch_size} --epochs ${epochs}\
         --weight_decay ${weight_decay} --seed_val ${seed_val} --lite_ftrs
done
