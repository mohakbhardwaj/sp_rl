#!/usr/bin/env bash

train_envs=('graphEnvHerb-v1' 'graphEnvHerb-v2' 'graphEnvHerb-v3' 'graphEnvHerb-v4' 'graphEnvHerb-v5')
valid_envs=('graphEnvHerbValidation-v1' 'graphEnvHerbValidation-v2' 'graphEnvHerbValidation-v3' 'graphEnvHerbValidation-v4' 'graphEnvHerbValidation-v5')


base_folder='../../sp_rl_new_experiments/'
folders=(${base_folder}'dataset_herb_1/bc_linear' ${base_folder}'dataset_herb_2/bc_linear' ${base_folder}'dataset_herb_3/bc_linear'\
         ${base_folder}'dataset_herb_4/bc_linear' ${base_folder}'dataset_herb_5/bc_linear')

run_idxs=(4)

printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`


num_iters=1
episodes_per_iter=200
num_valid_episodes=0
num_test_episodes=200

model='linear'
expert='length_oracle'
beta0=1.0
alpha=0.005
gamma=1.0
momentum=0.0
batch_size=64
epochs=20
weight_decay=0.0001
seed_val=0

for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_dagger.py --env ${train_envs[idx]} --valid_env ${valid_envs[idx]} --folder ${folders[idx]} --num_iters ${num_iters}\
         --num_episodes_per_iter ${episodes_per_iter} --num_valid_episodes ${num_valid_episodes} --num_test_episodes ${num_test_episodes}\
         --model ${model} --expert ${expert} --beta0 ${beta0} --alpha ${alpha} --momentum ${momentum} --gamma ${gamma} --batch_size ${batch_size} --epochs ${epochs}\
         --weight_decay ${weight_decay} --seed_val ${seed_val} --quad_ftrs
done
