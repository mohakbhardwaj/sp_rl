#!/usr/bin/env bash

#Specify the command line parameters
train_envs=('graphEnv2D-v1' 'graphEnv2D-v2' 'graphEnv2D-v3' 'graphEnv2D-v4' 'graphEnv2D-v5'\
            'graphEnv2D-v6' 'graphEnv2D-v7' 'graphEnv2D-v8')
valid_envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
            'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')


base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
folders=(${base_folder}'dataset_2d_1/bc_linear_full' ${base_folder}'dataset_2d_2/bc_linear_full' ${base_folder}'dataset_2d_3/bc_linear_full'\
         ${base_folder}'dataset_2d_4/bc_linear_full' ${base_folder}'dataset_2d_5/bc_linear_full' ${base_folder}'dataset_2d_6/bc_linear_full'\
         ${base_folder}'dataset_2d_7/bc_linear_full' ${base_folder}'dataset_2d_8/bc_linear_full')

run_idxs=(0 1 2 3 5 6)

printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`


num_iters=1
epsiodes_per_iter=20
num_valid_episodes=0
num_test_episodes=200

model='linear'
expert='length_oracle'
beta0=1.0
alpha=0.01
batch_size=32
epochs=3
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