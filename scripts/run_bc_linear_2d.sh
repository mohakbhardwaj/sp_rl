#!/usr/bin/env bash
#Specify the command line parameters
train_envs=('graphEnv2D-v1' 'graphEnv2D-v2' 'graphEnv2D-v3' 'graphEnv2D-v4' 'graphEnv2D-v5'\
            'graphEnv2D-v6' 'graphEnv2D-v7' 'graphEnv2D-v8')
valid_envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
            'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')


base_folder='../../sp_rl_new_experiments/'
folders=(${base_folder}'dataset_2d_1/bc_linear' ${base_folder}'dataset_2d_2/bc_linear' ${base_folder}'dataset_2d_3/bc_linear'\
         ${base_folder}'dataset_2d_4/bc_linear' ${base_folder}'dataset_2d_5/bc_linear' ${base_folder}'dataset_2d_6/bc_linear'\
         ${base_folder}'dataset_2d_7/bc_linear' ${base_folder}'dataset_2d_8/bc_linear')

run_idxs=(7)


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
momentum=0.0
gamma=1.0
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
