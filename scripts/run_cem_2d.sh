#!/usr/bin/env bash
#Specify the command line parameters
train_envs=('graphEnv2D-v1' 'graphEnv2D-v2' 'graphEnv2D-v3' 'graphEnv2D-v4' 'graphEnv2D-v5'\
            'graphEnv2D-v6' 'graphEnv2D-v7' 'graphEnv2D-v8')
valid_envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
            'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')


base_folder='../../sp_rl_new_experiments/'
folders=(${base_folder}'dataset_2d_1/cem_scratch' ${base_folder}'dataset_2d_2/cem_scratch' ${base_folder}'dataset_2d_3/cem_scratch'\
         ${base_folder}'dataset_2d_4/cem_linear' ${base_folder}'dataset_2d_5/cem_scratch' ${base_folder}'dataset_2d_6/cem_scratch'\
         ${base_folder}'dataset_2d_7/cem_scratch' ${base_folder}'dataset_2d_8/cem_scratch')
model_files=('dagger_10_100_100_linear_length_oracle_0.5_0.001_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.001_64_10_0.0001_0'\
             'dagger_10_100_100_linear_length_oracle_0.5_0.001_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.001_64_2_0.0001_0'\
             'dagger_10_100_100_linear_length_oracle_0.5_0.001_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.001_64_10_0.0001_0'\
             'dagger_10_100_100_linear_length_oracle_0.5_0.001_64_10_0.0001_0')

run_idxs=(3)


printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`


num_iters=10
episodes_per_iter=100
num_valid_episodes=100
num_test_episodes=200

model='linear'
batch_size=20
elite_frac=0.2
seed_val=0
init_std=1.0


for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_cem.py --env ${train_envs[idx]} --valid_env ${valid_envs[idx]} --folder ${folders[idx]} --num_iters ${num_iters}\
         --num_episodes_per_iter ${episodes_per_iter} --num_valid_episodes ${num_valid_episodes} --num_test_episodes ${num_test_episodes}\
         --model ${model} --model_file ${model_files[idx]} --batch_size ${batch_size} --elite_frac ${elite_frac} --init_std ${init_std} --seed_val ${seed_val} #--quad_ftrs
done
