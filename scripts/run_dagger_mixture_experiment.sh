#!/usr/bin/env bash
#Specify the command line parameters
train_env='graphEnv2D-v1'
valid_envs=('graphEnvMixture-v1' 'graphEnvMixture-v2' 'graphEnvMixture-v3' 'graphEnvMixture-v4' 'graphEnvMixture-v5' 'graphEnvMixture-v6')


base_folder='../../sp_rl_new_experiments/mixture_datasets/'
folders=(${base_folder}'5' ${base_folder}'15' ${base_folder}'30' ${base_folder}'60' \
         ${base_folder}'80' ${base_folder}'100')

model_files=('dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0'
             'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0'
             'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0')

run_idxs=(5)

printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`


num_iters=10
epsiodes_per_iter=100
num_valid_episodes=0
num_test_episodes=200

model='linear'
expert='length_oracle'
beta0=0.7
alpha=0.001
gamma=0.5
batch_size=32
epochs=1
weight_decay=0.001
seed_val=0


for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_dagger.py --env ${train_env} --valid_env ${valid_envs[idx]} --folder ${folders[idx]} --num_iters ${num_iters}\
         --num_episodes_per_iter ${epsiodes_per_iter} --num_valid_episodes ${num_valid_episodes} --num_test_episodes ${num_test_episodes}\
         --model ${model} --expert ${expert} --beta0 ${beta0} --gamma ${gamma} --alpha ${alpha} --batch_size ${batch_size} --epochs ${epochs}\
         --weight_decay ${weight_decay} --seed_val ${seed_val} --model_file ${model_files[idx]} --test --quad_ftrs  --render --step
done


