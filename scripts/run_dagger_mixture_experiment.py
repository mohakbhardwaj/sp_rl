#!/usr/bin/env bash
#Specify the command line parameters
train_env='graphEnv2D-v1'
valid_envs=('graphEnvMixture-v1' 'graphEnvMixture-v2' 'graphEnvMixture-v3' 'graphEnvMixture-v4' 'graphEnvMixture-v5' 'graphEnvMixture-v5')


base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
folders=(${base_folder}'mixture_1/5' ${base_folder}'mixture_1/15' ${base_folder}'mixture_1/30' ${base_folder}'mixture_1/60' \
         ${base_folder}'mixture_1/80' ${base_folder}'mixture_1/100')

model_files=('dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0'
             'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0'
             'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0')

run_idxs=(5)

printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`



num_test_episodes=200

model='linear'
expert='length_oracle'
beta0=0.7
alpha=0.001
batch_size=32
epochs=3
weight_decay=0.2
seed_val=0

# #
for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf idx
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_dagger.py --env ${train_env} --valid_env ${valid_envs[idx]} --folder ${folders[idx]} --num_test_episodes ${num_test_episodes}\
         --model ${model} --expert ${expert} --beta0 ${beta0} --alpha ${alpha} --batch_size ${batch_size} --epochs ${epochs}\
         --weight_decay ${weight_decay} --seed_val ${seed_val} --lite_ftrs --model_file ${model_files[idx]} --test
done
