
# env='graphEnvHerbValidation-v4'
# base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
# folder=${base_folder}'dataset_herb_4/dagger_linear_full'
# policy_files=('dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_1'\
#   'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_2' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_3' \
#   'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_4' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_5' \
#   'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_6' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_7' \
#   'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_8' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_9')

# model='linear'
# num_episodes=100
# seed_val=0

# python2.7 validate_policies.py  --env ${env} --files ${policy_files[@]} --folder ${folder} --num_episodes ${num_episodes} --seed_val ${seed_val} --lite_ftrs --model ${model}


#!/usr/bin/env bash
#Specify the command line parameters
# train_envs=('graphEnv2D-v1' 'graphEnv2D-v2' 'graphEnv2D-v3' 'graphEnv2D-v4' 'graphEnv2D-v5' 'graphEnv2D-v6' 'graphEnv2D-v7' 'graphEnv2D-v8' \
#             'graphEnvHerb-v4' 'graphEnvHerb-v5')
# valid_envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5' \
#             'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8' 'graphEnvHerbValidation-v4' 'graphEnvHerbValidation-v5')
valid_env='graphEnv2DValidation-v7'

# base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/dataset_2d_7/dagger_linear_full/'
# folders=(${base_folder}'policy_0/' ${base_folder}'policy_1/' ${base_folder}'policy_2/'\
#          ${base_folder}'policy_3/' ${base_folder}'policy_4/' ${base_folder}'policy_5/'\
#          ${base_folder}'policy_6/' ${base_folder}'policy_7/'\
#          ${base_folder}'policy_8/' ${base_folder}'policy_9/')


# model_files=('dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_1'
#              'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_2' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_3'
#              'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_4' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_5' \
#              'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_6' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_7' \
#              'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_8' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_9')

base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/dataset_2d_7/dagger_linear_hr/'
folders=(${base_folder}'policy_0/' ${base_folder}'policy_1/' ${base_folder}'policy_2/'\
         ${base_folder}'policy_3/' ${base_folder}'policy_4/' ${base_folder}'policy_5/'\
         ${base_folder}'policy_6/' ${base_folder}'policy_7/'\
         ${base_folder}'policy_8/' ${base_folder}'policy_9/')


model_files=('dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_1'
             'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_2' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_3'
             'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_4' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_5' \
             'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_6' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_7' \
             'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_8' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0_policy_int_9')


run_idxs=(0 1 2 3 4 5 6 7 8 9)

printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`


num_iters=10
epsiodes_per_iter=20
num_valid_episodes=0
num_test_episodes=200

model='linear'
expert='length_oracle'
beta0=0.7
alpha=0.001
batch_size=32
epochs=3
weight_decay=0.2
seed_val=0


for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_dagger.py --env ${valid_env} --valid_env ${valid_env} --folder ${folders[idx]} --num_iters ${num_iters}\
         --num_episodes_per_iter ${epsiodes_per_iter} --num_valid_episodes ${num_valid_episodes} --num_test_episodes ${num_test_episodes}\
         --model ${model} --expert ${expert} --beta0 ${beta0} --alpha ${alpha} --batch_size ${batch_size} --epochs ${epochs}\
         --weight_decay ${weight_decay} --seed_val ${seed_val} --lite_ftrs --model_file ${model_files[idx]} --test
done
