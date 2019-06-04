#!/usr/bin/env bash
#Specify the command line parameters
train_envs=('graphEnv2D-v1' 'graphEnv2D-v2' 'graphEnv2D-v3' 'graphEnv2D-v4' 'graphEnv2D-v5' 'graphEnv2D-v6' 'graphEnv2D-v7' 'graphEnv2D-v8' \
            'graphEnvHerb-v4' 'graphEnvHerb-v5')
valid_envs=('graphEnv2DTest-v1' 'graphEnv2DTest-v2' 'graphEnv2DTest-v3' 'graphEnv2DTest-v4' 'graphEnv2DTest-v5' \
            'graphEnv2DTest-v6' 'graphEnv2DTest-v7' 'graphEnv2DTest-v8' 'graphEnvHerbTest-v4' 'graphEnvHerbTest-v5')


base_folder='../../sp_rl_new_experiments/'

#Dagger linear
folders=(${base_folder}'dataset_2d_1/dagger_linear_extraf' ${base_folder}'dataset_2d_2/dagger_linear_extraf' ${base_folder}'dataset_2d_3/dagger_linear_extraf'\
         ${base_folder}'dataset_2d_4/dagger_linear_extraf' ${base_folder}'dataset_2d_5/dagger_linear_mix' ${base_folder}'dataset_2d_6/dagger_linear_extraf'\
         ${base_folder}'dataset_2d_7/dagger_linear_extraf' ${base_folder}'dataset_2d_8/dagger_linear_extraf'\
         ${base_folder}'dataset_herb_4/dagger_linear_mix'  ${base_folder}'dataset_herb_5/dagger_linear_mix')


model_files=('dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0'\
             'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0' 'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0'\
             'testing' 'testing'\
             'dagger_10_100_100_linear_length_oracle_0.5_0.005_64_10_0.0001_0', 'dagger_10_100_60_linear_length_oracle_0.5_0.005_64_10_0.0001_0'\
             'dagger_10_100_60_linear_length_oracle_0.5_0.005_64_10_0.0001_0', 'dagger_10_100_60_linear_length_oracle_0.5_0.005_64_10_0.0001_0')

#Dagger linear with heuristic roll-out
# folders=(${base_folder}'dataset_2d_1/dagger_linear_hr' ${base_folder}'dataset_2d_2/dagger_linear_hr' ${base_folder}'dataset_2d_3/dagger_linear_hr'\
#          ${base_folder}'dataset_2d_4/dagger_linear_hr' ${base_folder}'dataset_2d_5/dagger_linear_hr' ${base_folder}'dataset_2d_6/dagger_linear_hr'\
#          ${base_folder}'dataset_2d_7/dagger_linear_hr' ${base_folder}'dataset_2d_8/dagger_linear_hr'\
#          ${base_folder}'dataset_herb_4/dagger_linear_hrfull' ${base_folder}'dataset_herb_5/dagger_linear_hrfull')

# model_files=('dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0'
#              'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0'
#              'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' \
#              'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' \
#              'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0'  'dagger_10_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0')

#Behavior cloning linear
# folders=(${base_folder}'dataset_2d_1/bc_linear' ${base_folder}'dataset_2d_2/bc_linear' ${base_folder}'dataset_2d_3/bc_linear'\
#          ${base_folder}'dataset_2d_4/bc_linear' ${base_folder}'dataset_2d_5/bc_linear' ${base_folder}'dataset_2d_6/bc_linear'\
#          ${base_folder}'dataset_2d_7/bc_linear' ${base_folder}'dataset_2d_8/bc_linear'\
#          ${base_folder}'dataset_herb_4/bc_linear' ${base_folder}'dataset_herb_5/bc_linear')

# model_files=('dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_1_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0'
#              'dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0'
#              'dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' \
#              'dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' 'dagger_1_20_0_linear_length_oracle_0.7_0.001_32_3_0.2_lite_ftrs_0' \
#              'dagger_1_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0'  'dagger_1_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0')



run_idxs=(4)
# 3 5 6 7 8 9)
# 8 9)

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
gamma=0.5
batch_size=32
epochs=1
weight_decay=0.001
seed_val=0


for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Train Environment %s Validation Environment %s Folder %s\n" "${train_envs[idx]}" "${valid_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_dagger.py --env ${train_envs[idx]} --valid_env ${valid_envs[idx]} --folder ${folders[idx]} --num_iters ${num_iters}\
         --num_episodes_per_iter ${epsiodes_per_iter} --num_valid_episodes ${num_valid_episodes} --num_test_episodes ${num_test_episodes}\
         --model ${model} --expert ${expert} --beta0 ${beta0} --gamma ${gamma} --alpha ${alpha} --batch_size ${batch_size} --epochs ${epochs}\
         --weight_decay ${weight_decay} --seed_val ${seed_val} --model_file ${model_files[idx]} --test --quad_ftrs # --render --step
done
