#!/usr/bin/env bash
#Specify the command line parameters
train_envs=('graphEnv2D-v1' 'graphEnv2D-v2' 'graphEnv2D-v3' 'graphEnv2D-v4' 'graphEnv2D-v5'\
            'graphEnv2D-v6' 'graphEnv2D-v7' 'graphEnv2D-v8')
# valid_envs=('graphEnv2DValidation-v1' 'graphEnv2DValidation-v2' 'graphEnv2DValidation-v3' 'graphEnv2DValidation-v4' 'graphEnv2DValidation-v5'\
#             'graphEnv2DValidation-v6' 'graphEnv2DValidation-v7' 'graphEnv2DValidation-v8')


base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
folders=(${base_folder}'dataset_2d_1/q_linear' ${base_folder}'dataset_2d_2/q_linear' ${base_folder}'dataset_2d_3/q_linear'\
         ${base_folder}'dataset_2d_4/q_linear' ${base_folder}'dataset_2d_5/q_linear' ${base_folder}'dataset_2d_6/q_linear'\
         ${base_folder}'dataset_2d_7/q_linear' ${base_folder}'dataset_2d_8/q_linear')

run_idxs=(6)


printf "Changing directories"
echo `pwd`
cd ../examples
echo `pwd`



num_episodes=200
num_exp_episodes=70

model='linear'
expert='length_oracle'
eps0=1.0
epsf=0.1
alpha=0.01
batch_size=32
epochs=3
weight_decay=0.2
seed_val=0


# #
for ((i=0;i<${#run_idxs[@]};++i)); do
  idx=run_idxs[i]
  printf "====Train Environment %s Folder %s\n" "${train_envs[idx]}" "${folders[idx]} ===="
  python2.7 example_q_learning.py --env ${train_envs[idx]} --folder ${folders[idx]} \
            --num_episodes ${num_episodes} --num_exp_episodes ${num_exp_episodes} --model ${model} \
            --eps0 ${eps0} --epsf ${epsf} --alpha ${alpha} --batch_size ${batch_size} \
            --weight_decay ${weight_decay} --seed_val ${seed_val} --lite_ftrs
done
