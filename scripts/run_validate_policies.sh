
env='graphEnvHerbValidation-v4'
base_folder='../../rss_sp_rl/experiments/rss_lsp_datasets/'
folder=${base_folder}'dataset_herb_4/dagger_linear_full'
policy_files=('dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_0' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_1'\
  'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_2' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_3' \
  'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_4' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_5' \
  'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_6' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_7' \
  'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_8' 'dagger_10_20_0_linear_length_oracle_0.7_0.01_32_3_0.2_lite_ftrs_0_policy_int_9')

model='linear'
num_episodes=100
seed_val=0

python2.7 validate_policies.py  --env ${env} --files ${policy_files[@]} --folder ${folder} --num_episodes ${num_episodes} --seed_val ${seed_val} --lite_ftrs --model ${model}