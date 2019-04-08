#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pprint
def load_raw_data(file_paths):
  raw_results = []
  envs = [] 
  raw_acc = []
  for f_name in file_paths:
    raw_data = {}
    with open(os.path.abspath(f_name), 'r') as fp:
      data = json.load(fp)
      raw_data = data['test_rewards']
      if "test_acc" in data:
        raw_data_acc=data['test_acc']
        raw_acc.append(list(raw_data_acc.values()))
    raw_results.append(list(raw_data.values()))
    envs.append(list(raw_data.keys()))  
  return np.array(raw_results), raw_acc, np.array(envs)

def calculate_metrics(raw_results):
  avgs = np.mean(raw_results, axis=1)
  stds = np.std(raw_results, axis=1)
  medians = np.median(raw_results, axis=1)
  lquants = np.quantile(raw_results, 0.4, axis=1)
  uquants = np.quantile(raw_results, 0.6, axis=1)
  return avgs, stds, medians, lquants, uquants

def main(args):
  raw_results, raw_acc, envs = load_raw_data(args.files)
  print raw_results.shape, envs.shape
  avgs, stds, medians, lquants, uquants = calculate_metrics(raw_results) 
  pp = pprint.PrettyPrinter(depth=6)
  print('Average Expansions')
  pp.pprint(avgs)
  print('Medians')
  pp.pprint(medians)
  print('40% lower quantile')
  pp.pprint(lquants)
  print('60% upper quantile')
  pp.pprint(uquants)
  pp.pprint('Average accuracy')
  if len(raw_acc) > 0:
    avg_accuracy = np.mean(raw_acc) 
    pp.pprint(avg_accuracy)
  


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Compute metrics for data files')
  parser.add_argument('--files', type=str, required=True, nargs='+')
  args = parser.parse_args()
  main(args)


# median
# 40% lower quantile
# 60% upper quantile