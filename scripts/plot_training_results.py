#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
plt.rcParams["font.family"] = "Times New Roman"
import json
import csv
import pprint
import re

colors = ['#d95f02', '#1b9e77']
# legends = ['S']
# fill_colors=['']
alpha = 0.3 
fill_color = 'red'
base_alpha = 0.3
base_fill_color = 'gray'

font = {'family': 'times',
        'size': 10}


def load_raw_data(file):

    # print np.loadtxt(file, delimiter=',', usecols=(0, 2), unpack=True)
  with open(os.path.abspath(file)) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      arr = []
      # print row
      if row[0] == 'train_loss' or row[0] == 'dataset_size' or row[0] == 'train_accs' or row[0] == 'validation_accs': 
        continue
      for i in row[1:]:
        arr2 = []
        for j in i.split()[1:]:
          if j == ']': continue
          s = j.split(']')[0] 
          
          if s == ' ' or s =='\n' or s == None  :
            continue
          arr2.append(float(s))
        arr.append(arr2)

      if row[0] == 'train_rewards':
        train_rewards = np.asarray(arr)
      if row[0] == 'validation_rewards':
        validation_rewards = np.asarray(arr)

  return train_rewards, validation_rewards

def calculate_metrics(raw_results):
  avgs = np.mean(raw_results, axis=1)
  stds = np.std(raw_results, axis=1)
  medians = np.median(raw_results, axis=1)
  lquants = np.quantile(raw_results, 0.4, axis=1)
  uquants = np.quantile(raw_results, 0.6, axis=1)
  return avgs, stds, medians, lquants, uquants

def main(args):
  fig,ax = plt.subplots()

  for i,file in enumerate(args.files):
    train_rewards, validation_rewards = load_raw_data(file)
    _, _, medians, lquants, uquants = calculate_metrics(validation_rewards)
    xdata = range(medians.shape[0])
    ax.plot(xdata, medians, '--', color=colors[i])
    ax.fill_between(xdata, lquants, uquants, alpha=alpha, color=colors[i])
  ax.legend()
  ax.set_xlim([xdata[0], xdata[-1]])
  ax.set_ylim([-110, -20])
  ax.set_xlabel('Training iterations', fontsize=14)
  ax.set_ylabel('Median reward', fontsize=14)

  fig.savefig("/home/mohak/Documents/writing/research/rss_2019_lsp_learning/figs/dagger_plot_herb.pdf", bbox_inches='tight')
  plt.show()
    # print medians.shape, lquants.shape, uquants.shape
  

  


  # raw_results, raw_acc, envs = load_raw_data(args.files)
  # print raw_results.shape, envs.shape
  # avgs, stds, medians, lquants, uquants = calculate_metrics(raw_results) 
  # pp = pprint.PrettyPrinter(depth=6)
  # print('Average Expansions')
  # pp.pprint(avgs)
  # print('Medians')
  # pp.pprint(medians)
  # print('40% lower quantile')
  # pp.pprint(lquants)
  # print('60% upper quantile')
  # pp.pprint(uquants)
  # pp.pprint('Average accuracy')
  # if len(raw_acc) > 0:
  #   avg_accuracy = np.mean(raw_acc) 
  #   pp.pprint(avg_accuracy)
  


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Compute metrics for data files')
  parser.add_argument('--files', type=str, required=True, nargs='+')
  args = parser.parse_args()
  main(args)
