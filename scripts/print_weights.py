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


def print_weights(file):
  with open(os.path.abspath(file), 'r') as fp:
    data = json.load(fp)
    for iter in sorted(data.keys()):
      print('iter = {}, weights = {}, bias = {}'.format(iter, data[iter]['fc.weight'], data[iter]['fc.bias']))
    # print data
  

def main(args):
  for file in args.files:
    print file
    print_weights(file)
  


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Compute metrics for data files')
  parser.add_argument('--files', type=str, required=True, nargs='+')
  args = parser.parse_args()
  main(args)
