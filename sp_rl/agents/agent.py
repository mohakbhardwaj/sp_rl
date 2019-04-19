#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import gym
import sp_rl
import networkx as nx

class Agent(object):
  def __init__(self, env):
    self.env = env

  def filter_path(self, path, obs):
    """This function filters out edges that have already been evaluated"""
    path_f = list(filter(lambda x: obs[x] == -1, path))
    return path_f

  def get_path(self, graph):
    sp = graph.curr_shortest_path
    sp = self.env.to_edge_path(sp) 
    return sp

  # def get_filtered_path(self, graph):
  #   sp = graph.curr_shortest_path
  #   # sp = self.env.to_edge_path(sp)
  #   for node in path:
       
  #   return sp
