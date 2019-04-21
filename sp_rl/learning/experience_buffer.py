#!/usr/bin/env python
import torch
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


class ExperienceBuffer(Dataset):
  def __init__(self, capacity=np.inf):
    self.capacity = capacity
    self.memory_states = []
    self.memory_targets = []
    self.position = 0

  def push(self, states, targets):
    nsamples = states.shape[0]
    """Saves a transition."""
    for i in xrange(nsamples):
      curr_state = states[i]
      curr_target = targets[i]
      if len(self.memory_states) < self.capacity:      
        self.memory_states.append(None)
        self.memory_targets.append(None)

      self.memory_states[self.position]  = curr_state
      self.memory_targets[self.position] = curr_target

      if self.capacity < np.inf:
        self.position = (self.position + 1) % self.capacity
      else:
        self.position = self.position + 1

  # def sample(self, batch_size):
  #   return np.random.choice(self.memory, batch_size)

  def __len__(self):
    return len(self.memory_states)

  def __getitem__(self, idx):
    state = self.memory_states[idx]
    target = self.memory_targets[idx]
    sample = {'state': state, 'target': target}
    return sample

    