#!/usr/bin/env python
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset
# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))


class ExperienceBuffer(Dataset):

  def __init__(self, capacity=np.inf):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, *args):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
        self.memory.append(None)


    self.memory[self.position] = (args[0], args[1], args[2])#, args[3])
    if self.capacity < np.inf:
      self.position = (self.position + 1) % self.capacity
    else:
      self.position = self.position + 1

  def sample(self, batch_size):
    return np.random.choice(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

  def __getitem__(self, idx):
    return self.memory[idx]

  # def save(self, file):
    