#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experience_buffer import ExperienceBuffer
from torch_models import LinearNet



class QFunction():
  def __init__(self, model, batch_size, train_method='supervised', optim_method='adam', lr=0.001, momentum=0.9, weight_decay=0.1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.model = model
    self.criterion = nn.MSELoss()
    if optim_method =='adam':
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_method == 'sgd':
      self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    self.batch_size = batch_size
    self.train_method = train_method

  def predict(self, ftrs):
    return self.model(ftrs)

  def collate_fn(self, batch):
    action = [item[0] for item in batch]
    reward = [item[1] for item in batch]
    next_state = [item[2] for item in batch]
    # target = torch.LongTensor(target)
    return [action, reward, next_state]

  def train(self, D, gamma = 1.0, epochs=2):
    trainloader = DataLoader(D, batch_size=self.batch_size,
                               shuffle=True,
                               collate_fn=self.collate_fn, 
                               num_workers=0)
    data_pts = len(D)
    num_batches = float(data_pts/self.batch_size)
    if num_batches == 0.0: num_batches = 1
    avg_loss = 0.0
    # print num_batches, data_pts, epochs
    
    if self.train_method == 'supervised':
      for epoch in xrange(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
          actions, targets = data
          self.optimizer.zero_grad()
          outputs = self.model(actions)
          loss = self.criterion(outputs, targets)
          loss.backward()
          self.optimizer.step()

          running_loss += loss.item()
          # if i % 10 == 9:
          #   print('[%d, %5d] loss: %.3f' %
          #           (epoch + 1, i + 1, running_loss / (epoch*num_batches + i*1.0)))
          # raw_input('...')
        
          avg_loss += running_loss
      
      avg_loss = avg_loss/(epochs*num_batches*1.0)
      print 'Average loss = '.format(avg_loss)
    elif self.train_method == 'reinforcement':
      data = next(iter(trainloader))
      action, reward, next_state = data

      loss = 0.0
      target_b = []
      for i in xrange(len(next_state)):
        # print action[i]
        # print next_state[i].shape

        q_val_b = self.predict(action[i].unsqueeze(0))
        q_valsd = self.predict(next_state[i])
        best_q = torch.max(q_valsd)
        target = reward[i] + gamma * best_q
        # print  target
        target_b.append(target)
      
      self.optimizer.zero_grad()
      loss = self.criterion(q_val_b, torch.tensor(target_b))
      loss.backward()
      self.optimizer.step()

    return loss.item()

class Policy():
  def __init__(self, model, batch_size, train_method='supervised', optim_method='adam', lr=0.001, momentum=0.9, weight_decay=0.1, use_cuda=False):
    self.use_cuda = torch.cuda.is_available() if use_cuda else False
    self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu') 
    self.model = model
    self.train_method = train_method
    self.batch_size = batch_size
    if self.train_method == 'supervised':
      self.criterion = nn.CrossEntropyLoss()
    if optim_method =='adam':
      self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_method == 'sgd':
      self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)

  def predict(self, ftrs):
    scores = self.model(ftrs)
    # print scores
    probs = nn.LogSoftmax(dim=0)(scores) 
    # print scores, probs
    return probs

  def collate_fn(self, batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

  def train(self, D, epochs=2):
    trainloader = DataLoader(dataset=D, 
                             batch_size=self.batch_size,
                             shuffle=True, 
                             collate_fn = self.collate_fn,
                             num_workers=0)
    data_pts = len(D)
    # print('Number of datapoints = {}'.format(data_pts))
    num_batches = float(data_pts/self.batch_size)
    if num_batches == 0.0: num_batches = 1
    avg_loss = 0.0
    self.model.to(self.device)
    # print num_batches, data_pts, epochs
    for epoch in xrange(epochs):
      running_loss = 0.0
      for i, data in enumerate(trainloader,0):
        if self.train_method == 'supervised':
          state, targets = data
          # state.to(self.device), target.to(self.device)
          loss = self.cross_entropy_loss(state, targets)
          # print outputs
          # for i in xrange(actions.shape[0]):
            # print actions[i,:], targets[i,:]
        # elif self.train_method == 'reinforcement':
        #   action, next_state, reward = data
        #   q_vals = self.predict(next_state)
        #   outputs = self.model(actions)

        #zero the parameter gradients
        self.optimizer.zero_grad()
        
        # print (loss)
        # loss = self.criterion(outputs, targets)
        loss.backward()

        self.optimizer.step()
        # print ('step done')

        #print statistics
        running_loss += loss.item()
        # if i % 10 == 9:
        #   print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / (epoch*num_batches + i*1.0)))
        # # raw_input('...')
      
      avg_loss += running_loss
    
    avg_loss = avg_loss/(epochs*num_batches*1.0)
  
    # print('Policy trained. Average loss = {}'.format(avg_loss))
    return avg_loss

  def cross_entropy_loss(self, state, targets):
    loss = 0.0
    for i in xrange(len(state)):
      ftrs = state[i]
      ftrs = ftrs.to(self.device)
      logprobs = self.predict(ftrs)
      curr_t = targets[i].to(self.device)
      loss += -1.0*logprobs[curr_t] #-1.0*torch.log(probs[curr_t])#Do cross entropy loss here
    loss = loss/len(state)*1.0
    # print "Loss = ", loss
    # raw_input('..')
    return loss
