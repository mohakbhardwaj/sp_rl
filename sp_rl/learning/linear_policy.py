import numpy as np

class LinearPolicy(object):
  def __init__(self, w_init, b_init, alpha):
    self.ndims = w_init.size
    self.alpha = alpha
    self.w = w_init.reshape(self.ndims, 1)
    self.b = b_init

  def predict(self, ftr_batch):
    """Takes as input a batch of features and returns a batch of Q values"""

    n_batch = ftr_batch.shape[0]
    print "Predicting, current weights = {}".format([self.w, self.b])#, ftr_batch.shape, self.w.shape
    scores =  np.dot(ftr_batch, self.w).reshape(n_batch, 1)# + self.b
    probs = 1.0#Add sigmoid here

  def update_weights_grad(self, ftr_batch, loss_batch):
    # print "Updating, ", ftr_batch.shape, loss_batch.shape
    # n_samples = ftr_batch.shape[0]

    # delta_w = np.dot(ftr_batch.T, loss_batch)/(n_samples*1.0)
    # delta_b = np.dot(np.ones((1,n_samples)), loss_batch)/(n_samples*1.0)
    # self.w = self.w + self.alpha * delta_w 
    # self.b = self.b + self.alpha * delta_b#
    raise NotImplementedError

  def update_weights_lsq(self, ftr_batch, targets):
    raise NotImplementedError
