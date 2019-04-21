import numpy as np

def select_forward(act_ids, obs, iter, G, env):
  return 0

def select_backward(act_ids, obs, iter, G, env):
  return -1

def select_alternate(act_ids, obs, iter, G, env):
  if iter%2 == 0:
    return -1
  else:
    return 0

def select_prior(act_ids, obs, iter, G, env):
  "Choose the edge most likely to be in collision"
  priors = G.get_priors(act_ids)
  idx_prior = np.argmax(priors)
  # idx_prior = np.random.choice(np.flatnonzero(priors == priors.max()))
  return idx_prior

def select_posterior(act_ids, obs, iter, G, env):
  "Choose the edge most likely to be in collision given all the observations so far"
  post = G.get_posterior(act_ids, obs)
  idx_post = np.argmax(post)
  # idx_post = np.random.choice(np.flatnonzero(post == post.max()))
  return idx_post

def select_delta_len(act_ids, obs,iter, G, env):
  delta_lens = G.get_delta_len_util(act_ids)
  idx_lens = np.argmax(delta_lens)#np.random.choice(np.flatnonzero(delta_lens == delta_lens.max()))
  return idx_lens

def select_delta_prog(act_ids, obs, iter, G, env):
  delta_progs = G.get_delta_prog_util(act_ids)
  idx_prog = np.argmax(delta_progs)#np.random.choice(np.flatnonzero(delta_progs == delta_progs.max()))
  return idx_prog

def select_posterior_delta_len(act_ids, obs, iter, G, env):
  p = G.get_posterior(act_ids, obs)
  dl = G.get_delta_len_util(act_ids)
  pdl = p * dl
  idx_pdl = np.argmax(pdl)
  return idx_pdl 

def select_ksp_centrality(act_ids, obs, iter, G, env):
  ksp_centr = G.get_ksp_centrality(act_ids)
  print ksp_centr
  return np.argmax(ksp_centr)


def select_priorkshort(act_ids, obs, iter, G, env):
  edges = list(map(self.env.edge_from_action, act_ids))
  priors = G.get_priors(edges)
  kshort = [num*1.0/self.ftr_params['k']*1. for num in G.get_k_short_num(edges)]
  h = map(lambda x: x[0]*x[1], zip(kshort,priors))
  return np.argmax(h)

def select_posteriorksp(act_ids, obs, iter, G, env):
  p = G.get_posterior(act_ids, obs)
  ksp_centr = G.get_ksp_centrality(act_ids)
  pksp = p * ksp_centr
  return np.argmax(pksp)


def length_oracle(act_ids, obs, iter, G, env):
  scores = np.zeros(len(act_ids))
  for i, act_id in enumerate(act_ids):
    if env.curr_edge_stats[act_id] == 0:
      scores[i] = G.get_delta_len_util([act_id])
  return np.argmax(scores)

def length_oracle_2(act_ids, obs, iter, G, horizon=2):
  curr_sp = G.curr_shortest_path
  curr_sp_len = G.curr_shortest_path_len
  scores = np.array([0.0]*len(act_ids))
  
  for (j, action) in enumerate(act_ids):
    gt_edge = self.env.gt_edge_from_action(action)
    edge = (gt_edge.source(), gt_edge.target())
    if self.env.G.edge_properties['status'][gt_edge] == 0:
      G.update_edge(edge, 0)
      new_sp = G.curr_shortest_path
      new_sp_len = G.curr_shortest_path_len
      reward = (new_sp_len - curr_sp_len)

      edge_sp = G.in_edge_tup_form(new_sp)
      f_a = [self.env.action_from_edge(e) for e in edge_sp]
      value = np.max(self.get_scores(f_a, iter, G))
      scores[j] = reward + value
      G.update_edge(edge, -1, 0)
  
  return np.argmax(scores)



