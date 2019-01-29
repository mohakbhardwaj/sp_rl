import numpy as np


def select_forward(feas_actions, itr, env, G):
  return 0

def select_backward(feas_actions, itr, env, G):
  return len(feas_actions) - 1

def select_alternate(feas_actions, itr, env, G):
  if itr%2 == 0:
    return -1
  else:
    return 0

def select_prior(feas_actions, itr, env, G):
  "Choose the edge most likely to be in collision"
  edges = list(map(env.edge_from_action, feas_actions))
  priors = G.get_priors(edges)
  return np.argmax(priors)

def select_multiple(feas_actions, itr, env, G):
  """Choose the best edge according to all heuristics"""
  edges = list(map(env.edge_from_action, feas_actions))
  ftrs = G.get_features(edges, itr)
  h = np.max(ftrs, axis=1)
  return np.argmax(h)


def select_lookahead(feas_actions, itr, env, G):
  k_sp_nodes, k_sp, k_sp_len = G.curr_k_shortest()
  scores = [0]*len(feas_actions)
  for j, action in enumerate(feas_actions):
    edge = env.edge_from_action(action)
    # print "edge", edge, env.G[edge[0]][edge[1]]['status']
    if env.G[edge[0]][edge[1]]['status'] == 0:
      #do something
      for sp in k_sp:
        if (edge[0],edge[1]) in sp or (edge[1],edge[0]) in sp:
          scores[j] += 1
    else:
      continue
  return np.argmax(scores)

def length_oracle(self, feas_actions, iter, env, G):
  curr_sp = G.curr_shortest_path
  curr_sp_len = G.curr_shortest_path_len
  scores = np.array([0.0]*len(feas_actions))
  for (j, action) in enumerate(feas_actions):
    gt_edge = env.gt_edge_from_action(action)
    edge = (gt_edge.source(), gt_edge.target())
    if env.G.edge_properties['status'][gt_edge] == 0:
      G.update_edge(edge, 0)
      new_sp = G.curr_shortest_path
      new_sp_len = G.curr_shortest_path_len
      scores[j] = scores[j] + (new_sp_len-curr_sp_len)
      G.update_edge(edge, -1)
  return np.argmax(scores)