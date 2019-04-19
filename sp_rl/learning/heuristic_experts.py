  def select_forward(self, act_ids, obs, iter, G):
    return act_ids[0]

  def select_backward(self, act_ids, obs, iter, G):
    return act_ids[-1]

  def select_alternate(self, act_ids, obs, iter, G):
    if iter%2 == 0:
      return act_ids[-1]
    else:
      return act_ids[0]

  def select_prior(self, act_ids, obs, iter, G):
    "Choose the edge most likely to be in collision"
    priors = G.get_priors(act_ids)
    idx_prior = np.argmax(priors)
    # idx_prior = np.random.choice(np.flatnonzero(priors == priors.max()))
    return act_ids[idx_prior]

  def select_posterior(self, act_ids, obs, iter, G):
    "Choose the edge most likely to be in collision given all the observations so far"
    post = G.get_posterior(act_ids, obs)
    idx_post = np.argmax(post)
    # idx_post = np.random.choice(np.flatnonzero(post == post.max()))
    return act_ids[idx_post]

  def select_delta_len(self, act_ids, obs,iter, G):
    delta_lens = G.get_delta_len_util(act_ids)
    idx_lens = np.argmax(delta_lens)#np.random.choice(np.flatnonzero(delta_lens == delta_lens.max()))
    return act_ids[idx_lens]

  def select_delta_prog(self, act_ids, obs, iter, G):
    delta_progs = G.get_delta_prog_util(act_ids)
    idx_prog = np.argmax(delta_progs)#np.random.choice(np.flatnonzero(delta_progs == delta_progs.max()))
    return act_ids[idx_prog]

  def select_posterior_delta_len(self, act_ids, obs, iter, G):
    p = G.get_posterior(act_ids, obs)
    dl = G.get_delta_len_util(act_ids)
    pdl = p * dl
    idx_pdl = np.argmax(pdl)
    return act_ids[idx_pdl] 

  def select_ksp_centrality(self, act_ids, obs, iter, G):
    ksp_centr = G.get_ksp_centrality(act_ids)
    print ksp_centr
    return act_ids[np.argmax(ksp_centr)]


  def select_priorkshort(self, act_ids, obs, iter, G):
    edges = list(map(self.env.edge_from_action, act_ids))
    priors = G.get_priors(edges)
    kshort = [num*1.0/self.ftr_params['k']*1. for num in G.get_k_short_num(edges)]
    h = map(lambda x: x[0]*x[1], zip(kshort,priors))
    return act_ids[np.argmax(h)]
  
  def select_posteriorksp(seld, act_ids, obs, iter, G):
    p = G.get_posterior(act_ids, obs)
    ksp_centr = G.get_ksp_centrality(act_ids)
    pksp = p * ksp_centr
    return act_ids[np.argmax(pksp)]


  def length_oracle(self, act_ids, obs, iter, G, horizon=2):
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
        scores[j] = scores[j] + (new_sp_len-curr_sp_len)
        G.update_edge(edge, -1, 0)
    action = act_ids[np.argmax(scores)]
    return action

  def length_oracle_2(self, act_ids, obs, iter, G, horizon=2):
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
    
    action = act_ids[np.argmax(scores)]
    return action



  def get_scores(self, feas_actions, iter, G):
    curr_sp = G.curr_shortest_path
    curr_sp_len = G.curr_shortest_path_len
    scores = [0.0]*len(feas_actions)
    for (j, action) in enumerate(feas_actions):
      # edge = self.env.edge_from_action(action)
      gt_edge = self.env.gt_edge_from_action(action)
      edge = (gt_edge.source(), gt_edge.target())
      if self.env.G.edge_properties['status'][gt_edge] == 0:
        G.update_edge(edge, 0)
        new_sp = G.curr_shortest_path
        if len(new_sp) > 0:
          new_sp_len = G.curr_shortest_path_len
          reward = new_sp_len-curr_sp_len
        else: reward = np.inf
        scores[j] = reward
        G.update_edge(edge, -1)
    return scores