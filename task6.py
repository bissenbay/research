import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataGenerator:
  def __init__(self, n_nodes):
    self.n_nodes = n_nodes

  def get_train_data(self, batch_size):
    train_data = torch.rand([batch_size, self.n_nodes, 2]).to(device)
    return train_data

  def get_test_data(self, test_size):
    filename = '/home/shrek/Desktop/tsp-{}-instances-{}-n_nodes.pt'.format(test_size, self.n_nodes)

    if os.path.exists(filename):
        test_data = torch.load(filename, map_location=torch.device(device))
    else:
        test_data = torch.rand([test_size, self.n_nodes, 2]).to(device)
        torch.save(test_data, filename)
    return test_data

class Environment:
  def __init__(self, batch_size, n_nodes):
    self.batch_size = batch_size
    self.n_nodes = n_nodes

  def reset(self, A):
    self.D = torch.cdist(A, A)
    self.cur_loc = torch.randint(self.n_nodes - 1, self.n_nodes, (self.batch_size,)).to(device)
    self.r_total = torch.zeros(self.batch_size).to(device)
    self.v_nodes = torch.zeros([self.batch_size, self.n_nodes], dtype=torch.int32).to(device)
    self.v_nodes[torch.arange(self.batch_size), self.n_nodes - 1] = 1
    return self.cur_loc, self.v_nodes

  def step(self, a):
      last_node = self.cur_loc
      self.cur_loc = a
      self.v_nodes[torch.arange(self.batch_size), self.cur_loc] = 1
      self.v_nodes = torch.cat((self.v_nodes[:, :self.n_nodes - 1], (self.v_nodes.sum(dim=1) < self.n_nodes).to(torch.int32).unsqueeze(1)), 1)
      self.r_total += self.D[torch.arange(self.batch_size), self.cur_loc, last_node]
      return self.cur_loc, self.v_nodes

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv1d(input_size, hidden_size, 1).to(device)

  def forward(self, input):
    output = self.conv1(input)
    return output

class Decoder(nn.Module):
  def __init__(self, hidden_size, p):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=False, batch_first=True, bidirectional=False).to(device)
    self.attention = Attention(hidden_size).to(device)
    self.dropout_c = nn.Dropout(p).to(device)
    self.dropout_h = nn.Dropout(p).to(device)

  def forward(self, emb_cur_loc, last_hh, emb_graph):
    out, last_hh = self.lstm(emb_cur_loc, last_hh)
    out = out.squeeze(1)
    out = self.dropout_c(out)
    h_0 = self.dropout_h(last_hh[0])
    c_0 = self.dropout_h(last_hh[1])
    last_hh = (h_0, c_0)
    hy = last_hh[0].squeeze(0)
    logits = self.attention(emb_graph, hy)
    return logits, last_hh

class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.proj_emb_graph = nn.Conv1d(hidden_size, hidden_size, 1).to(device)
    self.proj_hidden = nn.Linear(hidden_size, hidden_size).to(device)
    self.V = nn.parameter.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True).to(device)).to(device)

  def forward(self, emb_graph, hidden_decoder):
    proj_emb_graph = self.proj_emb_graph(emb_graph)
    proj_hidden = self.proj_hidden(hidden_decoder)
    q = proj_hidden.view(emb_graph.shape[0], -1, 1).expand(emb_graph.shape[0], self.hidden_size, emb_graph.shape[2])
    v = self.V.expand(emb_graph.shape[0], 1, self.hidden_size)
    U = torch.bmm(v, torch.tanh(proj_emb_graph + q).to(device))
    return U.squeeze(1)

class Actor(nn.Module):
  def __init__(self, hidden_size, n_nodes, p):
    super(Actor, self).__init__()
    self.encoder = Encoder(2, hidden_size)
    self.decoder = Decoder(hidden_size, p)
    self.big_number = 100000
    self.logsoft = nn.LogSoftmax().to(device)

    for p in self.parameters():
      if len(p.shape) > 1:
        nn.init.xavier_uniform_(p)

  def encode_graph(self, coords):
    return self.encoder(coords)
  
  def forward(self, emb_cur_loc, v_nodes, emb_graph, last_hh):
    logits, last_hh = self.decoder(emb_cur_loc, last_hh, emb_graph)
    logits[v_nodes == 1] = -self.big_number
    logprobs = self.logsoft(logits)
    probs = torch.exp(logprobs)

    if self.training:
      m = torch.distributions.Categorical(probs)
      action = m.sample()
      log_p = m.log_prob(action)
    else:
      prob, action = torch.max(probs, 1)
      log_p = prob.log()
    return action, log_p, last_hh

class AttentionCritic(nn.Module):
    def __init__(self, hidden_size, use_tahn=False, C = 10):
        super(AttentionCritic, self).__init__()
        self.use_tahn = use_tahn 
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True)).to(device)
        self.project_ref = nn.Conv1d(hidden_size, hidden_size, 1).to(device)
        self.project_query = nn.Linear(hidden_size, hidden_size).to(device)
        self.C = C

    def forward(self, static_hidden, decoder_hidden):
        batch_size, hidden_size, n_nodes = static_hidden.size()
        e = self.project_ref(static_hidden)
        decoder_hidden = self.project_query(decoder_hidden)
        v = self.v.expand(batch_size, 1, hidden_size)
        q = decoder_hidden.view(batch_size, hidden_size, 1).expand(batch_size, hidden_size, n_nodes)
        u = torch.bmm(v, torch.tanh(e + q)).squeeze(1)

        if self.use_tahn:
            logits = self.C * self.tanh(u)
        else:
            logits = u 
            
        return e, logits 

class Critic(nn.Module):
  def __init__(self, hidden_size, n_processes=3, num_layers=1):
    super(Critic, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.static_encoder = Encoder(2, hidden_size)
    self.attention1 = AttentionCritic(hidden_size)
    self.attention2 = AttentionCritic(hidden_size)
    self.attention3 = AttentionCritic(hidden_size)
    self.fc1 = nn.Linear(hidden_size, hidden_size).to(device)
    self.fc2 = nn.Linear(hidden_size, 1).to(device)

    for p in self.parameters():
      if len(p.shape) > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, static):
    static_hidden = self.static_encoder(static)
    batch_size, _, n_nodes = static_hidden.size()
    hx = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    hy = hx.squeeze(0)

    e, logits = self.attention1(static_hidden, hy)
    probs = torch.softmax(logits, dim=1)

    hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)
    e, logits = self.attention2(static_hidden, hy)
    probs = torch.softmax(logits, dim=1)

    hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)
    e, logits = self.attention3(static_hidden, hy)
    probs = torch.softmax(logits, dim=1)
    hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)

    out = F.relu(self.fc1(hy))
    out = self.fc2(out)
    
    return out

class Agent:
  def __init__(self, seed, batch_size, hidden_size, n_nodes, p):
    super(Agent, self).__init__()
    torch.manual_seed(seed)
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.n_nodes = n_nodes
    self.data_gen = DataGenerator(n_nodes)
    self.env = Environment(batch_size, n_nodes)
    self.actor = Actor(hidden_size, n_nodes, p)
    self.critic = Critic(hidden_size)

  def reward(self, actions):
    sample = torch.stack(actions, 0)
    tilted = torch.cat((sample[-1].unsqueeze(0), sample[:-1]), 0)
    R = ((tilted-sample).pow(2)).sum(dim=(2, 3)).pow(0.5).sum(dim=0)
    return R 

  def train(self, n_epochs, n_steps, batch_size, n_nodes):
    actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
    critic_optim = optim.Adam(self.critic.parameters(), lr=1e-4)
    self.actor.train()
    self.critic.train()
    R_list = []
    xs = []
    for n_e in range(1, n_epochs + 1):
      data = self.data_gen.get_train_data(batch_size)
      cur_loc, v_nodes = self.env.reset(data)
      a = (torch.ones(self.batch_size, 1).long() * (self.n_nodes - 1)).to(device)
      embed_graph = self.actor.encode_graph(data.permute(0, 2, 1))
      h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
      c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
      last_hh = (h_0, c_0)
      log_list = []
      actions = []

      for n_s in range(n_steps):
        emb_cur_loc = torch.gather(embed_graph, 2, a.view(-1, 1, 1).expand(batch_size, self.hidden_size, 1)).detach()
        a, log_p, last_hh = self.actor(emb_cur_loc.transpose(2, 1), v_nodes.detach(), embed_graph, last_hh)
        cur_loc, v_nodes = self.env.step(a.clone().detach())
        log_list.append(log_p.unsqueeze(1))
        action = torch.gather(data, 1, a.view(-1, 1).repeat(1, 2).view(-1, 1, 2))
        actions.append(action.detach())

      log_p = torch.cat(log_list, dim=1)
      D = self.reward(actions)
      V = self.critic(data.permute(0, 2, 1)).view(batch_size)
      
      actor_loss = torch.mean((D - V).detach() * log_p.sum(dim=1))
      critic_loss = torch.mean(torch.pow((D - V), 2))
      
      actor_optim.zero_grad()
      actor_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2)
      actor_optim.step()

      critic_optim.zero_grad()
      critic_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2)
      critic_optim.step()

      if n_e % 200 == 0:
        xs.append(n_e)
        R, actions = self.test(batch_size, n_steps)
        R_mean = torch.mean(R)
        R_list.append(R_mean)
        print(R_mean)
    torch.save(torch.stack(R_list), '/home/shrek/Desktop/task6_rewards.pt')
    np.savetxt('task6_rewards.txt', R_list)


  def test(self, batch_size, n_steps):
    self.actor.eval()
    data = self.data_gen.get_test_data(batch_size)
    cur_loc, v_nodes = self.env.reset(data)
    a = (torch.ones(self.batch_size, 1).long() * (self.n_nodes - 1)).to(device)
    h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
    c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
    last_hh = (h_0, c_0)
    actions = []
    with torch.no_grad():
      emb_graph = self.actor.encode_graph(data.permute(0, 2, 1))
      for n_s in range(n_steps):
        emb_cur_loc = torch.gather(emb_graph, 2, a.view(-1, 1, 1).expand(self.batch_size, self.hidden_size, 1)).detach()
        a, log_p, last_hh = self.actor(emb_cur_loc.transpose(2, 1), v_nodes, emb_graph, last_hh)
        cur_loc, v_nodes = self.env.step(a.clone().detach())
        action = torch.gather(data, 1, a.view(-1, 1).repeat(1, 2).view(-1, 1, 2))
        actions.append(action.detach())
    D = self.reward(actions)
    self.actor.train()
    return D, actions

agent = Agent(5, 128, 128, 10, .1)
agent.train(1000000000, 10, 128, 10)
