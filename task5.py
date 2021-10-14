import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(5)

class DataGenerator:
  def __init__(self, n_nodes):
    self.n_nodes = n_nodes

  def get_train_data(self, batch_size):
    return torch.rand([batch_size, self.n_nodes, 2]).to(device)

  def get_test_data(self, test_size):
    return torch.load('/home/shrek/Desktop/test_data.pt', map_location=torch.device(device))

class Environment:
  def __init__(self, batch_size, n_nodes):
    self.batch_size = batch_size
    self.n_nodes = n_nodes

  def reset(self, A):
    self.D = torch.cdist(A, A)
    self.cur_loc = torch.randint(self.n_nodes - 1, self.n_nodes, (self.batch_size,)).to(device)
    self.r_total = torch.zeros(self.batch_size).to(device)
    self.v_nodes = torch.zeros([self.batch_size, self.n_nodes], dtype=torch.int32).to(device)
    return self.cur_loc, self.v_nodes

  def step(self, a):
      last_node = self.cur_loc
      self.cur_loc = a
      self.v_nodes[torch.arange(self.batch_size), self.cur_loc] = 1
      self.r_total += self.D[torch.arange(self.batch_size), self.cur_loc, last_node]
      return self.cur_loc, self.v_nodes

class Encoder(nn.Module):
  def __init__(self, in_dim, embed_size):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv1d(in_dim, embed_size, 1).to(device)

  def forward(self, coords):
    coords = self.conv1(coords.reshape(-1, 2, coords.shape[1]))
    return coords

class Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, p, batch_size):
    super(Decoder, self).__init__()
    self.LSTM = nn.LSTM(embed_size, hidden_size, bias=False, batch_first=True, bidirectional=False, dropout=0).to(device)
    self.Dropouth = nn.Dropout(p).to(device)
    self.Dropoutc = nn.Dropout(p).to(device)

  def forward(self, decoder_input, last_hh):
    out, t = self.LSTM(decoder_input, last_hh)
    out = self.Dropoutc(out)
    h_0 = self.Dropouth(t[0])
    c_0 = self.Dropouth(t[1])
    last_hh = (h_0, c_0)
    return h_0[0].squeeze(0), last_hh

class Attention(nn.Module):
  def __init__(self, hidden_size, n_nodes):
    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.proj_hidden = nn.Linear(hidden_size, hidden_size).to(device)
    self.proj_emb_graph = nn.Linear(hidden_size, hidden_size).to(device)
    self.tanh = nn.Tanh().to(device)
    self.V = nn.parameter.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True)).to(device)

  def forward(self, emb_graph, hidden_decoder):
    proj_emb_graph = self.proj_emb_graph(emb_graph.transpose(2, 1))
    proj_hidden = self.proj_hidden(hidden_decoder)
    proj_emb_graph = proj_emb_graph.transpose(2, 1)
    proj_hidden = proj_hidden.view(emb_graph.shape[0], -1, 1).expand(emb_graph.shape[0], self.hidden_size, emb_graph.shape[2])
    res = proj_emb_graph + proj_hidden
    tanh = self.tanh(res)
    U = torch.bmm(self.V.expand(emb_graph.shape[0], 1, self.hidden_size), tanh)
    return U

class Actor(nn.Module):
  def __init__(self, batch_size, embed_size, hidden_size, n_nodes, p):
    super(Actor, self).__init__()
    self.Encoder = Encoder(2, embed_size)
    self.Decoder = Decoder(embed_size, hidden_size, p, batch_size)
    self.Attention = Attention(hidden_size, n_nodes)
    self.logsoft = nn.LogSoftmax().to(device)

  def encode_graph(self, coords):
    return self.Encoder(coords)

  def forward(self, emb_cur_loc, v_nodes, emb_graph, last_hh):
    hidden_decoder, last_hh = self.Decoder(emb_cur_loc, last_hh)
    logits = self.Attention(emb_graph, hidden_decoder).reshape(v_nodes.shape)
    logits[v_nodes == 1] -= 100000
    logprobs = self.logsoft(logits)
    probs = torch.exp(logprobs)

    if self.training:
      a = torch.distributions.Categorical(probs)
      action = a.sample()
      log_p = a.log_prob(action)
    else:
      prob, action = torch.max(probs, 1)
      log_p = 0
    return action, log_p, last_hh

class AttentionCritic(nn.Module):
    def __init__(self, hidden_size, use_tahn=False, C = 10):
        super(AttentionCritic, self).__init__()
        self.use_tahn = use_tahn 
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True)).to(device)
        self.project_ref = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1).to(device)
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
    hy = torch.matmul(probs.unsqueeze(1), e.permute(0,2,1)).squeeze(1)
    e, logits = self.attention3(static_hidden, hy)
    probs = torch.softmax(logits, dim=1)
    hy = torch.matmul(probs.unsqueeze(1), e.permute(0, 2, 1)).squeeze(1)
    out = F.relu(self.fc1(hy))
    out = self.fc2(out)
    return out 

class Agent(nn.Module):
  def __init__(self, batch_size, embed_size, hidden_size, n_nodes, p):
    super(Agent, self).__init__()
    self.embed_size = embed_size
    self.data_gen = DataGenerator(n_nodes)
    self.env = Environment(batch_size, n_nodes)
    self.actor = Actor(batch_size, embed_size, hidden_size, n_nodes, p)
    self.critic = Critic(hidden_size)
    self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
    self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.0001)

  def test(self, batch_size, n_steps):
    self.actor.eval()
    coords = self.data_gen.get_test_data(batch_size)
    h_0 = torch.zeros(1, batch_size, self.embed_size).to(device)
    c_0 = torch.zeros(1, batch_size, self.embed_size).to(device)
    last_hh = (h_0, c_0)
    with torch.no_grad():
      cur_loc, v_nodes = self.env.reset(coords)
      embed_graph = self.actor.encode_graph(coords)
      for n_s in range(n_steps):
        emb_cur_loc = torch.gather(embed_graph, 2, cur_loc.view(-1, 1, 1).expand(batch_size, self.embed_size, 1)).detach()
        a, log_p, last_hh = self.actor(emb_cur_loc.transpose(2, 1), v_nodes, embed_graph, last_hh)
        cur_loc, v_nodes = self.env.step(a.detach())
      D = self.env.r_total.detach()
      self.actor.train()
    return D

  def forward(self, n_epochs, n_steps, batch_size, n_nodes):
    self.actor.train()
    R_list = []
    xs = []
    for n_e in range(1, n_epochs + 1):
      self.actor_optim.zero_grad()
      self.critic_optim.zero_grad()
      coords = self.data_gen.get_train_data(batch_size)
      cur_loc, v_nodes = self.env.reset(coords)
      embed_graph = self.actor.encode_graph(coords)
      h_0 = torch.zeros(1, batch_size, self.embed_size).to(device)
      c_0 = torch.zeros(1, batch_size, self.embed_size).to(device)
      last_hh = (h_0, c_0)
      log_list = []

      for n_s in range(n_steps):
        emb_cur_loc = torch.gather(embed_graph, 2, cur_loc.view(-1, 1, 1).expand(batch_size, self.embed_size, 1))
        a, log_p, last_hh = self.actor(emb_cur_loc.transpose(2, 1), v_nodes.detach(), embed_graph, last_hh)
        cur_loc, v_nodes = self.env.step(a.detach())
        log_list.append(log_p.unsqueeze(1))
      log_p = torch.cat(log_list, dim=1)
      D = self.env.r_total.detach()
      V = self.critic(coords).view(batch_size)
      actorLoss = torch.mean(log_p.sum(dim=1) * (D - V).detach())
      criticLoss = torch.mean(torch.pow((D - V), 2))

      if n_e % 200 == 0:
        xs.append(n_e)
        R = self.test(batch_size, n_steps)
        R_list.append(torch.mean(R))
        fig, ax = plt.subplots()
        ax.plot(xs, R_list)
        fig.savefig(f'/home/shrek/Desktop/imgs/{n_e}')
      torch.autograd.set_detect_anomaly(True)
      actorLoss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 2)
      self.actor_optim.step()
      torch.autograd.set_detect_anomaly(True)
      criticLoss.backward()
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2)
      self.critic_optim.step()

agent = Agent(128, 128, 128, 10, .1)
agent(1000000000, 10, 128, 10)
