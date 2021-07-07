from torch import nn
import torch
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, encoder, obs_size, action_size):
        super().__init__()
        hidden_size = 128
        self.encoder = encoder
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, obs):
        obs1 = torch.from_numpy(obs[0])
        obs2_list = []
        remain_req_edges = obs[1]
        for remain_req_edge in remain_req_edges:
            remain_req_edge = torch.from_numpy(remain_req_edge)
            obs2_list.append(self.encoder(remain_req_edge).view(-1,))
        obs2 = torch.cat(obs2_list, dim=0)
        obs = torch.cat([obs1, obs2], dim=0)
        obs.unsqueeze_(0)
        out = self.fc1(obs)
        out = torch.tanh(out)
        out = self.fc3(out)
        out = torch.tanh(out)
        out = self.fc4(out)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        # print(out)
        return out