from torch import nn
import torch
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, encoder, obs_size, action_size):
        super().__init__()
        hidden_size1 = 256
        hidden_size2 = 128
        self.encoder = encoder
        self.fc1 = nn.Linear(obs_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size2)
        self.fc5 = nn.Linear(hidden_size2, action_size)

    def forward(self, batch_obs):
        obs_list = []
        for obs in batch_obs:
            obs1 = torch.from_numpy(obs[0])
            obs2_list = []
            remain_req_edges = obs[1]
            for remain_req_edge in remain_req_edges:
                remain_req_edge = torch.from_numpy(remain_req_edge)
                obs2_list.append(self.encoder(remain_req_edge).view(-1,))
            obs2 = torch.cat(obs2_list, dim=0)
            obs = torch.cat([obs1, obs2], dim=0)
            obs.unsqueeze_(0)
            obs_list.append(obs)
        obs = torch.cat(obs_list, axis=0)
        out = self.fc1(obs)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        out = self.fc3(out)
        out = torch.tanh(out)
        out = self.fc4(out) 
        out = torch.tanh(out)
        out = self.fc5(out)
        out = F.softmax(out, dim=1)
        return out