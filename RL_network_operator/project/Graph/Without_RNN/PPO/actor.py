import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        hidden_size = 128
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, obs):
        out = self.fc1(obs)
        out = torch.tanh(out)
        out = self.fc3(out)
        out = torch.tanh(out)
        out = self.fc4(out)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out