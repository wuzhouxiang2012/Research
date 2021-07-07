import torch
from torch import nn
class DQNPtorchModel(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DQNPtorchModel, self).__init__()
        hidden_unit = 128
        self.fc1 = nn.Linear(state_dim,hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.fc3 = nn.Linear(hidden_unit, act_dim)
    def forward(self,x):
        y_predicted = self.fc3(nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x)))))
        return y_predicted