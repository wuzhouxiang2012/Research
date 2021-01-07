import torch
from torch import nn

class PGTorchModel(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(PGTorchModel, self).__init__()
        self.act_dim = act_dim
        self.state_dim = state_dim
        hid1_size = 10*act_dim
        # self.fc3 = layers.fc(size=hid1_size, act='tanh')
        # self.fc1 = layers.fc(size=hid1_size, act='tanh')
        # self.fc2 = layers.fc(size=act_dim, act='softmax')
        self.linear1 = nn.Linear(state_dim, hid1_size)
        self.linear2 = nn.Linear(hid1_size, act_dim)

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        # obs = self.fc3(obs)
        # out = self.fc1(obs)
        # out = self.fc2(out)
        # return out
        out = self.linear1(obs)
        out = nn.functional.tanh(out)
        out = self.linear2(out)
        out = nn.functional.softmax(out)
        return out