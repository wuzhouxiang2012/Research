import sys
import os
import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
sys.path.append("/Users/Bob/Documents/GitHub/Research/RL_network_operator/project")

from network_env import Environment
from evaluate import Evaluation
from distribution import Distribution
from request import Request
from utils import calc_advantage, run_episode

OBS_SIZE = 6
ACTION_SIZE = 2
# experiment hyper parameter
# total number of experiments
NUM_ITER = 1000
# for each experiment, simulate 100 trajectories.
NUM_EPISODE = 1
# for each experiment, tuning num_epoch times
NUM_EPOCH = 1
BATCH_SIZE = 899
BETA = 1
EPSILON = 0.1

 # create environment
dist1 = Distribution(id=0, vals=[2], probs=[1])
dist2 = Distribution(id=1, vals=[5], probs=[1])
dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

env = Environment(total_bandwidth = 10,\
    distribution_list=[dist1,dist2,dist3], \
    mu_list=[1,2,3], lambda_list=[3,2,1],\
    num_of_each_type_distribution_list=[300,300,300])
evaluation = Evaluation()

class PPODataset(Dataset):
    def __init__(self, obs_list, action_list, advantage_list):
        self.obs_list = torch.cat(obs_list, 0)
        self.action_list = torch.tensor(action_list, dtype=torch.int64)
        self.advantage_list = torch.tensor(advantage_list, dtype=torch.float32)

    def __getitem__(self, index):
        return self.obs_list[index,:], self.action_list[index], self.advantage_list[index]
    
    def __len__(self):
        return self.obs_list.shape[0]

#1. Initialize network
class PPO(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        hidden_size = action_size * 10
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, obs):
        out = self.fc1(obs)
        out = F.tanh(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out



policy_PPO = PPO(obs_size=OBS_SIZE, action_size=ACTION_SIZE)


optimizer = torch.optim.Adam(policy_PPO.parameters(), lr=0.01)



    
#2. Iteration 
for iter in range(NUM_ITER):
    #2.1  Using theta k to interact with the env
    # to collect {s_t, a_t} and compute advantage
    # advantage(s_t, a_t) = sum_{t^prime=t}^{T_n}(r_{t^prime}^{n})
    
    all_obs = []
    all_action = []
    all_advantage = []
    for episode in range(NUM_EPISODE):
        obs_list, action_list, reward_list = run_episode(env, policy_PPO)
        # batch_obs = torch.from_numpy(np.array(obs_list))
        # batch_action = torch.from_numpy(np.array(action_list))
        # batch_reward = torch.from_numpy(calc_advantage(reward_list, gamma=0.9))
        advantage_list = calc_advantage(reward_list, gamma=0.9)
        all_obs.extend(obs_list)
        all_action.extend(action_list)
        all_advantage.extend(advantage_list)
    dataset = PPODataset(obs_list=all_obs, action_list=all_action, advantage_list=all_advantage)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # optimize theta
    
    for epoch in range(NUM_EPOCH):
        for i, (batch_obs, batch_action, batch_adv) in enumerate(dataloader):
            # print(batch_obs.shape)
            pred_action_distribution = policy_PPO(batch_obs)
            true_action_distribution = nn.functional.one_hot(batch_action, num_classes=2)
            pred_choosed_action_log_prob = (torch.log(pred_action_distribution)*true_action_distribution).sum(1)
            loss = -1.0*(pred_choosed_action_log_prob*batch_adv).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    if iter%100==0:
        reward, acc = evaluation.evaluate_model(policy_PPO)
        print(f'iter{iter}:Reward{reward}, ACC{acc}, base{evaluation.reject_when_full_avg_reward},{evaluation.reject_when_full_avg_acc_rate}')



    