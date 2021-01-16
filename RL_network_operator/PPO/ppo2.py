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

obs_size = 6
action_size = 2
# experiment hyper parameter
# total number of experiments
num_iter = 1000
# for each experiment, simulate 100 trajectories.
num_episode = 10
# for each experiment, tuning num_epoch times
num_epoch = 10

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
        out = torch.tanh(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out



policy_PPO = PPO(obs_size=obs_size, action_size=action_size)
target_PPO = PPO(obs_size=obs_size, action_size=action_size)
target_PPO.load_state_dict(policy_PPO.state_dict())

optimizer = torch.optim.Adam(policy_PPO.parameters(), lr=0.01)
BETA = 1
EPSILON = 0.1


    
#2. Iteration 
for iter in range(num_iter):
    #2.1  Using theta k to interact with the env
    # to collect {s_t, a_t} and compute advantage
    # advantage(s_t, a_t) = sum_{t^prime=t}^{T_n}(r_{t^prime}^{n})
    
    all_obs = []
    all_action = []
    all_advantage = []
    for episode in range(num_episode):
        obs_list, action_list, reward_list = run_episode(env, target_PPO)
        # batch_obs = torch.from_numpy(np.array(obs_list))
        # batch_action = torch.from_numpy(np.array(action_list))
        # batch_reward = torch.from_numpy(calc_advantage(reward_list, gamma=0.9))
        advantage_list = calc_advantage(reward_list, gamma=0.9)
        all_obs.extend(obs_list)
        all_action.extend(action_list)
        all_advantage.extend(advantage_list)
    dataset = PPODataset(obs_list=all_obs, action_list=all_action, advantage_list=all_advantage)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # optimize theta
    
    for epoch in range(num_epoch):
        for i, (batch_obs, batch_action, batch_adv) in enumerate(dataloader):
            pred_action_distribution = policy_PPO(batch_obs)
            target_action_distribution = target_PPO(batch_obs).detach()
            true_action_distribution = nn.functional.one_hot(batch_action, num_classes=2)
            pred_choosed_action_prob = (pred_action_distribution*true_action_distribution).sum(1)
            target_choosed_action_prob = (target_action_distribution*true_action_distribution).sum(1)
            J = (pred_choosed_action_prob/target_choosed_action_prob*batch_adv).mean()
            J_clip = (torch.clamp(pred_choosed_action_prob/target_choosed_action_prob, 1-EPSILON, 1+EPSILON)*batch_adv).mean()
            # print(pred_choosed_action_prob/target_choosed_action_prob)
            loss = -1.0*torch.min(J, J_clip)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%10 ==0:
            reward, acc = evaluation.evaluate_model(policy_PPO)
            print(f'iter{iter}:Reward{reward:.4f}, ACC{acc:.4f}, base{evaluation.reject_when_full_avg_reward:.4f},{evaluation.reject_when_full_avg_acc_rate:.4f}')
    target_PPO.load_state_dict(policy_PPO.state_dict())



            
        





