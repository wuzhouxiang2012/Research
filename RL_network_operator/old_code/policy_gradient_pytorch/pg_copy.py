import sys
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch 
from torch import nn
import torch.nn.functional as F
sys.path.append("/Users/Bob/Documents/GitHub/Research/RL_network_operator/project")

from network_env import Environment
from evaluate import Evaluation
from distribution import Distribution
from request import Request
from utils import calc_advantage, run_episode

LEARNING_RATE = 1e-3


# 训练一个episode
def run_episode(env, model):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = torch.from_numpy(obs).view(1,-1)
        obs_list.append(obs)
        # choose action based on prob
        action = np.random.choice(range(2), p=model(obs).detach().numpy().reshape(-1,))  
        action_list.append(action)

        # obs, reward, done, info = env.step(action)
        obs, reward, done = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list

def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)

def main():
     # create environment
    dist1 = Distribution(id=0, vals=[2], probs=[1])
    dist2 = Distribution(id=1, vals=[5], probs=[1])
    dist3 = Distribution(id=2, vals=[2,8], probs=[0.5,0.5])

    env = Environment(total_bandwidth = 10,\
        distribution_list=[dist1,dist2,dist3], \
        mu_list=[1,2,3], lambda_list=[3,2,1],\
        num_of_each_type_distribution_list=[300,300,300])
    evaluation = Evaluation()
    obs_dim = 6
    act_dim = 2
    # logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
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
            out = F.softmax(out)
            return out

    model = PPO(obs_size=obs_dim, action_size=act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, model)
        
        if i % 10 == 0:
            # logger.info("Episode {}, Reward Sum {}.".format(
            #     i, sum(reward_list)))
            print("Episode {}, Reward Sum {}.".format(i, sum(reward_list)/len(reward_list)))
        # print(obs_list)
        
        batch_obs = torch.cat(obs_list, 0)
        batch_action = torch.from_numpy(np.array(action_list))
        batch_adv = torch.tensor(calc_advantage(reward_list, gamma=0.9))

        pred_action_distribution = model(batch_obs)
        true_action_distribution = nn.functional.one_hot(batch_action)
        pred_choosed_action_log_prob = (torch.log(pred_action_distribution)*true_action_distribution).sum(1)
        loss = -1.0 *(pred_choosed_action_log_prob*batch_adv).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            avg_reward, avg_acc_rate = evaluation.evaluate_model(model)
            print('avg_reward', avg_reward, 'avg_acc_rate', avg_acc_rate, 'base ', evaluation.reject_when_full_avg_reward)


if __name__ == '__main__':
    main()