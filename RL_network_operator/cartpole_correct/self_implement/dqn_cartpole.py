import sys
import os
import collections
import gym
import random
import torch
from torch import nn
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 32  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0005  # 学习率
GAMMA = 0.9 # reward 的衰减因子，一般取 0.9 到 0.999 不等

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return  torch.from_numpy(np.array(obs_batch).astype('float32')), \
                torch.from_numpy(np.array(action_batch)), \
                torch.from_numpy(np.array(reward_batch).astype('float32')).view(-1,1),\
                torch.from_numpy(np.array(next_obs_batch).astype('float32')), \
                torch.from_numpy(np.array(done_batch).astype('float32'))

    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(self,
                 critic,
                 obs_dim,
                 action_dim,
                 lr = LEARNING_RATE):
        self.e_greed = 0.1
        self.e_greed_decrement = 1e-6
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.global_step = 0
        # 每隔200个training steps再把model的参数复制到target_model中    
        self.update_target_steps = 200  
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)
        self.criteria_critic = nn.MSELoss()
    
    def sample(self, obs):
        sample = np.random.rand()
        if sample < self.e_greed:
            act = np.random.randint(self.action_dim)
        else:
            act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act
    def predict(self, obs):  # 选择最优动作
        obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        with torch.no_grad():
            return self.critic(obs).argmax(dim=1).item()

    def sync_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
    

    def learn(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1

        # train critic 
        pred_value = (self.critic(batch_obs)*F.one_hot(batch_action)).sum(dim=1).view(-1,1)  # 获取Q预测值
        
        with torch.no_grad(): 
            max_q = (self.target_critic(batch_next_obs).max(dim=1)[0].view(-1,1))
            target_value = batch_reward + (1 - batch_terminal.view(-1,1))*max_q
        self.optimizer_critic.zero_grad()
        loss_critic = self.criteria_critic(pred_value, target_value)
        loss_critic.backward()
        self.optimizer_critic.step()



        
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,batch_done) = rpm.sample(BATCH_SIZE)
            # s,a,r,s',done
            agent.learn(batch_obs, batch_action, batch_reward,batch_next_obs,batch_done)  

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


class PDCritirc(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        hidden_size = 128
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
    def forward(self, batch_obs):
        out = self.fc1(batch_obs)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out


def main():
    env = gym.make(
        'CartPole-v0'
    )  # CartPole-v0: expected reward > 180                MountainCar-v0 : expected reward > -120
    action_dim = env.action_space.n  # CartPole-v0: 2
    obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)
    rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

    critic = PDCritirc(obs_dim=obs_dim, action_dim=action_dim)
    agent = Agent(
        critic=critic,
        obs_dim = obs_dim,
        action_dim=action_dim)


    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000

    # start train
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1

        # test part
        eval_reward= evaluate(env, agent, render=False)  # render=True 查看显示效果
        print('episode:{}  Test reward:{}'.format(episode, eval_reward))



if __name__ == '__main__':
    main()