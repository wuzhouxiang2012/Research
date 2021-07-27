import collections
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy
import random
from Util import evaluate_reject_when_full, evaluate, evaluate_totally_random
from Env_generator import produce_env
from critic import Critic
from encoder import Encoder

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

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

        return  obs_batch, \
                torch.from_numpy(np.array(action_batch)), \
                torch.from_numpy(np.array(reward_batch).astype('float32')).view(-1,1),\
                next_obs_batch,\
                torch.from_numpy(np.array(done_batch).astype('float32'))

    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(self,
                 critic,
                 encoder,
                 obs_dim,
                 action_dim,
                 critic_lr,
                 encoder_lr,
                 gamma,
                 alpha,
                 update_target_steps=200):
        self.e_greed = 0.1
        self.e_greed_decrement = 1e-6
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.gamma = gamma
        self.alpha = alpha
        self.global_step = 0
        self.update_target_steps = update_target_steps  
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
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
        with torch.no_grad():
            obs1 = torch.from_numpy(obs[0])
            obs2_list = []
            remain_req_edges = obs[1]
            for remain_req_edge in remain_req_edges:
                remain_req_edge = torch.from_numpy(remain_req_edge)
                obs2_list.append(self.encoder(remain_req_edge).view(-1,))
            obs2 = torch.cat(obs2_list, dim=0)
            obs = torch.cat([obs1, obs2], dim=0)
            obs.unsqueeze_(0)
            obs = obs.to(device)
            return self.critic(obs).argmax(dim=1).item()

    def sync_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def learn(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal):
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
        batch_obs = torch.cat(obs_list, axis=0).to(device)

        with torch.no_grad():
            obs_list = []
            for obs in batch_next_obs:
                obs1 = torch.from_numpy(obs[0])
                obs2_list = []
                remain_req_edges = obs[1]
                for remain_req_edge in remain_req_edges:
                    remain_req_edge = torch.from_numpy(remain_req_edge)
                    obs2_list.append(self.target_encoder(remain_req_edge).view(-1,))
                obs2 = torch.cat(obs2_list, dim=0)
                obs = torch.cat([obs1, obs2], dim=0)
                obs.unsqueeze_(0)
                obs_list.append(obs)
            batch_next_obs = torch.cat(obs_list, axis=0).to(device)

        batch_action = batch_action.to(device)
        batch_reward = batch_reward.to(device)
        batch_terminal = batch_terminal.to(device)

        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        # train critic 
        pred_value = (self.critic(batch_obs)*\
            F.one_hot(batch_action, num_classes=self.action_dim))\
                .sum(dim=1).view(-1,1)  # 获取Q预测值
        
        # with torch.no_grad(): 
        #     max_q = (self.target_critic(batch_next_obs).max(dim=1)[0].view(-1,1))
        #     target_value = batch_reward + (1 - batch_terminal.view(-1,1))*max_q
        # double dqn
        with torch.no_grad(): 
            # argmax action of critic(s_{t+1})
            max_action = self.critic(batch_next_obs).argmax(dim=1)
            one_hot_max_action = F.one_hot(max_action, num_classes=self.action_dim)
            target_q = (self.target_critic(batch_next_obs)*one_hot_max_action).sum(dim=1, keepdim=True)
            target_value = batch_reward + (1 - batch_terminal.view(-1,1))*target_q*self.gamma
            target_value = (target_value-pred_value)*self.alpha + pred_value
        loss_critic = self.criteria_critic(pred_value, target_value)
        self.optimizer_critic.zero_grad()
        self.optimizer_encoder.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()
        self.optimizer_encoder.step()
    def save(self, critic_path, encoder_path):
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.encoder.state_dict(), encoder_path)
    
    def load(self, critic_path, encoder_path):
        self.critic.load_state_dict(torch.load(critic_path))
        self.encoder.load_state_dict(torch.load(encoder_path))


        
def run_episode(env, agent, rpm, memory_warmup_size, learn_freq, batch_size):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > memory_warmup_size) and (step % learn_freq == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,batch_done) = rpm.sample(batch_size)
            # s,a,r,s',done
            agent.learn(batch_obs, batch_action, batch_reward,batch_next_obs,batch_done)  
        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward




def train(show_baseline=False, continue_train=False, \
    learn_freq= 5, memory_size = 20000, total_time=20,\
    memory_warmup_size = 2000, batch_size = 32, critic_lr = 0.001, \
    encoder_lr=0.0001, gamma = 0.9, alpha = 0.9, max_episode=1000, \
    critic_path='dqn_critic', encoder_path='dqn_encoder'):
    
    evaluate_env_list_path = 'env_list_set1'
    if show_baseline:
        print(evaluate_reject_when_full(evaluate_env_list_path))
        print(evaluate_totally_random(evaluate_env_list_path))
    env = produce_env(total_time=total_time)
    action_dim = 4  
    obs_dim_1 = 45  
    request_dim = 17
    obs_dim_2 = 10
    obs_dim = obs_dim_1+obs_dim_2*7
    encoder = Encoder(input_size=request_dim, output_size=obs_dim_2, \
        use_rnn=False, use_gru=True, use_lstm=False)
    rpm = ReplayMemory(memory_size)  # DQN的经验回放池
    critic = Critic(obs_dim=obs_dim, action_dim=action_dim)
    critic.to(device)
    agent = Agent(
        critic=critic,
        encoder=encoder,
        obs_dim = obs_dim,
        action_dim=action_dim,
        critic_lr=critic_lr,
        encoder_lr=encoder_lr,
        gamma=gamma,
        alpha=alpha)

    if continue_train:
        agent.load(critic_path=critic_path, encoder_path=encoder_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < memory_warmup_size:
        run_episode(env, agent, rpm, memory_warmup_size, learn_freq, batch_size)


    # start train
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 100):
            total_reward = run_episode(env, agent, rpm, memory_warmup_size, learn_freq, batch_size)
            episode += 1
        eval_reward= evaluate(evaluate_env_list_path, agent, render=False)
        print('episode:{}  Test reward:{}'.format(episode, eval_reward))
    agent.save(critic_path=critic_path, encoder_path=encoder_path)

if __name__ == '__main__':
    train(show_baseline=False, continue_train=False, \
    learn_freq= 5, memory_size = 20000, total_time=20,\
    memory_warmup_size = 2000, batch_size = 32, critic_lr = 0.001, \
    encoder_lr=0.0001, gamma = 0.9, alpha = 0.9, max_episode=1000, \
    critic_path='dqn_critic', encoder_path='dqn_encoder')