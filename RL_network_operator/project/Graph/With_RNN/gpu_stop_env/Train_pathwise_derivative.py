import collections
import random
import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F
from util import evaluate_totally_random, evaluate, evaluate_reject_when_full
from env_generator import produce_env
from encoder import Encoder
from actor import Actor
from critic import Critic

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
                 actor,
                 critic,
                 encoder,
                 obs_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 encoder_lr,
                 gamma=0.9,
                 alpha = 0.9,
                 update_target_steps=200):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.global_step = 0
        self.gamma = gamma
        self.alpha = alpha
        # every 200 training steps, coppy model param into target_model
        self.update_target_steps = update_target_steps  
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.otpimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
        self.criteria_critic = nn.MSELoss()
    
    def sample(self, obs):
        # obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        # choose action based on prob
        with torch.no_grad():
            obs1 = torch.from_numpy(obs[0])
            obs2_list = []
            remain_req_edges = obs[1]
            for remain_req_edge in remain_req_edges:
                remain_req_edge = torch.from_numpy(remain_req_edge)
                obs2_list.append(self.target_encoder(remain_req_edge).view(-1,))
            obs2 = torch.cat(obs2_list, dim=0)
            obs = torch.cat([obs1, obs2], dim=0)
            obs.unsqueeze_(0)
            obs = obs.to(device)
            action = np.random.choice(range(self.action_dim), p=self.target_actor(obs).to('cpu').numpy().reshape(-1,))  
            return action

    def predict(self, obs):  # choose best action
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
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def learn(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal):
        # update target model
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        self.global_step %= 200

        # train critic 
        # predict q
        pred_value = (self.critic(batch_obs)*F.one_hot(batch_action, num_classes=self.action_dim)).sum(dim=1).view(-1,1)
        with torch.no_grad(): 
            batch_target_action = self.target_actor(batch_next_obs).argmax(dim=1)
            batch_one_hot_target_action = F.one_hot(batch_target_action, num_classes=self.action_dim)
            batch_next_q = (batch_one_hot_target_action*self.target_critic(batch_next_obs)).sum(dim=1).view(-1,1)
            target_value = batch_reward + (1 - batch_terminal.view(-1,1))*batch_next_q*self.gamma
            target_value = (target_value-pred_value)*self.alpha + pred_value
        loss_critic = self.criteria_critic(pred_value, target_value)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # train actor
        # find action maximize critic
        pred_action = self.actor(batch_obs)
        with torch.no_grad():
            true_one_hot_action = F.one_hot(self.critic(batch_obs).argmax(dim=1),num_classes=self.action_dim)
        loss_actor = -1.0*(torch.log(pred_action)*true_one_hot_action).sum(dim=1).mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
    def save(self, actor_path, critic_path, encoder_path):
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.encoder.state_dict(), encoder_path)
    
    def load(self, actor_path, critic_path, encoder_path):
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.load_state_dict(torch.load(actor_path))
        self.encoder.load_state_dict(torch.load(encoder_path))

        
    def learn(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal):
        # update target model
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        self.global_step %= 200

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

        # train critic 
        # predict q
        pred_value = (self.critic(batch_obs)*F.one_hot(batch_action, num_classes=self.action_dim)).sum(dim=1).view(-1,1)
        with torch.no_grad(): 
            batch_target_action = self.target_actor(batch_next_obs).argmax(dim=1)
            batch_one_hot_target_action = F.one_hot(batch_target_action, num_classes=self.action_dim)
            batch_next_q = (batch_one_hot_target_action*self.target_critic(batch_next_obs)).sum(dim=1).view(-1,1)
            target_value = batch_reward + (1 - batch_terminal.view(-1,1))*batch_next_q*self.gamma
            target_value = (target_value-pred_value)*self.alpha + pred_value
        loss_critic = self.criteria_critic(pred_value, target_value)

        # train actor
        # find action maximize critic
        pred_action = self.actor(batch_obs)
        with torch.no_grad():
            true_one_hot_action = F.one_hot(self.critic(batch_obs).argmax(dim=1),num_classes=self.action_dim)
        loss_actor = -1.0*(torch.log(pred_action)*true_one_hot_action).sum(dim=1).mean()
        
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        self.otpimizer_encoder.zero_grad()
        loss_critic.backward(retain_graph=True)
        loss_actor.backward(retain_graph=True)
        self.optimizer_critic.step()
        self.otpimizer_encoder.step()
        self.optimizer_actor.step()
        # self.otpimizer_encoder.zero_grad()
        # self.otpimizer_encoder.step()



        
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



def train(show_baseline=False, continue_train=False, continue_mimic=False,\
    load_critic_path='best_pd_critic', load_encoder_path='best_pd_encoder', \
    load_actor_path='best_pd_actor', save_critic_path='best_pd_critic' , \
    save_encoder_path='best_pd_encoder', save_actor_path='best_pd_actor', \
    mimic_actor_path = 'mimic_actor', mimic_encoder_path = 'mimic_encoder',\
    learn_freq= 5, memory_size = 20000, total_time=600,\
    memory_warmup_size = 2000, batch_size = 32, learning_rate = 0.001, \
    gamma = 0.9, alpha = 0.9, max_episode=1000, update_target_steps=200, \
    evaluate_env_list_path = 'env_list_set2',\):
    
    
    if show_baseline:
        print(evaluate_reject_when_full(evaluate_env_list_path))
        print(evaluate_totally_random(evaluate_env_list_path))
    env = produce_env(total_time=total_time)
    action_dim = 4  
    obs_dim_1 = 45  
    request_dim = 17
    obs_dim_2 = 10
    obs_dim = obs_dim_1+obs_dim_2*7
    rpm = ReplayMemory(memory_size)  # DQN的经验回放池
    encoder = Encoder(input_size=request_dim, output_size=obs_dim_2, \
        use_rnn=False, use_gru=True, use_lstm=False)
    actor = Actor(obs_size=obs_dim, action_size=action_dim)
    actor.to(device=device)
    critic = Critic(obs_dim=obs_dim, action_dim=action_dim)
    critic.to(device=device)
    if continue_mimic:
        actor.load_state_dict(torch.load(mimic_actor_path))
        encoder.load_state_dict(torch.load(mimic_encoder_path))
    agent = Agent(
        critic=critic,
        actor = actor,
        encoder = encoder,
        obs_dim = obs_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        alpha=alpha,
        update_target_steps=update_target_steps)

    if continue_train:
        agent.load(actor_path=load_actor_path, \
            critic_path=load_critic_path, encoder_path=load_encoder_path)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < memory_warmup_size:
        run_episode(env, agent, rpm, memory_warmup_size, learn_freq, batch_size)


    # start train
    episode = 0
    while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
        # train part
        for i in range(0, 10):
            total_reward = run_episode(env, agent, rpm, memory_warmup_size, learn_freq, batch_size)
            episode += 1

        # test part
        eval_reward= evaluate(evaluate_env_list_path, agent, render=False)
        print('episode:{}  Test reward:{}'.format(episode, eval_reward))
        agent.save(actor_path=save_actor_path, \
            critic_path=save_critic_path, encoder_path=save_encoder_path)
if __name__ == '__main__':
    train(show_baseline=True, continue_train=False, continue_mimic=False,\
    load_critic_path='best_pd_critic', load_encoder_path='best_pd_encoder', \
    load_actor_path='best_pd_actor', save_critic_path='best_pd_critic' , \
    save_encoder_path='best_pd_encoder', save_actor_path='best_pd_actor', \
    mimic_actor_path = 'mimic_actor', mimic_encoder_path = 'mimic_encoder',\
    learn_freq= 5, memory_size = 20000, total_time=40,\
    memory_warmup_size = 2000, batch_size = 32, learning_rate = 0.001, \
    gamma = 0.9, alpha = 0.9, max_episode=1000, update_target_steps=200)