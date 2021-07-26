import collections
import random
import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F
from Util import evaluate_totally_random, evaluate, evaluate_reject_when_full
from Env_generator import produce_env
from encoder import Encoder
from actor import Actor
from critic import Critic


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
                 obs_dim,
                 action_dim,
                 lr = 0.001,
                 gamma=0.9,
                 alpha = 0.9,
                 update_target_steps=200):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.global_step = 0
        self.gamma = gamma
        self.alpha = alpha
        # every 200 training steps, coppy model param into target_model
        self.update_target_steps = update_target_steps  
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)
        self.criteria_critic = nn.MSELoss()
    
    def sample(self, obs):
        obs = [obs]
        # choose action based on prob
        action = np.random.choice(range(self.action_dim), p=self.actor(obs).detach().numpy().reshape(-1,))  
        return action

    def predict(self, obs):  # choose best action
        obs = [obs]
        return self.actor(obs).argmax(dim=1).item()

    def sync_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

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
    def save(self, actor_path, critic_path):
        torch.save(self.critic.state_dict(), critic_path)
        torch.save(self.actor.state_dict(), actor_path)
    
    def load(self, actor_path, critic_path):
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.load_state_dict(torch.load(actor_path))

        
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
        loss_critic.backward()
        self.optimizer_critic.step()
        self.optimizer_critic.zero_grad()

        # train actor
        # find action maximize critic
        pred_action = self.actor(batch_obs)
        with torch.no_grad():
            true_one_hot_action = F.one_hot(self.critic(batch_obs).argmax(dim=1),num_classes=self.action_dim)
        loss_actor = -1.0*(torch.log(pred_action)*true_one_hot_action).sum(dim=1).mean()
        loss_actor.backward()
        self.optimizer_actor.step()
        self.optimizer_actor.zero_grad()



        
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



def train(show_baseline=False, continue_train=False, critic_path='best_pd_critic',\
    actor_path='best_pd_actor', learn_freq= 5, memory_size = 20000, \
    memory_warmup_size = 2000, batch_size = 32, learning_rate = 0.001, \
    gamma = 0.9, alpha = 0.9, max_episode=1000, update_target_steps=200):
    
    evaluate_env_list_path = 'env_list_set1'
    if show_baseline:
        print(evaluate_reject_when_full(evaluate_env_list_path))
        print(evaluate_totally_random(evaluate_env_list_path))
    env = produce_env()
    action_dim = 4  
    obs_dim_1 = 45  
    request_dim = 17
    obs_dim_2 = 10
    obs_dim = obs_dim_1+obs_dim_2*7
    encoder = Encoder(input_size=request_dim, output_size=obs_dim_2, \
        use_rnn=False, use_gru=True, use_lstm=False)
    rpm = ReplayMemory(memory_size)  # DQN的经验回放池
    actor = Actor(encoder, obs_size=obs_dim, action_size=action_dim)
    critic = Critic(obs_dim=obs_dim, action_dim=action_dim, encoder=encoder)
    agent = Agent(
        critic=critic,
        actor = actor,
        obs_dim = obs_dim,
        action_dim=action_dim,
        lr=learning_rate,
        gamma=gamma,
        alpha=alpha,
        update_target_steps=update_target_steps)

    if continue_train:
        agent.load(actor_path=actor_path, critic_path=critic_path)

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
    agent.save(actor_path=actor_path, critic_path=critic_path)
if __name__ == '__main__':
    train(show_baseline=False, continue_train=False)
