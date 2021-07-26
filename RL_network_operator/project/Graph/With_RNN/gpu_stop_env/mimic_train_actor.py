from numpy.core.einsumfunc import einsum_path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import random

from Util import evaluate, evaluate_reject_when_full, evaluate_totally_random
from Env_generator import produce_env
from actor import Actor
from encoder import Encoder

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  

class Agent():
    def __init__(self,
                 actor,
                 encoder,
                 obs_dim,
                 action_dim,
                 actor_lr = 0.001,
                 encoder_lr = 0.0001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.encoder = encoder
        # every 200 training steps, coppy model param into target_model
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)

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
            return self.actor(obs).to('cpu').argmax(dim=1).item()

    def learn(self, batch_obs, batch_action):
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

        batch_action = batch_action.to(device)

        pred_action_distribution = self.actor(batch_obs)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred_action_distribution, batch_action)
        self.optimizer_actor.zero_grad()
        self.optimizer_encoder.zero_grad()
        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_encoder.step()
        return loss
    
    def save(self, actor_path, encoder_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.encoder.state_dict(), encoder_path)
    
    def load(self, actor_path, encoder_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.encoder.load_state_dict(torch.load(encoder_path))
    
# run episode for train
def run_episode_baseline(env):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        # choose action based on prob
        choosed_action = 0
        for _action in range(1,4):
            if env.valid_deploy(action=_action):
                choosed_action = _action
                break
        action = choosed_action
        action_list.append(action)
        next_obs, reward, done, _ = env.step(action)
        if done:
            break
        obs = next_obs
    return obs_list, action_list, reward_list

def train(lr=0.001, num_iter=1000, num_episode=10, num_epoch=10, batch_size=32,\
    evaluate_env_list_path='env_list_set1', \
    train_total_time=60, show_baseline=False, \
    continue_train=False, actor_path = 'mimic_actor', encoder_path='mimic_encoder'):
    if show_baseline:
        print(evaluate_reject_when_full(evaluate_env_list_path))
        print(evaluate_totally_random(evaluate_env_list_path))
    env = produce_env(total_time=train_total_time)
    action_dim = 4  
    obs_dim_1 = 45  
    request_dim = 17
    obs_dim_2 = 10
    obs_dim = obs_dim_1+obs_dim_2*7
    encoder = Encoder(input_size=request_dim, output_size=obs_dim_2)
    actor = Actor(obs_size=obs_dim, action_size=action_dim)
    actor.to(device=device)
    agent = Agent(actor=actor, encoder=encoder, obs_dim=obs_dim, action_dim=action_dim)
    if continue_train:
        agent.load(actor_path=actor_path, encoder_path=encoder_path)
    for iter in range(num_iter):
        #2.1  Using theta k to interact with the env
        # to collect {s_t, a_t} and compute advantage
        # advantage(s_t, a_t) = sum_{t^prime=t}^{T_n}(r_{t^prime}^{n})
        
        all_obs = []
        all_action = []
        for episode in range(num_episode):
            obs_list, action_list, _ = run_episode_baseline(env)
            all_obs.extend(obs_list)
            all_action.extend(action_list)

        # optimize theta
        for epoch in range(num_epoch):
            # for i, (batch_obs, batch_action, batch_adv) in enumerate(dataloader):
                # agent.learn(batch_obs, batch_action, batch_adv)
            num_examples = len(all_obs) 
            indices = list(range(num_examples)) 
            random.shuffle(indices)
            
            for i in range(0, num_examples, batch_size):
                
                if i+batch_size<len(all_obs):
                    # print(indice[i:i+batch_size])
                    batch_obs = [all_obs[x] for x in indices[i:i+batch_size]]
                    batch_action = torch.tensor([all_action[x] for x in indices[i:i+batch_size]])
                else:
                    batch_obs = [all_obs[x] for x in indices[i:num_examples]]
                    batch_action = torch.tensor([all_action[x] for x in indices[i:num_examples]])

                agent.learn(batch_obs, batch_action)
        if iter%10 == 0:
            eval_reward= evaluate(evaluate_env_list_path, agent, render=False)  # render=True 查看显示效果
            print('itern:{}  Test reward:{}'.format(iter, eval_reward))
            agent.save(actor_path=actor_path, encoder_path=encoder_path)
        agent.save(actor_path=actor_path, encoder_path=encoder_path)

if __name__ == '__main__':
    train(lr=0.001, num_iter=1000, num_episode=10, num_epoch=10, batch_size=256,\
    evaluate_env_list_path='env_list_set1', \
    train_total_time=60, show_baseline=False, \
    continue_train=False, actor_path = 'mimic_actor', encoder_path='mimic_encoder')




