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
                 obs_dim,
                 action_dim,
                 lr = 0.001,
                 epsilon = 0.1,
                 update_target_steps=200):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.lr = lr
        self.epsilon = epsilon
        self.global_step = 0
        # every 200 training steps, coppy model param into target_model
        self.update_target_steps = update_target_steps  
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    
    def sample(self, obs):
        # obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        # choose action based on prob
        obs = [obs,]
        action = np.random.choice(range(self.action_dim), p=self.target_actor(obs).detach().numpy().reshape(-1,))  
        return action

    def predict(self, obs):  # choose best action
        # obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        obs = [obs,]
        return self.actor(obs).argmax(dim=1).item()

    def sync_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())

    def learn(self, batch_obs, batch_action, batch_adv):
        # update target model
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        self.global_step %= self.update_target_steps

        pred_action_distribution = self.actor(batch_obs)
        target_action_distribution = self.target_actor(batch_obs).detach()
        true_action_distribution = nn.functional.one_hot(batch_action, num_classes=self.action_dim)
        pred_choosed_action_prob = (pred_action_distribution*true_action_distribution).sum(1, keepdim=True)
        target_choosed_action_prob = (target_action_distribution*true_action_distribution).sum(1,keepdim=True)
        J = (pred_choosed_action_prob/target_choosed_action_prob*batch_adv.view(-1,1))
        J_clip = (torch.clamp(pred_choosed_action_prob/target_choosed_action_prob, 1-self.epsilon, 1+self.epsilon)*batch_adv.view(-1,1))
        loss = -1.0*(torch.cat((J, J_clip), dim=1).min(dim=1)[0]).mean()
        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()
        return loss

def calc_advantage(reward_list, gamma=1.0, base_line=0.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + gamma * G_i+1
        reward_list[i] += (gamma * reward_list[i + 1]-1)  # Gt
    return reward_list
    
# run episode for train
def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        # choose action based on prob
        action = agent.sample(obs)
        action_list.append(action)
        next_obs, reward, done, _ = env.step(action)
        reward_list.append(reward)
        if done:
            break
        obs = next_obs
    return obs_list, action_list, reward_list


def train(gamma = 0.9, base_line=0.5, lr=0.001, epsilon=0.1, \
    num_iter=1000, num_episode=10, num_epoch=10, batch_size=128,\
    evaluate_env_list_path='env_list_set1',\
    train_total_time=600, show_baseline=False):
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
    actor = Actor(encoder, obs_size=obs_dim, action_size=action_dim)
    agent = Agent(
        actor=actor,
        obs_dim = obs_dim,
        action_dim=action_dim,
        lr=lr,
        epsilon=epsilon,
        update_target_steps=200)

    for iter in range(num_iter):
        #2.1  Using theta k to interact with the env
        # to collect {s_t, a_t} and compute advantage
        # advantage(s_t, a_t) = sum_{t^prime=t}^{T_n}(r_{t^prime}^{n})
        
        all_obs = []
        all_action = []
        all_advantage = []
        for episode in range(num_episode):
            obs_list, action_list, reward_list = run_episode(env, agent)
            advantage_list = calc_advantage(reward_list, gamma, base_line)
            all_obs.extend(obs_list)
            all_action.extend(action_list)
            all_advantage.extend(advantage_list)
        # dataset = PPODataset(obs_list=all_obs, action_list=all_action, advantage_list=all_advantage)
        # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

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
                    batch_adv = torch.tensor([all_advantage[x] for x in indices[i:i+batch_size]])
                else:
                    batch_obs = [all_obs[x] for x in indices[i:num_examples]]
                    batch_action = torch.tensor([all_action[x] for x in indices[i:num_examples]])
                    batch_adv = torch.tensor([all_advantage[x] for x in indices[i:num_examples]])

                agent.learn(batch_obs, batch_action, batch_adv)
        if iter%10 == 0:
            eval_reward= evaluate(evaluate_env_list_path, agent, render=False)  # render=True 查看显示效果
            print('itern:{}  Test reward:{}'.format(iter, eval_reward))

if __name__ == '__main__':
    train(show_baseline=True)




