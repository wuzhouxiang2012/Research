import collections
import gym
import random
import torch
from torch import nn
import numpy as np
import copy
import torch.nn.functional as F

# Learning Frequency, do not learn after every step, 
# should learn after accumulating some experience，increase efficiency
LEARN_FREQ = 5  
# replay memory size 
MEMORY_SIZE = 20000 
# save some experience before learn
MEMORY_WARMUP_SIZE = 200 
# sample size from replay memory
BATCH_SIZE = 32 
LEARNING_RATE = 0.001 
# reward decay factor, usually from 0.9 to 0.999
GAMMA = 0.9

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
                 actor,
                 critic,
                 obs_dim,
                 action_dim,
                 lr = 0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.global_step = 0
        # every 200 training steps, coppy model param into target_model
        self.update_target_steps = 200  
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)
        self.criteria_critic = nn.MSELoss()
    
    def sample(self, obs):
        obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        # choose action based on prob
        action = np.random.choice(range(2), p=self.actor(obs).detach().numpy().reshape(-1,))  
        return action

    def predict(self, obs):  # choose best action
        obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
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
        pred_value = (self.critic(batch_obs)* \
            F.one_hot(batch_action, num_classes=self.action_dim))\
                .sum(dim=1).view(-1,1)
        with torch.no_grad(): 
            batch_target_action = self.target_actor(batch_next_obs).argmax(dim=1)
            batch_one_hot_target_action = \
                F.one_hot(batch_target_action, num_classes=self.action_dim)
            batch_next_q = (batch_one_hot_target_action*self.target_critic(batch_next_obs)).sum(dim=1).view(-1,1)
            target_value = batch_reward + (1 - batch_terminal.view(-1,1))*batch_next_q
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


        
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # exploration
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


# evaluate agent, run 5 episodes, return mean reward
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # pick best action
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

class PDActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        hidden_size = 128
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
    def forward(self, obs):
        out = self.fc1(obs)
        out = torch.tanh(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out

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
    rpm = ReplayMemory(MEMORY_SIZE) 

    actor = PDActor(obs_dim=obs_dim,  action_dim=action_dim)
    critic = PDCritirc(obs_dim=obs_dim, action_dim=action_dim)
    agent = Agent(
        actor=actor,
        critic=critic,
        obs_dim = obs_dim,
        action_dim=action_dim)

    # preserve some data in replay memory
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000

    # start train
    episode = 0
    while episode < max_episode: 
        # train part
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1

        # test part
        # render=True show animation result
        eval_reward= evaluate(env, agent, render=True) 
        print('episode:{}  Test reward:{}'.format(episode, eval_reward))



if __name__ == '__main__':
    main()