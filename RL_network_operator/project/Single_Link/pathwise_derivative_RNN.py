import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
import copy
import pickle
import collections
from env_RNN import Environment
from env_evaluate_RNN import EvaluateEnvironment
from request import Request
from request_type import ElasticRequestType, StaticRequestType
from utils import evaluate_RNN, always_accept, always_reject, reject_when_full_RNN, random_decide
OBS_DIM = 11
ACTION_DIM = 2
NUM_LEARN_REQUEST = 10
NUM_TEST_REQUEST = 300
TOTAL_BANDWIDTH = 10
LEARNING_RATE = 0.001 
# clip epsilon
EPSILON = 0.1
# total number of experiments
NUM_ITER = 1000
# for each experiment, simulate NUM_EPISODE trajectories.
NUM_EPISODE = 2
# for each experiment, tuning NUM_EPOCH times
NUM_EPOCH = 2
BATCH_SIZE = 10
GAMMA = 0.5
MEMORY_WARMUP_SIZE = 2
MEMORY_SIZE = 10000
LEARN_FREQ=10

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  
# evaluate banchmark
# evaluate envorinment list
dist1 = StaticRequestType(id=0, bandwidth=2, coming_rate=3, service_rate=1)
dist2 = StaticRequestType(id=1, bandwidth=5, coming_rate=2, service_rate=2)
dist3 = ElasticRequestType(id=2, bandwidth_list=[2,8], coming_rate=1,\
    service_rate=3, distribution=[0.5, 0.5],\
    switch_mu_matrix=[[-0.1, 0.1],[0.1,-0.1]])
dist4 = ElasticRequestType(id=3, bandwidth_list=[2,4,6], coming_rate=4,\
    service_rate=4, distribution=[0.5,0.2,0.3],\
    switch_mu_matrix=[[-0.2,   0.1,  0.1],[0.35, -0.4,  0.05],[0.1,    0.1,  -0.2]])
dist5 = ElasticRequestType(id=4, bandwidth_list=[1,9], coming_rate=3,\
    service_rate=3, distribution=[0.5, 0.5],\
    switch_mu_matrix=[[-0.1, 0.1],[0.1,-0.1]])
request_type_list = [dist1,dist2,dist3, dist4, dist5]
validate_env_dir = open('10-300-5env_list.obj', 'rb')
validate_env_list = pickle.load(validate_env_dir)
test_env_dir = open('10-300-5_test_env_list.obj', 'rb')
test_env_list = pickle.load(test_env_dir)
env = Environment(total_bandwidth = TOTAL_BANDWIDTH,\
    request_type_list = request_type_list, \
    num_of_each_request_type_list=[NUM_LEARN_REQUEST for _ in request_type_list])
# always_accept(env_list=validate_env_list)
# always_reject(validate_env_list)
# reject_when_full_RNN(env_list=validate_env_list)

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, num_layers=2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        out, _ = self.gru(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
        out = self.fc(out)
        # out: (n, 10)
        return out

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size, encoder):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_list.append(s)
            
            action_list.append(a)
            reward_list.append(r)
            next_obs_list.append(s_p)
            done_list.append(done)

        obs_part1_list = [torch.from_numpy(obs[0].astype(np.float32)).to(device) for obs in obs_list]
        obs_part2_list = [torch.from_numpy(obs[1].astype(np.float32)).unsqueeze(0).to(device) for obs in obs_list]

        obs_part2_encode_list = [encoder(obs) for obs in obs_part2_list]
        obs_uniform_list = []
        for obs1, obs2 in zip(obs_part1_list,obs_part2_encode_list):
            obs_uniform_list.append(torch.cat((obs1, obs2),1))
        obs_uniform_torch_tensor = torch.cat(obs_uniform_list, dim=0)
        next_obs_part1_list = [torch.from_numpy(obs[0].astype(np.float32)).to(device) for obs in next_obs_list]
        next_obs_part2_list = [torch.from_numpy(obs[1].astype(np.float32)).unsqueeze(0).to(device) for obs in next_obs_list]

        next_obs_part2_encode_list = [encoder(obs) for obs in next_obs_part2_list]
        next_obs_uniform_list = []
        for obs1, obs2 in zip(next_obs_part1_list,next_obs_part2_encode_list):
            next_obs_uniform_list.append(torch.cat((obs1, obs2),1))
        next_obs_uniform_torch_tensor = torch.cat(next_obs_uniform_list, dim=0)
        action_torch_tensor = torch.tensor(action_list, dtype=torch.int64)
        reward_torch_tensor = torch.tensor(reward_list, dtype=torch.float32)
        done_torch_tensor = torch.tensor(done_list, dtype=torch.float32)

        # return  torch.from_numpy(np.array(obs_batch).astype('float32')), \
        #         torch.from_numpy(np.array(action_batch)), \
        #         torch.from_numpy(np.array(reward_batch).astype('float32')).view(-1,1),\
        #         torch.from_numpy(np.array(next_obs_batch).astype('float32')), \
        #         torch.from_numpy(np.array(done_batch).astype('float32'))
        return obs_uniform_torch_tensor,\
            action_torch_tensor,\
            reward_torch_tensor,\
            next_obs_uniform_torch_tensor,\
            done_torch_tensor

    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(self,
                 actor,
                 critic,
                 encoder,
                 obs_dim,
                 action_dim,
                 lr = 0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.global_step = 0
        # every 200 training steps, coppy model param into target_model
        self.update_target_steps = 50  
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)
        self.optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr)
        self.criteria_critic = nn.MSELoss()
    
    def sample(self, obs):
        with torch.no_grad():
            obs_part1 = torch.from_numpy(obs[0].astype(np.float32)).to(device)
            obs_part2 = torch.from_numpy(obs[1].astype(np.float32)).unsqueeze(0).to(device)
            obs_part2_encoding = self.target_encoder(obs_part2).detach()
            obs = torch.cat((obs_part1, obs_part2_encoding),1)
            # choose action based on prob
            action = np.random.choice(range(2), p=self.target_actor(obs).detach().to('cpu').numpy().reshape(-1,))  
            return action

    def predict(self, obs):  # choose best action
        with torch.no_grad():
            obs_part1 = torch.from_numpy(obs[0].astype(np.float32)).to(device)
            obs_part2 = torch.from_numpy(obs[1].astype(np.float32)).unsqueeze(0).to(device)
            obs_part2_encoding = self.encoder(obs_part2).detach()
            obs = torch.cat((obs_part1, obs_part2_encoding),1)
            return self.actor(obs).argmax(dim=1).detach().item()

    def sync_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def learn(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal):
        # update target model
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        self.global_step %= 500

        # train critic 
        # predict q
        pred_value = (self.critic(batch_obs).clone()*F.one_hot(batch_action)).clone().sum(dim=1).view(-1,1)
        print('pred_value', pred_value.shape)
        with torch.no_grad(): 
            batch_target_action = self.target_actor(batch_next_obs).argmax(dim=1)
            print('batch_target_action', batch_target_action.shape)
            batch_one_hot_target_action = F.one_hot(batch_target_action, num_classes=2)
            print('batch_one_hot_target_action', batch_one_hot_target_action.shape)
            batch_next_q = (batch_one_hot_target_action*self.target_critic(batch_next_obs)).sum(dim=1).view(-1,1)
            print('batch_next_q', batch_next_q.shape)
            # print('mid', (batch_reward.view(-1,1) + (1 - batch_terminal.view(-1,1))).shape)
            # target_value = batch_reward.view(-1,1) + (1 - batch_terminal.view(-1,1))*batch_next_q
            target_value = batch_reward.view(-1,1) + batch_next_q
            print('target_value', target_value.shape)
        loss_critic = self.criteria_critic(pred_value, target_value)
        print('###', loss_critic)
        self.optimizer_critic.zero_grad()
        print('step1')
        loss_critic.backward()
        print('step2')
        self.optimizer_critic.step()

        # train actor
        # find action maximize critic
        pred_action = self.actor(batch_obs)
        print('pred_action', pred_action.shape)
        with torch.no_grad():
            true_one_hot_action = F.one_hot(self.critic(batch_obs).argmax(dim=1),num_classes=2)
        loss_actor = -1.0*(torch.log(pred_action)*true_one_hot_action).sum(dim=1).mean()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        self.optimizer_encoder.step()
        self.optimizer_encoder.zero_grad()

        
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
            (batch_obs, batch_action, batch_reward, batch_next_obs,batch_done) = rpm.sample(BATCH_SIZE, encoder=agent.target_encoder)
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



rpm = ReplayMemory(MEMORY_SIZE) 

actor = PDActor(obs_dim=OBS_DIM,  action_dim=ACTION_DIM).to(device)
best_actor = PDActor(obs_dim=OBS_DIM,  action_dim=ACTION_DIM).to(device)
critic = PDCritirc(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(device)
best_critic = PDCritirc(obs_dim=OBS_DIM, action_dim=ACTION_DIM).to(device)
encoder = Encoder(input_size=5, output_size=5).to(device)
best_encoder = Encoder(input_size=5, output_size=5).to(device)
best_reward = -1000000.0
agent = Agent(
    actor=actor,
    critic=critic,
    encoder=encoder,
    obs_dim = OBS_DIM,
    action_dim=ACTION_DIM)

# preserve some data in replay memory
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 500

# start train
episode = 0
while episode < max_episode: 
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm)
        episode += 1

    print(f'itern{iter}: ', end='')
    avg_reward, avg_acc, avg_static, avg_initial, avg_scale = evaluate_RNN(validate_env_list, agent)
    if avg_reward>best_reward:
        best_actor.load_state_dict(agent.actor.state_dict())
        best_critic.load_state_dict(agent.critic.state_dict())
        best_encoder.load_state_dict(agent.encoder.state_dict())
        best_reward = avg_reward
        print('best model update')
print('Begin Test')
test_agent = Agent(
    actor=best_actor,
    critic=best_critic,
    encoder=best_encoder,
    obs_dim = OBS_DIM,
    action_dim=ACTION_DIM)
print(f'1 gamma={GAMMA} verify it is best on validation set')
reject_when_full_RNN(env_list=validate_env_list)
evaluate_RNN(validate_env_list, test_agent)
print(f'2 gamma={GAMMA}caculate the performance on test set')
reject_when_full_RNN(env_list=test_env_list)
evaluate_RNN(test_env_list, test_agent)