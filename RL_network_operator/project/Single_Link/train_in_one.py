import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
import copy
import pickle
from env_RNN import Environment
from env_evaluate_RNN import EvaluateEnvironment
from request import Request
from request_type import ElasticRequestType, StaticRequestType
from utils import evaluate_RNN, always_accept, always_reject, reject_when_full_RNN, random_decide

GAMMA_LIST = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for GAMMA in GAMMA_LIST:
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
    NUM_LEARN_REQUEST = 10
    NUM_TEST_REQUEST = 300
    TOTAL_BANDWIDTH = 20
    request_type_list = [dist1,dist2,dist3, dist4, dist5]
    # validate_env_list = []
    # for i in range(5):
    #     validate_env_list.append(EvaluateEnvironment(total_bandwidth = TOTAL_BANDWIDTH,\
    #     request_type_list=request_type_list, \
    #     num_of_each_request_type_list=[NUM_TEST_REQUEST for _ in request_type_list]))
    validate_env_dir = open('20-300-5env_list.obj', 'rb')
    validate_env_list = pickle.load(validate_env_dir)
    test_env_dir = open('20-300-5_test_env_list.obj', 'rb')
    test_env_list = pickle.load(test_env_dir)
    env = Environment(total_bandwidth = TOTAL_BANDWIDTH,\
        request_type_list = request_type_list, \
        num_of_each_request_type_list=[NUM_LEARN_REQUEST for _ in request_type_list])
    always_accept(env_list=validate_env_list)
    always_reject(validate_env_list)
    # random_decide(env_list=validate_env_list)
    reject_when_full_RNN(env_list=validate_env_list)
    # random_decide(env_list=validate_env_list)
    LEARNING_RATE = 0.001 
    # clip epsilon
    EPSILON = 0.1
    # total number of experiments
    NUM_ITER = 1000
    # for each experiment, simulate NUM_EPISODE trajectories.
    NUM_EPISODE = 5
    # for each experiment, tuning NUM_EPOCH times
    NUM_EPOCH = 10
    BATCH_SIZE = 128
    # GAMMA = 0.5
    class Agent():
        def __init__(self,
                    actor,
                    encoder,
                    obs_dim,
                    action_dim,
                    lr = 0.001,
                    epsilon = 0.1):
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.actor = actor
            self.encoder = encoder
            self.target_actor = copy.deepcopy(actor)
            self.target_encoder = copy.deepcopy(encoder)
            self.lr = lr
            self.epsilon = epsilon
            self.global_step = 0
            # every 200 training steps, coppy model param into target_model
            self.update_target_steps = 200  
            self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
            self.optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=lr)

        
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
            self.target_encoder.load_state_dict(self.encoder.state_dict())
        def learn(self, batch_obs, batch_action, batch_adv):
            # update target model
            # if self.global_step % self.update_target_steps == 0:
            #     self.sync_target()
            # self.global_step += 1
            # self.global_step %= 200

            pred_action_distribution = self.actor(batch_obs)
            target_action_distribution = self.target_actor(batch_obs).detach()
            true_action_distribution = nn.functional.one_hot(batch_action, num_classes=2)
            pred_choosed_action_prob = (pred_action_distribution*true_action_distribution).sum(1, keepdim=True)
            target_choosed_action_prob = (target_action_distribution*true_action_distribution).sum(1,keepdim=True)
            J = (pred_choosed_action_prob/target_choosed_action_prob*batch_adv.view(-1,1))
            J_clip = (torch.clamp(pred_choosed_action_prob/target_choosed_action_prob, 1-self.epsilon, 1+self.epsilon)*batch_adv.view(-1,1))
            self.optimizer_actor.zero_grad()
            self.optimizer_encoder.zero_grad()
            loss = -1.0*(torch.cat((J, J_clip), dim=1).min(dim=1)[0]).mean()
            loss.backward()
            self.optimizer_actor.step()
            self.optimizer_encoder.step()
            return loss
        
        def save(self, save_path):
            torch.save(self.actor.state_dict(), save_path)

        def load(self, save_path):
            self.actor.load_state_dict(torch.load(save_path))
            self.sync_target()

    def calc_advantage(reward_list, gamma=1.0):
        for i in range(len(reward_list) - 2, -1, -1):
            # G_i = r_i + gamma * G_i+1
            reward_list[i] += gamma * reward_list[i + 1]  # Gt
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

    class PPODataset(Dataset):
        def __init__(self, obs_list, action_list, advantage_list):
            # self.obs_list = torch.from_numpy(np.array(obs_list).astype(np.float32))
            self.obs_list = obs_list
            # self.action_list = torch.tensor(action_list, dtype=torch.int64)
            self.action_list = action_list
            # self.advantage_list = torch.tensor(advantage_list, dtype=torch.float32)
            self.advantage_list = advantage_list
        def __getitem__(self, index):
            return self.obs_list[index], self.action_list[index], self.advantage_list[index]
        
        def __len__(self):
            return len(self.obs_list)

    def data_iter(dataset, batch_size, encoder, shaffle=False):
        num_examples = len(dataset)
        indices = list(range(num_examples))
        if shaffle:
            random.shuffle(indices)  
        for i in range(0, num_examples, batch_size):
            selected_indices = indices[i: min(i + batch_size, num_examples)]
            obs_list = [dataset.obs_list[idx] for idx in selected_indices]
            obs_part1_list = [torch.from_numpy(obs[0].astype(np.float32)).to(device) for obs in obs_list]
            obs_part2_list = [torch.from_numpy(obs[1].astype(np.float32)).unsqueeze(0).to(device) for obs in obs_list]

            obs_part2_encode_list = [encoder(obs) for obs in obs_part2_list]
            obs_uniform_list = []
            for obs1, obs2 in zip(obs_part1_list,obs_part2_encode_list):
                obs_uniform_list.append(torch.cat((obs1, obs2),1))
            obs_uniform_torch_tensor = torch.cat(obs_uniform_list, dim=0)
            action_list = [dataset.action_list[idx] for idx in selected_indices]
            action_torch_tensor = torch.tensor(action_list, dtype=torch.int64)
            advantage_list = [dataset.advantage_list[idx] for idx in selected_indices]
            advantage_torch_tensor = torch.tensor(advantage_list, dtype=torch.float32)
            yield obs_uniform_torch_tensor, action_torch_tensor, advantage_torch_tensor


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

    action_dim = 2
    obs_dim = 11
    PPOactor = PPO(obs_size=obs_dim, action_size=action_dim).to(device)
    best_actor = PPO(obs_size=obs_dim, action_size=action_dim).to(device)
    encoder = Encoder(input_size=5, output_size=5).to(device)
    best_encoder = Encoder(input_size=5, output_size=5).to(device)
    best_reward = -1000000.0
    agent = Agent(
        actor=PPOactor,
        encoder=encoder,
        obs_dim = obs_dim,
        action_dim=action_dim)
    for iter in range(NUM_ITER):
        #2.1  Using theta k to interact with the env
        # to collect {s_t, a_t} and compute advantage
        # advantage(s_t, a_t) = sum_{t^prime=t}^{T_n}(r_{t^prime}^{n})
        
        all_obs = []
        all_action = []
        all_advantage = []
        for episode in range(NUM_EPISODE):
            obs_list, action_list, reward_list = run_episode(env, agent)
            advantage_list = calc_advantage(reward_list, gamma=GAMMA)
            all_obs.extend(obs_list)
            all_action.extend(action_list)
            all_advantage.extend(advantage_list)
        dataset = PPODataset(obs_list=all_obs, action_list=all_action, advantage_list=all_advantage)
        # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        dataloader = data_iter(dataset=dataset, batch_size=BATCH_SIZE, encoder=agent.encoder)

        # optimize theta
        
        for epoch in range(NUM_EPOCH):
            for i, (batch_obs, batch_action, batch_adv) in enumerate(dataloader):
                batch_obs, batch_action, batch_adv = batch_obs.to(device), batch_action.to(device), batch_adv.to(device)
                agent.learn(batch_obs, batch_action, batch_adv)
        agent.sync_target()
        if iter%50 == 0:
            # eval_reward= evaluate(env_list=validate_env_list, agent=agent) 
            # print('itern:{}  Test reward:{}'.format(iter, eval_reward))
            print(f'itern{iter}: ', end='')
            avg_reward, avg_acc, avg_static, avg_initial, avg_scale = evaluate_RNN(validate_env_list, agent)
            if avg_reward>best_reward:
                best_actor.load_state_dict(agent.actor.state_dict())
                best_encoder.load_state_dict(agent.encoder.state_dict())
                best_reward = avg_reward
                print('best model update')
    # agent.save(model_path)
    print('Begin Test')
    test_agent = Agent(actor=best_actor,encoder=best_encoder,obs_dim = obs_dim,action_dim=action_dim)
    print(f'1 gamma={GAMMA} verify it is best on validation set')
    reject_when_full_RNN(env_list=validate_env_list)
    evaluate_RNN(validate_env_list, test_agent)
    print(f'2 gamma={GAMMA}caculate the performance on test set')
    reject_when_full_RNN(env_list=test_env_list)
    evaluate_RNN(test_env_list, test_agent)
