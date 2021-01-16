import numpy as np
import torch
class PGTorchAgent():
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.algorithm = algorithm

    def sample(self, obs):
        act_prob = self.algorithm.predict(torch.from_numpy(obs))
        act = np.random.choice(range(self.act_dim), p=act_prob.detach().numpy())  # 根据动作概率选取动作
        return act

    def predict(self, obs):
        act_prob = self.algorithm.predict(torch.from_numpy(obs))
        act = np.argmax(act_prob.detach().numpy())  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        return self.algorithm.learn(obs, act, reward)
    
    def restore(self, save_path):
        self.algorithm.model.load_state_dict(torch.load(save_path))

    def save(self, save_path):
        torch.save(self.algorithm.model.state_dict(), save_path)