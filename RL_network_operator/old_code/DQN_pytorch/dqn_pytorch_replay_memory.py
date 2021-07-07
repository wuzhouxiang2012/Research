import random
import collections
import numpy as np
import torch


class DQNPytorchReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch = [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)

        return torch.from_numpy(np.array(obs_batch).astype('float32')), \
            torch.from_numpy(np.array(action_batch).astype('int')), \
            torch.from_numpy(np.array(reward_batch).astype('float32')),\
            torch.from_numpy(np.array(next_obs_batch).astype('float32'))

    def __len__(self):
        return len(self.buffer)
